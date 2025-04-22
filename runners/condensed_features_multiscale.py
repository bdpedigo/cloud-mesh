# %%
import json
import logging
import os
import sys
import time
import traceback
from functools import partial
from pathlib import Path

import numpy as np
import requests
import toml
import urllib3
from caveclient import CAVEclient
from taskqueue import TaskQueue, queueable

from meshmash import (
    condensed_hks_pipeline,
    find_nucleus_point,
    get_synapse_mapping,
    interpret_path,
    save_condensed_features,
    save_condensed_graph,
    save_id_to_mesh_map,
    scale_mesh,
)

urllib3.disable_warnings()

# suppress warnings for WARNING:urllib3.connectionpool:Connection pool is full...

logging.getLogger("urllib3").setLevel(logging.CRITICAL)

DATASTACK = os.environ.get("DATASTACK", "minnie65_phase3_v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "cautious-fig-thaw")
VERBOSE = str(os.environ.get("VERBOSE", "True")).lower() == "true"
N_JOBS = int(os.environ.get("N_JOBS", -2))
REPLICAS = int(os.environ.get("REPLICAS", 1))
MATERIALIZATION_VERSION = int(os.environ.get("MATERIALIZATION_VERSION", 1300))
QUEUE_NAME = os.environ.get("QUEUE_NAME", "ben-skedit")
RUN = os.environ.get("RUN", False)
REQUEST = os.environ.get("REQUEST", not RUN)
LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL", "ERROR")
LEASE_SECONDS = int(os.environ.get("LEASE_SECONDS", 7200))

logging.basicConfig(level=LOGGING_LEVEL)

logging.getLogger("meshmash").setLevel(level=LOGGING_LEVEL)

# redirect logging to stdout
# logging.getLogger().handlers = []
# logging.getLogger().addHandler(logging.StreamHandler())

import logging
import sys

root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

url_path = Path("~/.cloudvolume/secrets")
url_path = url_path / "discord-secret.json"

with open(url_path.expanduser(), "r") as f:
    URL = json.load(f)["url"]


def replace_none(parameters):
    for key, value in parameters.items():
        if isinstance(value, dict):
            parameters[key] = replace_none(value)
        elif value == "None":
            parameters[key] = None
    return parameters


client = CAVEclient(DATASTACK, version=MATERIALIZATION_VERSION)

cv = client.info.segmentation_cloudvolume(progress=False)

model_folder = Path(__file__).parent.parent / "models" / MODEL_NAME

parameters = toml.load(model_folder / "parameters.toml")
parameters = replace_none(parameters)
scales = eval(parameters["scale_mesh"]["scales"])

path = Path(f"gs://bdp-ssa//{DATASTACK}/{MODEL_NAME}")

cf, _ = interpret_path(path)

feature_out_path = path / "features"
graph_out_path = path / "graphs"
synapse_mappings_path = path / "synapse-mappings"

# %%


@queueable
def run_for_root(root_id):
    if not client.chunkedgraph.is_latest_roots([root_id])[0]:
        logging.info(f"Root {root_id} is not latest, skipping.")
        return None
    try:
        total_time = time.time()
        if cf.exists(f"features/{scales[-1]}/{root_id}.npz"):
            logging.info(f"Features already extracted for {root_id}")
            return None

        logging.info(f"Loading mesh for {root_id}")
        currtime = time.time()
        raw_mesh = cv.mesh.get(root_id, **parameters["cv-mesh-get"])[root_id]
        raw_mesh = (raw_mesh.vertices, raw_mesh.faces)
        mesh_time = time.time() - currtime

        currtime = time.time()
        synapse_mapping = get_synapse_mapping(
            root_id, raw_mesh, client, **parameters["project_points_to_mesh"]
        )
        save_id_to_mesh_map(synapse_mappings_path / f"{root_id}.npz", synapse_mapping)
        logging.info(f"Saved synapse mappings for {root_id}")
        synapse_mapping_time = time.time() - currtime

        try:
            nuc_point = find_nucleus_point(root_id, client)
            if nuc_point is None:
                logging.info(f"No nucleus point found for {root_id}")
        except Exception as e:
            logging.info(f"Error finding nucleus point for {root_id}: {e}")
            nuc_point = None

        for scale in scales:
            mesh = scale_mesh(raw_mesh, scale)
            result = condensed_hks_pipeline(
                mesh,
                nuc_point=nuc_point,
                verbose=VERBOSE,
                n_jobs=N_JOBS,
                **parameters["condensed_hks_pipeline"],
            )

            labels = result.labels
            condensed_features = result.condensed_features

            out_file = feature_out_path / str(scale) / f"{root_id}.npz"
            save_condensed_features(
                out_file,
                condensed_features,
                labels,
                **parameters["save_condensed_features"],
            )
            logging.info(f"Saved features for {root_id}")

            if scale == 1.0:
                graph_file = graph_out_path / f"{root_id}.npz"
                save_condensed_graph(
                    graph_file,
                    np.log(result.condensed_nodes),
                    result.condensed_edges,
                    **parameters["save_condensed_graph"],
                )
                logging.info(f"Saved edges for {root_id}")

                timing_dict = result.timing_info
                timing_dict["root_id"] = str(root_id)
                timing_dict["n_vertices"] = raw_mesh[0].shape[0]
                timing_dict["n_faces"] = raw_mesh[1].shape[0]
                timing_dict["n_condensed_nodes"] = condensed_features.shape[0] - 1
                timing_dict["n_condensed_edges"] = result.condensed_edges.shape[0]
                timing_dict["mesh_time"] = mesh_time
                timing_dict["synapse_mapping_time"] = synapse_mapping_time
                timing_dict["replicas"] = REPLICAS
                timing_dict["n_jobs"] = N_JOBS
                timing_dict["timestamp"] = time.time()
                timing_dict["total_time"] = time.time() - total_time

                cf.put_json(f"timings/{root_id}.json", timing_dict)
                logging.info(f"Saved timings for {root_id}")

    except Exception as e:
        msg = f"Error processing {root_id}"
        msg += "\n\n" + str(e)
        exc_type, exc_value, exc_traceback = sys.exc_info()
        exp = traceback.format_exception(exc_type, exc_value, exc_traceback, limit=50)
        tb = "".join(exp)
        msg += "\n" + str(tb)
        logging.error(msg)
        requests.post(URL, json={"content": msg})


# run_for_root(864691135591548171)

# %%

tq = TaskQueue(f"https://sqs.us-west-2.amazonaws.com/629034007606/{QUEUE_NAME}")


def stop_fn(elapsed_time):
    if elapsed_time > LEASE_SECONDS:
        logging.info("Stopping due to time limit.")
        requests.post(URL, json={"content": "Stopping due to time limit."})
        return True


if RUN:
    tq.poll(lease_seconds=LEASE_SECONDS, verbose=False, tally=False, stop_fn=stop_fn)

# %%

if REQUEST:
    import pandas as pd

    data_folder = Path(__file__).parent.parent / "data"

    labels_df = pd.read_csv(data_folder / "unified_labels.csv")
    # labels_df.query("source == 'vortex_compartment_targets'", inplace=True)
    root_ids = labels_df["root_id"].unique().tolist()
    # custom_roots = [
    #     864691135494786192,
    #     864691135851320007,
    #     864691135940636929,
    #     864691137198691137,
    # ]
    # root_ids += custom_roots
    # request_table = client.materialize.query_table("allen_v1_column_types_slanted_ref")
    # root_ids += (
    #     request_table.query("pt_root_id != 0")
    #     .drop_duplicates("pt_root_id", keep=False)["pt_root_id"]
    #     .unique()
    #     .tolist()
    # )
    # tasks = [partial(run_for_root, root_id) for root_id in root_ids]

    # tq.insert(tasks)

# %%


# %%
# TODO add this in
# lookup = client.materialize.query_view("nucleus_detection_lookup_v1").set_index("id")
# # %%
# lookup.loc[request_table["id_ref"]]

dones = list(cf.list("features/1.5"))
dones = np.array(
    [int(done.split("/")[-1].split(".")[0]) for done in dones if done.endswith(".npz")]
)
diff = np.setdiff1d(root_ids, dones)
print(len(diff))

# %%
out = labels_df.query("root_id.isin(@diff)")["nucleus_id"].unique()

# %%
diff = client.materialize.query_table(
    "nucleus_detection_v0", filter_in_dict=dict(id=out)
)['pt_root_id'].unique()

#%%
diff = np.setdiff1d(diff, dones)
print(len(diff))
print(diff)

# %%
VERBOSE = True
for root_id in diff:
    print(root_id)
    run_for_root(root_id)

# %%
missing_ids = np.setdiff1d(root_ids, dones)
tasks = [partial(run_for_root, root_id) for root_id in missing_ids]
