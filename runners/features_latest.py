# %%
import json
import logging
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import requests
import toml
import urllib3
from caveclient import CAVEclient
from taskqueue import TaskQueue, queueable

from meshmash import (
    chunked_hks_pipeline,
    find_nucleus_point,
    get_synapse_mapping,
    interpret_path,
    save_condensed_edges,
    save_condensed_features,
    save_id_to_mesh_map,
)


urllib3.disable_warnings()

# suppress warnings for WARNING:urllib3.connectionpool:Connection pool is full...

logging.getLogger("urllib3").setLevel(logging.CRITICAL)


MODEL_NAME = os.environ.get("MODEL_NAME", "foggy-forest-call")
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
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler())

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


client = CAVEclient("minnie65_phase3_v1", version=MATERIALIZATION_VERSION)

cv = client.info.segmentation_cloudvolume(progress=False)

model_folder = Path(__file__).parent.parent / "models" / MODEL_NAME

parameters = toml.load(model_folder / "parameters.toml")
parameters = replace_none(parameters)

path = Path(f"gs://bdp-ssa/minnie/{MODEL_NAME}")

cf, _ = interpret_path(path)

feature_out_path = path / "features"
edge_out_path = path / "edges"
synapse_mappings_path = path / "synapse-mappings"

non_hks_features = [
    "x",
    "y",
    "z",
    "area",
    "n_vertices",
    "component_area",
    "component_n_vertices",
    "distance_to_nucleus",
]

# %%
bad_root = 864691136125099814
raw_mesh = cv.mesh.get(bad_root, **parameters["cv-mesh-get"])[bad_root]
raw_mesh = (raw_mesh.vertices, raw_mesh.faces)
result = chunked_hks_pipeline(
    raw_mesh,
    query_indices=None,
    verbose=VERBOSE,
    n_jobs=N_JOBS,
    **parameters["chunked_hks_pipeline"],
)

# %%


@queueable
def run_for_root(root_id):
    try:
        needs_features = True
        needs_edges = True
        needs_synapse_mappings = True

        if cf.exists(f"features/{root_id}.npz"):
            logging.info(f"Features already extracted for {root_id}")
            needs_features = False
        else:
            logging.info(f"Extracting features for {root_id}")

        if cf.exists(f"edges/{root_id}.npz"):
            logging.info(f"Edges already extracted for {root_id}")
            needs_edges = False
        else:
            logging.info(f"Extracting edges for {root_id}")

        if cf.exists(f"synapse-mappings/{root_id}.npz"):
            logging.info(f"Synapse mappings already extracted for {root_id}")
            needs_synapse_mappings = False
        else:
            logging.info(f"Extracting synapse mappings for {root_id}")

        mesh_time = None
        if any([needs_features, needs_edges, needs_synapse_mappings]):
            logging.info(f"Loading mesh for {root_id}")
            currtime = time.time()
            raw_mesh = cv.mesh.get(root_id, **parameters["cv-mesh-get"])[root_id]
            raw_mesh = (raw_mesh.vertices, raw_mesh.faces)
            mesh_time = time.time() - currtime

        synapse_mapping_time = None
        if needs_synapse_mappings:
            currtime = time.time()
            synapse_mapping = get_synapse_mapping(
                root_id, raw_mesh, client, **parameters["project_points_to_mesh"]
            )
            save_id_to_mesh_map(
                synapse_mappings_path / f"{root_id}.npz", synapse_mapping
            )
            logging.info(f"Saved synapse mappings for {root_id}")
            synapse_mapping_time = time.time() - currtime

        if needs_features or needs_edges:
            try:
                nuc_point = find_nucleus_point(root_id, client)
                if nuc_point is None:
                    logging.info(f"No nucleus point found for {root_id}")
            except Exception as e:
                logging.info(f"Error finding nucleus point for {root_id}: {e}")
                nuc_point = None

            result = chunked_hks_pipeline(
                raw_mesh,
                query_indices=None,
                nuc_point=nuc_point,
                verbose=VERBOSE,
                n_jobs=N_JOBS,
                **parameters["chunked_hks_pipeline"],
            )

            labels = result.labels
            condensed_features = result.condensed_features

            # for consistency with the rest of the pipeline, make sure null label is present
            assert -1 in condensed_features.index

            # log transform to avoid overflow
            condensed_features[non_hks_features] = np.log(
                condensed_features[non_hks_features]
            )

            out_file = feature_out_path / f"{root_id}.npz"
            edge_file = edge_out_path / f"{root_id}.npz"
            save_condensed_features(
                out_file,
                condensed_features,
                labels,
                **parameters["save_condensed_features"],
            )
            logging.info(f"Saved features for {root_id}")

            save_condensed_edges(edge_file, result.condensed_edges)
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

            cf.put_json(f"timings/{root_id}.json", timing_dict)
            logging.info(f"Saved timings for {root_id}")

    except Exception as e:
        msg = f"Error processing {root_id}"
        msg += "\n" + str(e)
        logging.error(msg)
        msg += "\n" + str(e.__traceback__)
        requests.post(URL, json={"content": msg})


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
    request_table = client.materialize.query_table("allen_v1_column_types_slanted_ref")
    root_ids = (
        request_table.query("pt_root_id != 0")
        .drop_duplicates("pt_root_id", keep=False)["pt_root_id"]
        .unique()
    )
    tasks = [partial(run_for_root, root_id) for root_id in root_ids]

    # tq.insert(tasks)

# %%
# TODO add this in
# lookup = client.materialize.query_view("nucleus_detection_lookup_v1").set_index("id")
# # %%
# lookup.loc[request_table["id_ref"]]

dones = list(cf.list("features"))
dones = [int(d.split(".")[0].split("/")[1]) for d in dones if ".npz" in d]

# %%
missing_ids = np.setdiff1d(root_ids, dones)
tasks = [partial(run_for_root, root_id) for root_id in missing_ids]
# %%
# tq.insert(tasks)

# %%
print('here')
run_for_root(864691136125099814)

#%%
run_for_root(864691135489514810)


# %%
