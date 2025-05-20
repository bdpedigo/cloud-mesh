# %%
import json
import logging
import os
import platform
import sys
import time
import traceback
from functools import partial
from pathlib import Path

import numpy as np
import requests
import toml
import urllib3
from caveclient import CAVEclient, set_session_defaults
from cloudvolume import CloudVolume
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

SYSTEM = platform.system()

urllib3.disable_warnings()

# suppress warnings for WARNING:urllib3.connectionpool:Connection pool is full...

logging.getLogger("urllib3").setLevel(logging.CRITICAL)

DATASTACK = os.environ.get("DATASTACK", "minnie65_phase3_v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "absolute-solo-yak")
VERBOSE = str(os.environ.get("VERBOSE", "True")).lower() == "true"
N_JOBS = int(os.environ.get("N_JOBS", -2))
REPLICAS = int(os.environ.get("REPLICAS", 1))
MATERIALIZATION_VERSION = int(os.environ.get("MATERIALIZATION_VERSION", 1300))
QUEUE_NAME = os.environ.get("QUEUE_NAME", "ben-skedit")
RUN = os.environ.get("RUN", False)
REQUEST = os.environ.get("REQUEST", not RUN)
LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL", "ERROR")
LEASE_SECONDS = int(os.environ.get("LEASE_SECONDS", 7200))
RECOMPUTE = os.environ.get("RECOMPUTE", False)
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", 5))
BACKOFF_FACTOR = int(os.environ.get("BACKOFF_FACTOR", 4))
BACKOFF_MAX = int(os.environ.get("BACKOFF_MAX", 240))
MAX_RUNS = int(os.environ.get("MAX_RUNS", 5))

logging.basicConfig(level=LOGGING_LEVEL)

logging.getLogger("meshmash").setLevel(level=LOGGING_LEVEL)

# redirect logging to stdout
# logging.getLogger().handlers = []
# logging.getLogger().addHandler(logging.StreamHandler())

if SYSTEM == "Darwin":
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = root.handlers[0]
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)


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


model_folder = Path(__file__).parent.parent / "models" / MODEL_NAME

parameters = toml.load(model_folder / "parameters.toml")
parameters = replace_none(parameters)

# %%


@queueable
def run_for_root(root_id, datastack, version, scale=1.0, track_synapses="both"):
    path = Path(f"gs://bdp-ssa//{datastack}/{MODEL_NAME}")

    cf, _ = interpret_path(path)

    if scale == 1.0 and cf.exists(f"features/{root_id}.npz") and not RECOMPUTE:
        logging.info(f"Features already extracted for {root_id}")
        return None
    elif (
        scale != 1.0 and cf.exists(f"features/{scale}/{root_id}.npz") and not RECOMPUTE
    ):
        logging.info(f"Features already extracted for {root_id} at scale {scale}")
        return None

    set_session_defaults(
        max_retries=MAX_RETRIES, backoff_factor=BACKOFF_FACTOR, backoff_max=BACKOFF_MAX
    )
    client = CAVEclient(datastack, version=version)

    if not client.chunkedgraph.is_latest_roots([root_id])[0]:
        logging.info(f"Root {root_id} is not latest, skipping.")
        return None

    feature_out_path = path / "features"
    if scale != 1.0:
        feature_out_path = feature_out_path / str(scale)
    graph_out_path = path / "graphs"
    pre_synapse_mappings_path = path / "pre-synapse-mappings"
    post_synapse_mappings_path = path / "post-synapse-mappings"

    track_post_synapses = False
    track_pre_synapses = False
    if track_synapses == "both":
        track_post_synapses = True
        track_pre_synapses = True
    elif track_synapses == "post":
        track_post_synapses = True
    elif track_synapses == "pre":
        track_pre_synapses = True

    try:
        total_time = time.time()
        logging.info(f"Loading mesh for {root_id}")
        currtime = time.time()
        if datastack == "zheng_ca3":
            cv = CloudVolume(
                "gs://zheng_mouse_hippocampus_production/v2/seg_m195", progress=False
            )
            raw_mesh = cv.mesh.get(root_id)[root_id]
            raw_mesh = raw_mesh.deduplicate_vertices(is_chunk_aligned=True)
            raw_mesh = (raw_mesh.vertices, raw_mesh.faces)
            # new_vertices, new_faces = pcu.deduplicate_mesh_vertices(
            #     raw_mesh[0],
            #     raw_mesh[1].astype("int32"),
            #     epsilon=1e-12,
            #     return_index=False,
            # )
            # raw_mesh = (new_vertices, new_faces)
        else:
            cv = client.info.segmentation_cloudvolume(progress=False)
            raw_mesh = cv.mesh.get(root_id, **parameters["cv-mesh-get"])[root_id]
            raw_mesh = (raw_mesh.vertices, raw_mesh.faces)
        mesh_time = time.time() - currtime
        logging.info(f"Loaded mesh for {root_id}, has {raw_mesh[0].shape[0]} vertices")

        if scale != 1.0:
            raw_mesh = scale_mesh(raw_mesh, scale)

        currtime = time.time()
        if track_post_synapses:
            post_synapse_mapping = get_synapse_mapping(
                root_id,
                raw_mesh,
                client,
                side="post",
                **parameters["project_points_to_mesh"],
            )
            if len(post_synapse_mapping) == 0:
                logging.info(f"No synapse mapping found for {root_id}")
            else:
                save_id_to_mesh_map(
                    pre_synapse_mappings_path / f"{root_id}.npz", post_synapse_mapping
                )
                logging.info(
                    f"Saved {len(post_synapse_mapping)} synapse mappings for {root_id}"
                )

        if track_pre_synapses:
            pre_synapse_mapping = get_synapse_mapping(
                root_id,
                raw_mesh,
                client,
                side="pre",
                **parameters["project_points_to_mesh"],
            )
            if len(pre_synapse_mapping) == 0:
                logging.info(f"No synapse mapping found for {root_id}")
            else:
                save_id_to_mesh_map(
                    post_synapse_mappings_path / f"{root_id}.npz", pre_synapse_mapping
                )
                logging.info(
                    f"Saved {len(pre_synapse_mapping)} synapse mappings for {root_id}"
                )

        synapse_mapping_time = time.time() - currtime

        try:
            nuc_point = find_nucleus_point(root_id, client, update_root_id="check")
            if nuc_point is None:
                logging.info(f"No nucleus point found for {root_id}")
            else:
                logging.info(f"Found nucleus point for {root_id} at {nuc_point[0]}")
        except Exception as e:
            logging.info(f"Error finding nucleus point for {root_id}: {e}")
            nuc_point = None

        result = condensed_hks_pipeline(
            raw_mesh,
            nuc_point=nuc_point,
            verbose=VERBOSE,
            n_jobs=N_JOBS,
            **parameters["condensed_hks_pipeline"],
        )

        labels = result.labels
        condensed_features = result.condensed_features

        out_file = feature_out_path / f"{root_id}.npz"
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
                result.condensed_nodes,
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
            timing_dict["system"] = SYSTEM

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


# %%

tq = TaskQueue(f"https://sqs.us-west-2.amazonaws.com/629034007606/{QUEUE_NAME}")


# def stop_fn(elapsed_time):
#     if elapsed_time > LEASE_SECONDS:
#         logging.info("Stopping due to time limit.")
#         requests.post(URL, json={"content": "Stopping due to time limit."})
#         return True


def stop_fn(executed):
    if executed >= MAX_RUNS:
        quit()


if RUN:
    tq.poll(lease_seconds=LEASE_SECONDS, verbose=False, tally=False, stop_fn=stop_fn)

# %%

if REQUEST:
    import pandas as pd
    from cloudfiles import CloudFiles

    tasks = []
    if False:
        cell_table = pd.read_feather(
            "/Users/ben.pedigo/code/meshrep/cloud-mesh/data/v1dd_single_neuron_soma_ids.feather"
        )

        root_ids = np.unique(cell_table["pt_root_id"])

        cf = CloudFiles(f"gs://bdp-ssa/v1dd/{MODEL_NAME}")
        done_files = list(cf.list("features"))
        done_roots = [
            int(file.split("/")[-1].split(".")[0])
            for file in done_files
            if file.endswith(".npz")
        ]
        root_ids = np.setdiff1d(root_ids, done_roots)

        tasks += [partial(run_for_root, root_id, "v1dd", 974) for root_id in root_ids]

    if False:
        datastack = "minnie65_phase3_v1"
        version = 1300
        client = CAVEclient(datastack_name=datastack, version=version)

        table = pd.read_csv(
            "/Users/ben.pedigo/code/meshrep/meshrep/experiments/cautious-fig-thaw/labels.csv"
        )
        root_ids = table["pt_root_id"].unique()
        # column_table = client.materialize.query_table(
        #     "allen_v1_column_types_slanted_ref"
        # )
        # root_ids = column_table.query("pt_root_id != 0")["pt_root_id"].unique()
        tasks += [
            partial(run_for_root, root_id, datastack, version, track_synapses="both")
            for root_id in root_ids
        ]

    if True:
        datastack = "zheng_ca3"
        version = 195
        table = pd.read_csv(
            "/Users/ben.pedigo/code/meshrep/cloud-mesh/data/zheng_ca3/updated_segids_20250423_manual.csv",
            index_col=0,
        )
        root_ids = table["seg_m195_20250423"].unique()

        # tasks += [
        #     partial(
        #         run_for_root,
        #         root_id,
        #         datastack,
        #         version,
        #         scale=1.0,
        #         track_synapses=False,
        #     )
        #     for root_id in root_ids
        # ]
        tasks += [
            partial(
                run_for_root,
                root_id,
                datastack,
                version,
                scale=0.75,
                track_synapses=False,
            )
            for root_id in root_ids
        ]

    if len(tasks) > 0:
        tq.insert(tasks)

    # data_folder = Path(__file__).parent.parent / "data"

    # labels_df = pd.read_csv(data_folder / "unified_labels.csv")
    # labels_df.query("source == 'vortex_compartment_targets'", inplace=True)
    # root_ids = labels_df["root_id"].unique().tolist()
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

    #

# %%
run_for_root(
    648518346442090245,
    "zheng_ca3",
    195,
    scale=1.0,
    track_synapses=False,
)

# %%
