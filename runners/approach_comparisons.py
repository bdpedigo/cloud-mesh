# %%
import json
import logging
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import toml
import urllib3
from caveclient import CAVEclient
from taskqueue import TaskQueue, queueable

from meshmash import (
    chunked_hks_pipeline2,
    find_nucleus_point,
    get_synapse_mapping,
    interpret_path,
    save_condensed_features,
    save_condensed_graph,
    save_id_to_mesh_map,
)

# suppress warnings for WARNING:urllib3.connectionpool:Connection pool is full...
urllib3.disable_warnings()
logging.getLogger("urllib3").setLevel(logging.CRITICAL)


VERBOSE = str(os.environ.get("VERBOSE", "False")).lower() == "true"
print("VERBOSE:", VERBOSE)
N_JOBS = int(os.environ.get("N_JOBS", 1))
REPLICAS = int(os.environ.get("REPLICAS", 1))
MATERIALIZATION_VERSION = int(os.environ.get("MATERIALIZATION_VERSION", 1300))
QUEUE_NAME = os.environ.get("QUEUE_NAME", "ben-skedit")
RUN = os.environ.get("RUN", True)
REQUEST = os.environ.get("REQUEST", False)
LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL", "ERROR")
LEASE_SECONDS = int(os.environ.get("LEASE_SECONDS", 12 * 3600))

logging.basicConfig(level=LOGGING_LEVEL)
logging.getLogger("meshmash").setLevel(level=LOGGING_LEVEL)
# redirect logging to stdout
# logging.getLogger().handlers = []
# logging.getLogger().addHandler(logging.StreamHandler())

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

models_folder = Path(__file__).parent.parent / "models"

base_path = Path("gs://bdp-ssa/comparisons/")

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


@queueable
def run_for_root(root_id, model_name):
    if VERBOSE: 
        print("Working on root_id:", root_id, "model_name:", model_name)
    model_out_path = base_path / model_name

    feature_out_path = base_path / "features"
    edge_out_path = base_path / "edges"
    synapse_mappings_path = base_path / "synapse-mappings"

    cf, _ = interpret_path(model_out_path)

    model_folder = models_folder / model_name
    parameters = toml.load(model_folder / "parameters.toml")
    parameters = replace_none(parameters)

    try:
        logging.info(f"Loading mesh for {root_id}")
        currtime = time.time()
        raw_mesh = cv.mesh.get(root_id, **parameters["cv-mesh-get"])[root_id]
        raw_mesh = (raw_mesh.vertices, raw_mesh.faces)
        mesh_time = time.time() - currtime

        currtime = time.time()
        logging.info(f"Mapping synapses for {root_id}")
        synapse_mapping = get_synapse_mapping(
            root_id, raw_mesh, client, **parameters["project_points_to_mesh"]
        )
        save_id_to_mesh_map(synapse_mappings_path / f"{root_id}.npz", synapse_mapping)
        logging.info(f"Saved synapse mappings for {root_id}")
        synapse_mapping_time = time.time() - currtime

        logging.info(f"Finding nucleus point for {root_id}")
        try:
            nuc_point = find_nucleus_point(root_id, client)
            if nuc_point is None:
                logging.info(f"No nucleus point found for {root_id}")
        except Exception as e:
            msg = f"Error finding nucleus point for {root_id}: {e}"
            logging.info(msg)
            msg += "\n" + str(e.__traceback__)
            requests.post(URL, json={"content": msg})
            nuc_point = None

        logging.info(f"Running chunked HKS pipeline for {root_id}")
        result = chunked_hks_pipeline2(
            raw_mesh,
            query_indices=None,
            nuc_point=nuc_point,
            verbose=VERBOSE,
            n_jobs=N_JOBS,
            **parameters["chunked_hks_pipeline"],
        )

        labels = result.labels
        condensed_features = result.condensed_features
        condensed_nodes = result.condensed_nodes
        condensed_edges = result.condensed_edges

        # for consistency with the rest of the pipeline, make sure null label is present
        assert -1 in condensed_features.index

        # log transform to avoid overflow
        condensed_nodes[non_hks_features] = np.log(condensed_nodes[non_hks_features])

        out_file = feature_out_path / f"{root_id}.npz"

        save_condensed_features(
            out_file,
            condensed_features,
            labels,
            **parameters["save_condensed_features"],
        )
        logging.info(f"Saved features for {root_id}")

        graph_file = edge_out_path / f"{root_id}.npz"
        save_condensed_graph(
            graph_file,
            condensed_nodes,
            condensed_edges,
            **parameters["save_condensed_graph"],
        )
        logging.info(f"Saved graph for {root_id}")

        timing_dict = result.timing_info
        timing_dict["root_id"] = str(root_id)
        timing_dict["n_vertices"] = raw_mesh[0].shape[0]
        timing_dict["n_faces"] = raw_mesh[1].shape[0]
        timing_dict["n_condensed_nodes"] = condensed_features.shape[0] - 1
        timing_dict["n_condensed_edges"] = condensed_edges.shape[0]
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
data_folder = Path(__file__).parent.parent / "data"

labels_df = pd.read_csv(data_folder / "unified_labels.csv")
labels_df.query("source == 'vortex_compartment_targets'", inplace=True)
root_ids = labels_df["root_id"].unique()

model_names = ["shiny-wolf-message", "shiny-wolf-message-strawman"]
if REQUEST:
    tasks = []
    for root_id in root_ids:
        for model_name in model_names:
            tasks.append(partial(run_for_root, root_id, model_name))

    tq.insert(tasks)

# %%
