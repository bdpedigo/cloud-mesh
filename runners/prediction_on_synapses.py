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
import pandas as pd
import requests
import toml
import urllib3
from caveclient import CAVEclient, set_session_defaults
from cloudfiles import CloudFiles
from joblib import load
from taskqueue import TaskQueue, queueable

from cloudigo import exists, get_dataframe, put_dataframe
from meshmash import (
    condensed_hks_pipeline,
    find_nucleus_point,
    get_label_components,
    get_synapse_mapping,
    read_condensed_features,
    read_id_to_mesh_map,
    save_condensed_features,
    save_condensed_graph,
    save_id_to_mesh_map,
)

SYSTEM = platform.system()

urllib3.disable_warnings()

# suppress warnings for WARNING:urllib3.connectionpool:Connection pool is full...

logging.getLogger("urllib3").setLevel(logging.CRITICAL)

DATASTACK = os.environ.get("DATASTACK", "minnie65_phase3_v1")
PARAMETER_NAME = os.environ.get("PARAMETER_NAME", "absolute-solo-yak")
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
MAX_RUNS = int(os.environ.get("MAX_RUNS", 10))


logging.basicConfig(level=LOGGING_LEVEL)

logging.getLogger("meshmash").setLevel(level=LOGGING_LEVEL)

# redirect logging to stdout
# logging.getLogger().handlers = []
# logging.getLogger().addHandler(logging.StreamHandler())

if SYSTEM == "Darwin":
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = root.handlers[0]
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
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


model_folder = Path(__file__).parent.parent / "models" / PARAMETER_NAME
model_path = model_folder / "hks_model_calibrated.joblib"
model = load(model_path)

parameters = toml.load(model_folder / "parameters.toml")
parameters = replace_none(parameters)


# %%


@queueable
def run_for_root(root_id, datastack, version):
    set_session_defaults(
        max_retries=MAX_RETRIES, backoff_factor=BACKOFF_FACTOR, backoff_max=BACKOFF_MAX
    )
    client = CAVEclient(datastack, version=version)

    if not client.chunkedgraph.is_latest_roots([root_id])[0]:
        logging.info(f"Root {root_id} is not latest, skipping.")
        return None

    # cv = client.info.segmentation_cloudvolume(progress=False)

    # path = Path(f"gs://bdp-ssa//{datastack}/{PARAMETER_NAME}")

    # cf, _ = interpret_path(path)

    # feature_path = path / "features"
    # # graph_out_path = path / "graphs"
    # synapse_mappings_path = path / "synapse-mappings"

    try:
        synapse_prediction_path = (
            f"gs://bdp-ssa/v1dd/{PARAMETER_NAME}/synapse-predictions/{root_id}.csv.gz"
        )

        if exists(synapse_prediction_path):
            logging.info(f"Synapse predictions already exist for {root_id}")

        logging.info(f"Loading features for {root_id}, {datastack}")
        currtime = time.time()
        feature_path = Path(
            f"gs://bdp-ssa/{datastack}/{PARAMETER_NAME}/features/{root_id}.npz"
        )
        condensed_features, condensed_ids = read_condensed_features(feature_path)
        elapsed_time = time.time() - currtime
        logging.info(
            f"Loaded features for {root_id}, {datastack}, took {elapsed_time:.2f} seconds"
        )

        logging.info(f"Performing prediction for {root_id}, {datastack}")
        currtime = time.time()
        # NOTE: add any joining of features that needs to happen here
        condensed_features = condensed_features[model.feature_names_in_]

        condensed_predictions = model.predict(condensed_features)
        condensed_predictions = pd.Series(
            condensed_predictions, index=condensed_features.index, name="pred_label"
        )
        condensed_predictions.loc[-1] = None

        condensed_posteriors = model.predict_proba(condensed_features)
        condensed_posteriors = pd.DataFrame(
            condensed_posteriors, index=condensed_features.index, columns=model.classes_
        )
        condensed_posteriors.loc[-1] = None

        condensed_posterior_entropy = -np.sum(
            condensed_posteriors * np.log(condensed_posteriors), axis=1
        )

        synapse_mapping_path = Path(
            f"gs://bdp-ssa/{datastack}/{PARAMETER_NAME}/synapse-mappings/{root_id}.npz"
        )

        synapse_mapping = read_id_to_mesh_map(synapse_mapping_path)
        synapse_mapping = pd.Series(
            index=synapse_mapping[:, 0], data=synapse_mapping[:, 1]
        )

        synapse_predictions = condensed_predictions.loc[
            condensed_ids[synapse_mapping]
        ].values
        synapse_predictions = pd.Series(
            synapse_predictions, index=synapse_mapping.index, name="pred_label"
        )

        synapse_posterior_max = (
            condensed_posteriors.loc[condensed_ids[synapse_mapping]].max(axis=1).values
        )
        synapse_posterior_max = pd.Series(
            synapse_posterior_max, index=synapse_mapping.index, name="posterior_max"
        )

        synapse_posterior_entropy = condensed_posterior_entropy.loc[
            condensed_ids[condensed_ids[synapse_mapping]]
        ].values
        synapse_posterior_entropy = pd.Series(
            synapse_posterior_entropy,
            index=synapse_mapping.index,
            name="posterior_entropy",
        )
        logging.info(
            f"Ran prediciton for {root_id}, {datastack}, took {time.time() - currtime:.2f} seconds"
        )

        # TODO this could be generalized to be done w/o mesh given the compressed graph,
        # I think.
        logging.info(f"Loading mesh for {root_id}, {datastack}")
        currtime = time.time()
        cv = client.info.segmentation_cloudvolume(progress=False)
        raw_mesh = cv.mesh.get(root_id, **parameters["cv-mesh-get"])[root_id]
        mesh = (raw_mesh.vertices, raw_mesh.faces)
        elapsed_time = time.time() - currtime
        logging.info(
            f"Loaded mesh for {root_id}, {datastack}, has {mesh[0].shape[0]} vertices, took {elapsed_time:.2f} seconds"
        )

        currtime = time.time()
        logging.info(f"Getting label components for {root_id}, {datastack}")
        mesh_predictions = condensed_predictions.loc[condensed_ids]

        label_components = get_label_components(mesh, mesh_predictions)
        synapse_label_components = label_components[synapse_mapping]
        synapse_label_components = pd.Series(
            synapse_label_components,
            index=synapse_mapping.index,
            name="component_id",
        )

        logging.info(
            f"Got label components for {root_id}, {datastack}, took {time.time() - currtime:.2f} seconds"
        )

        currtime = time.time()
        logging.info(f"Saving synapse predictions for {root_id}, {datastack}")
        synapse_summary = pd.concat(
            [
                synapse_predictions,
                synapse_posterior_max,
                synapse_posterior_entropy,
                synapse_label_components,
            ],
            axis=1,
        )
        synapse_summary.rename_axis("id", inplace=True)
        synapse_summary.dropna(how="any", inplace=True)

        component_counts = (
            synapse_summary.query('pred_label == "spine"')
            .groupby("component_id")
            .size()
            .sort_values(ascending=False)
        )
        multi_components = component_counts[component_counts > 1].index  # noqa
        remove_indices = synapse_summary.query(
            "~component_id.isin(@multi_components)"
        ).index
        synapse_summary.loc[remove_indices, "component_id"] = -1
        synapse_summary.rename(columns={"component_id": "multispine_id"}, inplace=True)

        put_dataframe(synapse_summary, synapse_prediction_path)

        logging.info(
            f"Saved synapse predictions for {root_id}, {datastack}, took {time.time() - currtime:.2f} seconds"
        )

        # synapses["pred_label"] = predictions_df["pred_label"]
        # synapses["component_id"] = synapses["mesh_index"].map(label_components.__getitem__)
        # component_counts = (
        #     synapses.query("pred_label == 'spine'").groupby("component_id").size()
        # )
        # synapses["spine_component_count"] = (
        #     synapses["component_id"].map(component_counts).fillna(0).astype(int)
        # )
        # multispine_synapses = synapses[synapses["spine_component_count"] > 1]
        # multispine_synapses = multispine_synapses[["component_id", "spine_component_count"]]
        # multispine_synapses["root_id"] = root_id

    except Exception as e:
        msg = f"Error processing {root_id}"
        msg += "\n\n" + str(e)
        exc_type, exc_value, exc_traceback = sys.exc_info()
        exp = traceback.format_exception(exc_type, exc_value, exc_traceback, limit=50)
        tb = "".join(exp)
        msg += "\n" + str(tb)
        logging.error(msg)
        if SYSTEM != "Darwin":
            requests.post(URL, json={"content": msg})

    return

    try:
        total_time = time.time()
        if cf.exists(f"features/{root_id}.npz") and not RECOMPUTE:
            logging.info(f"Features already extracted for {root_id}")
            return None

        logging.info(f"Loading mesh for {root_id}")
        currtime = time.time()
        raw_mesh = cv.mesh.get(root_id, **parameters["cv-mesh-get"])[root_id]
        raw_mesh = (raw_mesh.vertices, raw_mesh.faces)
        mesh_time = time.time() - currtime
        logging.info(f"Loaded mesh for {root_id}, has {raw_mesh[0].shape[0]} vertices")

        currtime = time.time()
        synapse_mapping = get_synapse_mapping(
            root_id, raw_mesh, client, **parameters["project_points_to_mesh"]
        )
        if len(synapse_mapping) == 0:
            logging.info(f"No synapse mapping found for {root_id}")
        else:
            save_id_to_mesh_map(
                synapse_mappings_path / f"{root_id}.npz", synapse_mapping
            )
            logging.info(f"Saved {len(synapse_mapping)} synapse mappings for {root_id}")
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


root_id = 864691132533489754
root_id = 864691132533646426

# %%
run_for_root(root_id, "v1dd", 974)

# %%
synapse_summary = get_dataframe(
    f"gs://bdp-ssa/v1dd/absolute-solo-yak/synapse-predictions/{root_id}.csv.gz",
    index_col=0,
)
synapse_summary
# %%

client = CAVEclient("v1dd", version=974)
synapse_table = client.materialize.synapse_query(
    post_ids=root_id,
    desired_resolution=[1, 1, 1],
)
synapse_table["pred_label"] = synapse_table["id"].map(synapse_summary["pred_label"])

# %%

synapse_groups = (
    synapse_summary.reset_index(drop=False)
    .query("multispine_id != -1")
    .groupby("multispine_id")["id"]
    .unique()
)

import networkx as nx

synapse_pairs = []
for multispine_id, synapse_ids in synapse_groups.items():
    for source, target in nx.utils.pairwise(synapse_ids):
        synapse_pairs.append((multispine_id, source, target))
synapse_pairs = pd.DataFrame(
    synapse_pairs, columns=["multispine_id", "source", "target"]
)
synapse_pairs["source_position"] = (
    synapse_table.set_index("id").loc[synapse_pairs["source"], "ctr_pt_position"].values
)
synapse_pairs["target_position"] = (
    synapse_table.set_index("id").loc[synapse_pairs["target"], "ctr_pt_position"].values
)

# %%
from nglui import site_utils, statebuilder

site_utils.set_default_config(target_site="spelunker", datastack_name="v1dd")

img, seg = statebuilder.from_client(client)
seg.add_selection_map(fixed_ids=root_id)

anno = statebuilder.AnnotationLayerConfig(
    name="labeled_synapses",
    mapping_rules=statebuilder.PointMapper(
        point_column="ctr_pt_position",
        description_column="id",
        tag_column="pred_label",
        set_position=True,
        linked_segmentation_column="pre_pt_root_id",
    ),
    tags=["soma", "shaft", "spine"],
    data_resolution=[1, 1, 1],
)

line = statebuilder.AnnotationLayerConfig(
    name="multispines",
    mapping_rules=statebuilder.LineMapper(
        point_column_a="source_position",
        point_column_b="target_position",
        description_column="multispine_id",
    ),
    data_resolution=[1, 1, 1],
)

sb1 = statebuilder.StateBuilder(layers=[img, seg, anno], client=client)
sb2 = statebuilder.StateBuilder(layers=[line], client=client)

sb = statebuilder.ChainedStateBuilder([sb1, sb2])

state_dict = sb.render_state(
    [synapse_table, synapse_pairs],
    return_as="dict",
    client=client,
)

shader = """
void main() {
int is_soma = int(prop_tag0());
int is_shaft = int(prop_tag1());
int is_spine = int(prop_tag2());
    
if ((is_soma + is_shaft + is_spine) == 0) {
    setColor(vec3(0.0, 0.0, 0.0));
} else if ((is_soma + is_shaft + is_spine) > 1) {
    setColor(vec3(1.0, 1.0, 1.0));
} else if (is_soma > 0) {
    setColor(vec3(0, 0.890196, 1.0));
} else if (is_shaft > 0) {
    setColor(vec3(0.9372549, 0.90196078, 0.27058824));
} else if (is_spine > 0) {
    setColor(vec3(0.91372549, 0.20784314, 0.63137255));
}
setPointMarkerSize(15.0);
}
"""
state_dict["layers"][-2]["shader"] = shader

state_id = client.state.upload_state_json(state_dict)

base_url = "https://spelunker.cave-explorer.org/#!middleauth+https://globalv1.em.brain.allentech.org/nglstate/api/v1/"

url = base_url + str(state_id)
print(url)
# %%
# synapse_summary.query("component_id.isin(@multi_components)").groupby("component_id")[
#     "id"
# ].unique()

# %%

# %%

datastack = "v1dd"
version = 974
root_id = 864691132533489754


# cv = client.info.segmentation_cloudvolume(progress=False)

# path = Path(f"gs://bdp-ssa//{datastack}/{PARAMETER_NAME}")

# cf, _ = interpret_path(path)

# feature_path = path / "features"
# # graph_out_path = path / "graphs"
# synapse_mappings_path = path / "synapse-mappings"


synapse_summary.head(20)

# %%

tq = TaskQueue(f"https://sqs.us-west-2.amazonaws.com/629034007606/{QUEUE_NAME}")


# def stop_fn(elapsed_time):
#     if elapsed_time > LEASE_SECONDS:
#         logging.info("Stopping due to time limit.")
#         requests.post(URL, json={"content": "Stopping due to time limit."})
#         return True


def stop_fn(executed):
    if executed > MAX_RUNS:
        quit()


if RUN:
    tq.poll(lease_seconds=LEASE_SECONDS, verbose=False, tally=False, stop_fn=stop_fn)

# %%

if REQUEST:
    import pandas as pd
    from cloudfiles import CloudFiles

    cell_table = pd.read_feather(
        "/Users/ben.pedigo/code/meshrep/cloud-mesh/data/v1dd_single_neuron_soma_ids.feather"
    )

    root_ids = np.unique(cell_table["pt_root_id"])

    cf = CloudFiles(f"gs://bdp-ssa/v1dd/{PARAMETER_NAME}")
    done_files = list(cf.list("features"))
    done_roots = [
        int(file.split("/")[-1].split(".")[0])
        for file in done_files
        if file.endswith(".npz")
    ]
    root_ids = np.setdiff1d(root_ids, done_roots)
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

    tasks = [partial(run_for_root, root_id, "v1dd", 974) for root_id in root_ids]

    # tq.insert(tasks)

# %%
