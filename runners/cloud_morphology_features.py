# %%
import json
import logging
import os
import platform
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import toml
import urllib3
from cloud_mesh import CloudMorphology
from cloudfiles import CloudFiles
from taskqueue import TaskQueue, queueable

SYSTEM = platform.system()

urllib3.disable_warnings()

# suppress warnings for WARNING:urllib3.connectionpool:Connection pool is full...

# logging.getLogger("urllib3").setLevel(logging.CRITICAL)

MODEL_NAME = os.environ.get("MODEL_NAME", "absolute-solo-yak")
VERBOSE = str(os.environ.get("VERBOSE", "True")).lower() == "true"
N_JOBS = int(os.environ.get("N_JOBS", -2))
REPLICAS = int(os.environ.get("REPLICAS", 1))
QUEUE_NAME = os.environ.get("QUEUE_NAME", "ben-skedit")
RUN = os.environ.get("RUN", False)
REQUEST = os.environ.get("REQUEST", not RUN)
LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL", "ERROR")
LEASE_SECONDS = int(os.environ.get("LEASE_SECONDS", 7200))
RECOMPUTE = bool(os.environ.get("RECOMPUTE", "False").lower() == "true")
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", 4))
BACKOFF_FACTOR = int(os.environ.get("BACKOFF_FACTOR", 4))
BACKOFF_MAX = int(os.environ.get("BACKOFF_MAX", 120))
MAX_RUNS = int(os.environ.get("MAX_RUNS", 5))
MODEL_VARIANT = os.environ.get("MODEL_VARIANT", "hks_model_calibrated")


# logging.basicConfig(level="ERROR")

# logging.getLogger("meshmash").setLevel(level=LOGGING_LEVEL)

if SYSTEM == "Darwin" and False:
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
PARAMETER_NAME = "absolute-solo-yak"

# model_folder = Path(__file__).parent.parent / "models" / PARAMETER_NAME
# model_path = model_folder / f"{MODEL_VARIANT}.joblib"
# model = load(model_path)


# %%


@queueable
def run_for_root(
    root_id: int,
    datastack: str,
    version: int,
):
    # total_time = time.time()
    morphology = CloudMorphology(
        root_id=root_id,
        version=version,
        datastack=datastack,
        model_name=MODEL_VARIANT,
        # model=model,
        parameters=parameters,
        parameter_name=PARAMETER_NAME,
        select_label="spine",
        lookup_nucleus=False,
        recompute=RECOMPUTE,
        verbose=VERBOSE,
        n_jobs=N_JOBS,
        prediction_schema="new",
    )
    morphology.morphometry_summary
    morphology.post_synapse_predictions

    # morphology.pre_synapse_mappings
    # morphology.post_synapse_mappings
    # morphology.condensed_features

    # morphology.post_synapse_predictions

    # synapse_time = time.time()
    # morphology.pre_synapse_mapping
    # synapse_time = time.time() - synapse_time

    # feature_time = time.time()
    # morphology.condensed_features
    # feature_time = time.time() - feature_time

    # total_time = time.time() - total_time

    # if morphology._mesh is not None:
    #     timing_dict = {}
    #     timing_dict["root_id"] = str(root_id)
    #     timing_dict["n_vertices"] = morphology.mesh[0].shape[0]
    #     timing_dict["n_faces"] = morphology.mesh[1].shape[0]
    #     timing_dict['n_pre_synapses'] = len(morphology.pre_synapse_mapping)
    #     timing_dict["synapse_time"] = synapse_time
    #     timing_dict["feature_time"] = feature_time
    #     timing_dict["total_time"] = total_time
    #     timing_dict["timestamp"] = time.time()

    #     path = Path(f"gs://bdp-ssa//{datastack}/{MODEL_NAME}")

    #     cf, _ = interpret_path(path)

    #     cf.put_json(f"pre_object_timings/{root_id}.json", timing_dict)

    return True


tq = TaskQueue(f"https://sqs.us-west-2.amazonaws.com/629034007606/{QUEUE_NAME}")


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
        # version = 1412
        # client = CAVEclient(datastack_name=datastack, version=version)

        # NOTE: this was for getting cells with labels in my training set
        # table = pd.read_csv(
        #     "/Users/ben.pedigo/code/meshrep/meshrep/experiments/cautious-fig-thaw/labels.csv"
        # )
        # root_ids = table["pt_root_id"].unique()

        # NOTE: this was for getting all cells in the column
        # table = (
        #     client.materialize.query_table("allen_v1_column_types_slanted_ref")
        #     .drop_duplicates("pt_root_id", keep=False)
        #     .query("pt_root_id != 0")
        #     .set_index("pt_root_id")
        # )

        # NOTE: this was for getting all putative neurons
        # table = (
        #     client.materialize.query_view("""aibs_cell_info""")
        #     .drop_duplicates("pt_root_id", keep=False)
        #     .set_index("pt_root_id")
        #     .query("broad_type == 'excitatory' or broad_type == 'inhibitory'")
        #     .query("pt_root_id != 0")
        # )
        # root_ids = table.index.unique()

        # NOTE: this is for looking at a handful of inhibitory cells from 611
        table = pd.read_csv(
            "/Users/ben.pedigo/code/meshrep/611_inhibitory_cells_at_1474.csv"
        ).set_index("pt_root_id")
        root_ids = table.index.unique()
        tasks += [
            partial(
                run_for_root,
                root_id,
                datastack,
                1474,
            )
            for root_id in root_ids
        ]

        # NOTE: this was for getting column inputs at 1412
        table = (
            pd.read_csv("meshrep/data/random/synapses_onto_column_count_1412.csv")
            .set_index("pre_pt_root_id")
            .query("synapse_onto_column_count_1412 >= 3")
        )
        root_ids = table.index.unique()
        tasks += [
            partial(run_for_root, root_id, datastack, 1412) for root_id in root_ids
        ]

        # NOTE: this was for getting column inputs at 117
        # table = (
        #     pd.read_csv(
        #         "/Users/ben.pedigo/code/meshrep/meshrep/data/random/old_pre_status.csv"
        #     )
        #     .set_index("old_pre_pt_root_id")
        #     .query("(synapse_onto_column_count_1412 >= 3)")
        # )
        # root_ids = table.index.unique()

        # Note: column outputs

        # root_ids = np.random.permutation(root_ids)
        # tasks += [partial(run_for_root, root_id, datastack, version) for root_id in root_ids]
        print(len(tasks))

    if False:
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
            )
            for root_id in root_ids
        ]

    if len(tasks) > 0 and False:
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
# run_for_root(root_ids[-1], 'minnie65_phase3_v1', 1412)


# %%
# root_id = root_ids[0]
# run_for_root(root_id, datastack, 1412)
# datastack = "zheng_ca3"
# version = 195
# table = pd.read_csv(
#     "/Users/ben.pedigo/code/meshrep/cloud-mesh/data/zheng_ca3/updated_segids_20250423_manual.csv",
#     index_col=0,
# )
# root_ids = table["seg_m195_20250423"].unique()


# for root_id in tqdm(root_ids):
#     run_for_root(root_id, datastack, version)
# %%
# from joblib import Parallel, delayed
# from tqdm_joblib import tqdm_joblib

# with tqdm_joblib(total=len(root_ids)):
#     results = Parallel(n_jobs=N_JOBS, verbose=False)(
#         delayed(run_for_root)(root_id, datastack, version) for root_id in root_ids
#     )
# %%
# tasks += [
#     partial(
#         run_for_root,
#         root_id,
#         datastack,
#         version,
#     )
#     for root_id in root_ids
# ]

# %%
from joblib import load

model_folder = Path(__file__).parent.parent / "models" / PARAMETER_NAME
model_path = model_folder / f"{MODEL_VARIANT}.joblib"
model = load(model_path)

# %%
cell_table = pd.read_feather(
    "/Users/ben.pedigo/code/meshrep/cloud-mesh/data/v1dd_single_neuron_soma_ids.feather"
)
root_ids = np.unique(cell_table["pt_root_id"])

root_id = root_ids[-12]
datastack = "v1dd"
version = 974
morphology = CloudMorphology(
    root_id=root_id,
    version=version,
    datastack=datastack,
    model_name=MODEL_VARIANT,
    model=model,
    parameters=parameters,
    parameter_name=PARAMETER_NAME,
    select_label="spine",
    lookup_nucleus=False,
    recompute=False,
    verbose=True,
    prediction_schema="new",
    n_jobs=-1,
)

morphology.morphometry_summary
morphology.post_synapse_predictions

# %%
from caveclient import CAVEclient

version = 1154

client = CAVEclient(datastack_name=datastack, version=version)

neuron_table = client.materialize.query_table("neurons_soma_model")


# %%

import pyvista as pv

plotter = pv.Plotter()

mesh = morphology.mesh

plotter.add_mesh(
    pv.make_tri_mesh(*mesh),
    opacity=0.5,
)
plotter.add_points(
    morphology.morphometry_summary[["x", "y", "z"]].values,
    render_points_as_spheres=True,
    point_size=5,
    scalars=morphology.morphometry_summary["n_post_synapses"].values,
    # scalars=np.log(morphology.morphometry_summary["size_nm3"].values),
)
plotter.enable_fly_to_right_click()

plotter.show()
