# %%
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="divide by zero encountered in log"
)

import datetime
import json
import logging
import os
import platform
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import toml
import urllib3
from cloud_mesh import CloudMorphology
from cloudfiles import CloudFiles
from joblib import load
from nglui.statebuilder import ViewerState
from taskqueue import TaskQueue, queueable

SYSTEM = platform.system()

urllib3.disable_warnings()

# suppress warnings for WARNING:urllib3.connectionpool:Connection pool is full...

logging.getLogger("urllib3").setLevel(logging.CRITICAL)

PARAMETER_NAME = os.environ.get("PARAMETER_NAME", "absolute-solo-yak")
MODEL_NAME = os.environ.get("MODEL_NAME", "auburn-elk-detour")
MODEL_VARIANT = os.environ.get("MODEL_VARIANT", "synapse_hks_model")

VERBOSE = str(os.environ.get("VERBOSE", "True")).lower() == "true"
N_JOBS = int(os.environ.get("N_JOBS", -2))
REPLICAS = int(os.environ.get("REPLICAS", 1))
QUEUE_NAME = os.environ.get("QUEUE_NAME", "ben-skedit")
RUN = os.environ.get("RUN", False)
REQUEST = os.environ.get("REQUEST", not RUN)
LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL", "ERROR")
LEASE_SECONDS = int(os.environ.get("LEASE_SECONDS", 1000))
RECOMPUTE = bool(os.environ.get("RECOMPUTE", "False").lower() == "true")
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", 4))
BACKOFF_FACTOR = int(os.environ.get("BACKOFF_FACTOR", 4))
BACKOFF_MAX = int(os.environ.get("BACKOFF_MAX", 120))
MAX_RUNS = int(os.environ.get("MAX_RUNS", 100))
TEST = bool(os.environ.get("TEST", "False").lower() == "true")
LINK_PROB = float(os.environ.get("LINK_PROB", 0.0001))  # 1 in 10,000

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


parameter_folder = Path(__file__).parent.parent / "models" / PARAMETER_NAME
parameters = toml.load(parameter_folder / "parameters.toml")
parameters = replace_none(parameters)

CAGED_PARAMETER_NAME = "internal-mud-secure"
parameter_folder = Path(__file__).parent.parent / "models" / CAGED_PARAMETER_NAME
caged_parameters = toml.load(parameter_folder / "parameters.toml")
caged_parameters = replace_none(caged_parameters)

model_folder = Path(__file__).parent.parent / "models" / MODEL_NAME
model_path = model_folder / f"{MODEL_VARIANT}.joblib"
model = load(model_path)
MODEL_KEY = f"{MODEL_NAME}-{MODEL_VARIANT}"
# %%

if False:
    datastack = "zheng_ca3"
    version = 357
    root_ids = np.loadtxt(
        "/Users/ben.pedigo/code/meshrep/cloud-mesh/data/zheng_ca3/ca3_object_selection_v357.txt",
        dtype="str",
    ).astype(np.int64)
    root_ids = np.random.permutation(root_ids)

    cf = CloudFiles(
        "gs://bdp-ssa/zheng_ca3/v357/absolute-solo-yak/auburn-elk-detour-simple_hks_model/post-synapse-predictions"
    )
    exists = cf.exists([f"{root_id}.csv.gz" for root_id in root_ids])

    exists_map = pd.Series(exists)
    exists_map.index = exists_map.index.map(
        lambda x: int(x.split("/")[-1].split(".")[0])
    )
    missing = exists_map[~exists_map].index

    print(len(missing))

    from cloud_mesh import MorphClient

    mc = MorphClient(
        "zheng_ca3",
        hks_parameters="absolute-solo-yak",
        verbose=VERBOSE,
        n_jobs=N_JOBS,
        model_name=MODEL_KEY,
        model_target="spine",
        version=version,
    )
    has_hks = mc.has_synapse_mesh_mappings(root_ids)


# %%
from caveclient import CAVEclient

client = CAVEclient("h01_c3_flat")
client.info.get_datastack_info()

# %%


def emit_link(morphology):
    predictions = morphology.post_synapse_predictions
    positions = morphology.mesh[0][morphology.post_synapse_mappings]
    positions = pd.DataFrame(
        positions, columns=["x", "y", "z"], index=morphology.post_synapse_mappings.index
    )
    predictions = predictions.join(positions, how="inner")
    predictions["point"] = predictions[["x", "y", "z"]].values.tolist()

    vs = (
        ViewerState(client=morphology.client)
        .add_layers_from_client(alpha_3d=0.6)
        .add_segments([morphology.root_id])
        .add_points(
            predictions.query("pred_label == 'spine'"),
            name="spines",
            point_column="point",
            data_resolution=[1, 1, 1],
            swap_visible_segments_on_move=False,
            color="#FF00CC",
        )
        .add_points(
            predictions.query("pred_label == 'shaft'"),
            name="shaft",
            point_column="point",
            data_resolution=[1, 1, 1],
            swap_visible_segments_on_move=False,
            color="#E5FF00",
        )
        .add_points(
            predictions.query("pred_label == 'soma'"),
            name="soma",
            point_column="point",
            data_resolution=[1, 1, 1],
            swap_visible_segments_on_move=False,
            color="#00FFFF",
        )
    )
    link = vs.to_url(shorten=True)
    requests.post(URL, json={"content": link})


@queueable
def run_for_root(
    root_id: int,
    datastack: str,
    version: int,
    timestamp: Optional[datetime.datetime] = None,
    test: bool = False,
):
    # total_time = time.time()
    morphology = CloudMorphology(
        root_id=root_id,
        version=version,
        datastack=datastack,
        model_name=MODEL_KEY,
        model=model,
        parameters=parameters,
        parameter_name=PARAMETER_NAME,
        select_label="spine",
        lookup_nucleus=True,
        recompute=RECOMPUTE,
        verbose=VERBOSE if not test else True,
        n_jobs=N_JOBS if not test else -1,
        prediction_schema="new",
        timestamp=timestamp,
        cage=False,
    )
    morphology.pre_synapse_mappings
    morphology.post_synapse_mappings
    morphology.condensed_features
    morphology.morphometry_summary
    morphology.post_synapse_predictions

    if (LINK_PROB > 0 and np.random.rand() < LINK_PROB) or test:
        emit_link(morphology)

    return True


@queueable
def run_for_root_caged(
    root_id: int,
    datastack: str,
    version: int,
    timestamp: Optional[datetime.datetime] = None,
    test: bool = False,
):
    # total_time = time.time()
    morphology = CloudMorphology(
        root_id=root_id,
        version=version,
        datastack=datastack,
        parameters=caged_parameters,
        parameter_name=CAGED_PARAMETER_NAME,
        lookup_nucleus=True,
        recompute=RECOMPUTE,
        verbose=VERBOSE if not test else True,
        n_jobs=N_JOBS if not test else -1,
        timestamp=timestamp,
        cage=True,
    )
    morphology.cage_mesh
    morphology.condensed_features

    return morphology


RECOMPUTE = False


test_root = 864691132355457423
test_datastack = "h01_c3_flat"
test_version = 1053
morphology = run_for_root_caged(test_root, test_datastack, test_version, test=True)
morphology.cage_mesh
morphology._cage_mapping

# %%

QUEUE_NAME = "ben-skedit"
tq = TaskQueue(f"https://sqs.us-west-2.amazonaws.com/629034007606/{QUEUE_NAME}")


def stop_fn(executed):
    if executed >= MAX_RUNS:
        quit()


if TEST:
    test_root = 864691132467958274
    test_datastack = "v1dd"
    test_version = 1196
    run_for_root(test_root, test_datastack, test_version)
elif RUN:
    print("Polling...")
    tq.poll(lease_seconds=LEASE_SECONDS, verbose=False, tally=False, stop_fn=stop_fn)

# %%

import polars as pl
from cloud_mesh import MorphClient

mc = MorphClient(
    "v1dd",
    version=1196,
    hks_parameters="absolute-solo-yak",
    verbose=VERBOSE,
    n_jobs=N_JOBS,
    model_name=MODEL_KEY,
    model_target="spine",
)

root_ids = pl.read_parquet("v1dd_root_selections.parquet")
root_ids = root_ids["post_pt_root_id"].unique().to_numpy()
root_ids = np.random.permutation(root_ids)
mc.has_hks(root_ids[:100])
# %%


# %%


if REQUEST:
    import pandas as pd
    import polars as pl
    from caveclient import CAVEclient
    from cloud_mesh import MorphClient
    from cloudfiles import CloudFiles

    tasks = []

    if True:
        datastack = "v1dd"
        version = 1196
        root_ids = pl.read_parquet("v1dd_root_selections.parquet")
        root_ids = root_ids["post_pt_root_id"].unique().to_numpy()
        root_ids = np.random.permutation(root_ids)
        path = "gs://bdp-ssa/v1dd/absolute-solo-yak/features"
        queries = [f"{root_id}.npz" for root_id in root_ids]
        cf = CloudFiles(path)
        does_exist = cf.exists(queries)
        exists_indicator = np.vectorize(does_exist.get)(queries)
        root_ids = root_ids[~exists_indicator]
        tasks += [
            partial(
                run_for_root,
                root_id,
                datastack,
                version,
            )
            for root_id in root_ids
        ]

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
        version = 1154
        datastack = "v1dd"
        client = CAVEclient(datastack_name=datastack, version=version)

        neuron_table = client.materialize.query_table("neurons_soma_model")
        neuron_table = (
            neuron_table.drop_duplicates("pt_root_id", keep=False)
            .query("pt_root_id != 0")
            .set_index("pt_root_id")
        )
        root_ids = neuron_table.index.unique()
        root_ids = np.random.permutation(root_ids)
        tasks += [
            partial(
                run_for_root,
                root_id,
                datastack,
                version,
            )
            for root_id in root_ids
        ]

    if False:
        pass
        # NOTE: this was for getting cells with labels in my training set
        # table = pd.read_csv(
        #     "/Users/ben.pedigo/code/meshrep/meshrep/experiments/cautious-fig-thaw/labels.csv"
        # )
        # root_ids = table["pt_root_id"].unique()

        # NOTE: this was for getting all cells in the column
        # version = 1412
        # client = CAVEclient(datastack_name=datastack, version=version)

        # table = client.materialize.query_table(
        #     "allen_v1_column_types_slanted_ref"
        # ).set_index("target_id")

        # root_ids = table.index.unique()

    if False:
        datastack = "minnie65_phase3_v1"
        version = 1412
        client = CAVEclient(datastack, version=version)
        # column_table = client.materialize.query_table(
        #     "allen_v1_column_types_slanted_ref"
        # ).set_index("pt_root_id")
        cell_table = client.materialize.query_view("aibs_cell_info")
        neuron_table = (
            cell_table.query("broad_type.isin(['excitatory', 'inhibitory'])")
            .copy()
            .set_index("id")
        )
        # neuron_table = neuron_table.query("pt_root_id != 0")
        root_ids = neuron_table["pt_root_id"].unique()
        root_ids = np.random.permutation(root_ids)
        tasks += [
            partial(run_for_root, root_id, datastack, version) for root_id in root_ids
        ]
        # root_ids = np.random.choice(root_ids, size=N_PER_BATCH, replace=False)

        # currtime = time.time()
        # mc = MorphClient(
        #     "minnie65_phase3_v1",
        #     hks_parameters="absolute-solo-yak",
        #     verbose=VERBOSE,
        #     n_jobs=N_JOBS,
        # )

        # has_hks = mc.has_hks(root_ids)

        # root_ids = root_ids[~has_hks]
        # print(len(root_ids), "morphs are missing HKS features.")

    if False:
        datastack = "minnie65_phase3_v1"
        version = 1412
        client = CAVEclient(datastack, version=version)
        cell_table = client.materialize.query_table("allen_v1_column_types_slanted_ref")
        cell_table = cell_table.drop_duplicates("pt_root_id", keep=False).set_index(
            "pt_root_id"
        )
        root_ids = cell_table.index.unique()

        root_ids = np.random.permutation(root_ids)
        tasks += [
            partial(run_for_root, root_id, datastack, version) for root_id in root_ids
        ]

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
        # table = pd.read_csv(
        #     "/Users/ben.pedigo/code/meshrep/611_inhibitory_cells_at_1474.csv"
        # ).set_index("pt_root_id")
        # root_ids = table.index.unique()
        # tasks += [
        #     partial(
        #         run_for_root,
        #         root_id,
        #         datastack,
        #         1474,
        #     )
        #     for root_id in root_ids
        # ]

        # NOTE: this was for getting column inputs at 1412
        # table = (
        #     pd.read_csv("meshrep/data/random/synapses_onto_column_count_1412.csv")
        #     .set_index("pre_pt_root_id")
        #     .query("synapse_onto_column_count_1412 >= 3")
        # )
        # root_ids = table.index.unique()

        # tasks += [
        #     # partial(run_for_root, root_id, datastack, 1412) for root_id in root_ids
        # ]

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
        # tasks += [
        #     partial(run_for_root, root_id, datastack, version) for root_id in root_ids
        # ]
        # print(len(tasks))

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
    if False:
        datastack = "zheng_ca3"
        version = 357
        root_ids = np.loadtxt(
            "/Users/ben.pedigo/code/meshrep/cloud-mesh/data/zheng_ca3/ca3_object_selection_v357.txt",
            dtype="str",
        ).astype(np.int64)
        root_ids = np.random.permutation(root_ids)

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

# %%


if len(tasks) > 0:
    for task_group in np.array_split(tasks, 20):
        tq.insert(task_group)
