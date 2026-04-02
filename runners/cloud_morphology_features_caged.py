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
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import toml
import urllib3
from cloud_mesh import CloudMorphology
from cloudfiles import CloudFile
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
LINK_PROB = float(os.environ.get("LINK_PROB", 0.1))  # 1 in 10

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
    currtime = time.time()
    morphology.condensed_features
    feature_time = time.time() - currtime
    morphology.morphometry_summary
    morphology.post_synapse_predictions

    if (LINK_PROB > 0 and np.random.rand() < LINK_PROB) or test:
        emit_link(morphology)

    if morphology._mesh is not None and not test:
        timing_dict = {}
        timing_dict["root_id"] = str(root_id)
        timing_dict["n_vertices"] = morphology.mesh[0].shape[0]
        timing_dict["n_faces"] = morphology.mesh[1].shape[0]
        timing_dict["feature_time"] = feature_time
        timing_dict["timestamp"] = time.time()

        cf = CloudFile(
            f"gs://bdp-ssa//{datastack}/{PARAMETER_NAME}/timings/{root_id}.json"
        )
        cf.put_json(timing_dict)

    return True


@queueable
def run_for_root_caged(
    root_id: int,
    datastack: str,
    version: int,
    timestamp: Optional[datetime.datetime] = None,
    lookup_nucleus=True,
    test: bool = False,
):
    # total_time = time.time()
    morphology = CloudMorphology(
        root_id=root_id,
        version=version,
        datastack=datastack,
        parameters=caged_parameters,
        parameter_name=CAGED_PARAMETER_NAME,
        lookup_nucleus=lookup_nucleus,
        recompute=RECOMPUTE,
        verbose=VERBOSE if not test else True,
        n_jobs=N_JOBS if not test else -1,
        timestamp=timestamp,
        cage=True,
    )
    morphology.cage_mesh
    currtime = time.time()
    morphology.condensed_features
    feature_time = time.time() - currtime

    if morphology._mesh is not None and not test:
        timing_dict = {}
        timing_dict["root_id"] = str(root_id)
        timing_dict["n_vertices"] = morphology.mesh[0].shape[0]
        timing_dict["n_faces"] = morphology.mesh[1].shape[0]
        timing_dict["feature_time"] = feature_time
        timing_dict["timestamp"] = time.time()

        cf = CloudFile(
            f"gs://bdp-ssa//{datastack}/{CAGED_PARAMETER_NAME}/timings/{root_id}.json"
        )
        cf.put_json(timing_dict)
    if test:
        return morphology
    else:
        return True


# print("Polling...")QUEUE_NAME = "ben-skedit"
# tq = TaskQueue(f"https://sqs.us-west-2.amazonaws.com/629034007606/{QUEUE_NAME}")

# %%
# test_root = 864691132290801922
# test_datastack = "h01_c3_flat"
# test_version = 1054
# # RECOMPUTE = True
# morphology = run_for_root(test_root, test_datastack, test_version, test=True)

# %%

# client = CAVEclient(test_datastack, version=test_version)
# client.materialize.get_versions()
# client.info.segmentation_cloudvolume()

# %%


# morphology.cage_mesh
# morphology._cage_mapping

# test_root = 16264492
# test_datastack = "j0251"
# test_version = 0
# morphology = run_for_root_caged(
#     test_root, test_datastack, test_version, test=True, lookup_nucleus=False
# )
# morphology.cage_mesh
# morphology._cage_mapping
# %%

QUEUE_NAME = "ben-skedit"
tq = TaskQueue(f"https://sqs.us-west-2.amazonaws.com/629034007606/{QUEUE_NAME}")


def stop_fn(executed):
    if executed >= MAX_RUNS:
        quit()

#%%

TEST = False
if TEST:
    # test_root = 864691132467958274
    # test_datastack = "v1dd"
    # test_version = 1196
    # run_for_root(test_root, test_datastack, test_version)
    test_root = 16264492
    test_datastack = "j0251"
    test_version = 0
    morphology = run_for_root_caged(
        test_root, test_datastack, test_version, test=True, lookup_nucleus=False
    )
    morphology.cage_mesh
    quit()
elif RUN:
    print("Polling...")
    tq.poll(lease_seconds=LEASE_SECONDS, verbose=False, tally=False, stop_fn=stop_fn)


# %%


# %%

if REQUEST:
    datastack_to_version = {
        "h01_c3_flat": 1054,
        "minnie65_phase3_v1": 1718,
        "zheng_ca3": 357,
    }

    from pathlib import Path

    import numpy as np

    # #
    # # Microns
    # #

    # microns_version = datastack_to_version["minnie65_phase3_v1"]
    # microns_labels = make_label_table(
    #     "now",
    #     root_version=microns_version,
    #     threshold=100,
    # )

    # microns_labeled_ids = (
    #     microns_labels.groupby(f"pt_root_id_{microns_version}")
    #     .size()
    #     .rename("count")
    #     .sort_values(ascending=False)
    #     .iloc[:200]
    # ).to_frame()

    # client = CAVEclient("minnie65_phase3_v1", version=microns_version)
    # info = client.materialize.views.aibs_cell_info().query()
    # info = info.set_index("pt_root_id").sort_index()

    # microns_labeled_ids = microns_labeled_ids.join(info[["broad_type"]], how="left")

    # microns_keep_ids = microns_labeled_ids.query("broad_type == 'excitatory'").index[
    #     :80
    # ]

    # microns_random_ids = (
    #     info.query("broad_type == 'excitatory' and ~pt_root_id.isin(@microns_keep_ids)")
    #     .sample(80, random_state=8888)
    #     .index
    # )

    # #
    # # H01
    # #
    # h01_version = datastack_to_version["h01_c3_flat"]
    # h01_id_path = Path("/Users/ben.pedigo/code/meshrep/validation_ids.txt")

    # client = CAVEclient("h01_c3_flat", version=h01_version)

    # cell_types_table = client.materialize.query_table(table="cells")
    # cell_types_table = cell_types_table.sort_values("pt_root_id")

    # pyramidal_cell_types = cell_types_table.query(
    #     "cell_type == 'PYRAMIDAL' and pt_root_id != 0"
    # )

    # validation_ids = pyramidal_cell_types.sample(500, random_state=8888)[
    #     "pt_root_id"
    # ].values

    # loaded_validation_ids = np.loadtxt("validation_ids.txt", dtype="int64")

    # assert np.array_equal(np.sort(validation_ids), np.sort(loaded_validation_ids))

    # h01_labeled_ids = validation_ids[:80]

    # h01_random_ids = (
    #     pyramidal_cell_types.query("pt_root_id not in @validation_ids")
    #     .sample(80, random_state=8888)["pt_root_id"]
    #     .values
    # )

    # #
    # # j0251
    # #

    # params = {
    #     "celltype": "MSN",
    #     "min_dendrite_length": "200.0",
    # }
    # base_url = "https://syconn.esc.mpcdf.mpg.de"
    # response = requests.get(f"{base_url}/j0251/neurons/json", params=params)
    # neurons = response.json()
    # neuron_ids = np.sort(neurons["neuron_ids"])

    # rng = np.random.default_rng(8888)
    # j0251_keep_ids = rng.choice(neuron_ids, size=80, replace=False)

    # tasks = []

    # for root_id in microns_keep_ids:
    #     tasks.append(
    #         partial(run_for_root, root_id, "minnie65_phase3_v1", microns_version)
    #     )
    #     tasks.append(
    #         partial(run_for_root_caged, root_id, "minnie65_phase3_v1", microns_version)
    #     )
    # for root_id in microns_random_ids:
    #     tasks.append(
    #         partial(run_for_root, root_id, "minnie65_phase3_v1", microns_version)
    #     )
    #     tasks.append(
    #         partial(run_for_root_caged, root_id, "minnie65_phase3_v1", microns_version)
    #     )

    # for root_id in h01_labeled_ids:
    #     tasks.append(partial(run_for_root, root_id, "h01_c3_flat", h01_version))
    #     tasks.append(partial(run_for_root_caged, root_id, "h01_c3_flat", h01_version))
    # for root_id in h01_random_ids:
    #     tasks.append(partial(run_for_root, root_id, "h01_c3_flat", h01_version))
    #     tasks.append(partial(run_for_root_caged, root_id, "h01_c3_flat", h01_version))

    # for root_id in j0251_keep_ids:
    #     tasks.append(
    #         partial(run_for_root_caged, root_id, "j0251", 0, lookup_nucleus=False)
    #     )

    from functools import partial

    cells = Path(
        "/Users/ben.pedigo/code/meshrep/mouse_hippocampus_ca3_cell_annotations_export.csv"
    )

    cells = pd.read_csv(cells)

    cells = cells.query("subtypes.notna()").sort_values('Cell ID').groupby("subtypes").sample(80, random_state=8888)
    root_ids = cells["Cell ID"].values
    datastack = "zheng_ca3"
    version = datastack_to_version[datastack]

    tasks= []

    for root_id in root_ids:
        tasks.append(partial(run_for_root, root_id, datastack, version))
        tasks.append(partial(run_for_root_caged, root_id, datastack, version))


    if len(tasks) > 0:
        # shuffle
        from random import shuffle

        shuffle(tasks)

        batch_size = 10_000
        n_batches = (len(tasks) + batch_size - 1) // batch_size
        print(n_batches, "batches")

        for task_group in np.array_split(tasks, n_batches):
            tq.insert(task_group)

# %%
