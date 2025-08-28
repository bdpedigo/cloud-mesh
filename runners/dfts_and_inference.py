# %%

import os
import platform
import time
from functools import partial
from pathlib import Path

import joblib
import pandas as pd
from caveclient import CAVEclient
from cloud_mesh import DataFrameTensorStore, MorphClient
from joblib import Parallel, delayed, load
from taskqueue import TaskQueue, queueable
from tqdm_joblib import tqdm_joblib

from morphsync import MorphSync

n_threads = joblib.cpu_count()
print()
print(f"Joblib sees {n_threads} threads available.")
print()

# %%

PARAMETER_NAME = os.environ.get("PARAMETER_NAME", "absolute-solo-yak")
MODEL_NAME = os.environ.get("MODEL_NAME", "auburn-elk-detour")
MODEL_VARIANT = os.environ.get("MODEL_VARIANT", "synapse_hks_model")

VERBOSE = str(os.environ.get("VERBOSE", "True")).lower() == "true"
N_JOBS = int(os.environ.get("N_JOBS", -1))
REPLICAS = int(os.environ.get("REPLICAS", 1))
QUEUE_NAME = os.environ.get("QUEUE_NAME", "ben-paleo")
RUN: bool = str(os.environ.get("RUN", False)).lower() == "true"
print()
print("RUN =", RUN)
print()
REQUEST: bool = str(os.environ.get("REQUEST", not RUN)).lower() == "true"
print()
print("REQUEST =", REQUEST)
print()

if platform.system() != "Darwin":
    REQUEST = False

LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL", "ERROR")
LEASE_SECONDS = int(os.environ.get("LEASE_SECONDS", 7200))
RECOMPUTE = bool(os.environ.get("RECOMPUTE", "False").lower() == "true")
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", 3))
BACKOFF_FACTOR = int(os.environ.get("BACKOFF_FACTOR", 3))
BACKOFF_MAX = int(os.environ.get("BACKOFF_MAX", 120))
MAX_RUNS = int(os.environ.get("MAX_RUNS", 5))
N_BATCHES = int(os.environ.get("N_BATCHES", 25))
VERSION = int(os.environ.get("VERSION", 1412))
DATASTACK = os.environ.get("DATASTACK", "minnie65_phase3_v1")


model_folder = Path(__file__).parent.parent / "models" / MODEL_NAME
model_path = model_folder / f"{MODEL_VARIANT}.joblib"
model = load(model_path)
MODEL_KEY = f"{MODEL_NAME}-{MODEL_VARIANT}"

tq = TaskQueue(f"https://sqs.us-west-2.amazonaws.com/629034007606/{QUEUE_NAME}")


# %%

target_n_rows = 210_000_000
post_path = Path(
    f"gs://bdp-ssa/{DATASTACK}/{PARAMETER_NAME}/{VERSION}/post-synapse-feature-ts"
)
pre_path = Path(
    f"gs://bdp-ssa/{DATASTACK}/{PARAMETER_NAME}/{VERSION}/pre-synapse-feature-ts"
)
post_prediction_path = Path(
    f"gs://bdp-ssa/{DATASTACK}/{PARAMETER_NAME}/{VERSION}/{MODEL_NAME}-{MODEL_VARIANT}/post-synapse-predictions-ts"
)


client = CAVEclient(DATASTACK, version=VERSION)
column_table = client.materialize.query_table(
    "allen_v1_column_types_slanted_ref"
).set_index("pt_root_id")
cell_table = client.materialize.query_view("aibs_cell_info")
neuron_table = cell_table.query("broad_type.isin(['excitatory', 'inhibitory'])")
neuron_table = neuron_table.query("pt_root_id != 0")
all_root_ids = neuron_table["pt_root_id"].unique()
root_batches = all_root_ids % N_BATCHES

parallel_kwargs = dict(n_jobs=N_JOBS, mmap_mode=None, backend="threading")


# %%
def get_features_from_morphs(morph, side="post") -> pd.DataFrame:
    if morph.has_layer(f"{side}_synapses") and morph.has_layer("hks_features"):
        features = morph.get_mapped_nodes(
            f"{side}_synapses", "hks_features", replace_index=True
        )

        features = features.dropna()

        other_features = morph.get_mapped_nodes(
            f"{side}_synapses",
            "supermoxel_graph",
            replace_index=True,
        )

        features = features.join(other_features[["distance_to_nucleus"]], how="left")

        return features
    else:
        return pd.DataFrame()


@queueable
def run_batch(i):
    print()
    print("Running batch", i)
    print()
    root_ids = all_root_ids[root_batches == i]

    currtime = time.time()
    mc = MorphClient(
        "minnie65_phase3_v1",
        hks_parameters="absolute-solo-yak",
        verbose=VERBOSE,
        n_jobs=N_JOBS,
        copy=True,
    )

    has_hks = mc.has_hks(root_ids)

    missing_roots = root_ids[~has_hks]
    print(len(missing_roots), "morphs are missing HKS features.")

    print(f"Found {has_hks.mean()} morphs with HKS features.")
    root_ids = root_ids[has_hks]
    morphs = [MorphSync(name=root_id) for root_id in root_ids]
    mc.add_hks(morphs)
    mc.add_supermoxel_graph(morphs)
    mc.add_synapse_mesh_mappings(morphs, side="post")
    mc.add_synapse_mesh_mappings(morphs, side="pre")

    with tqdm_joblib(
        desc="Getting features on morphs", total=len(morphs), disable=not VERBOSE
    ):
        post_synapse_features = Parallel(**parallel_kwargs)(
            delayed(get_features_from_morphs)(morph, side="post") for morph in morphs
        )

    post_synapse_features = pd.concat(post_synapse_features, axis=0)
    post_synapse_features.index.name = "synapse_id"

    if not DataFrameTensorStore.exists(post_path):
        post_store = DataFrameTensorStore.initialize_from_sample(
            post_path, post_synapse_features, n_rows=target_n_rows
        )
    post_store = DataFrameTensorStore(post_path, verbose=True)

    post_store.write_dataframe(post_synapse_features)

    posteriors = model.predict_proba(post_synapse_features)

    posteriors = pd.DataFrame(
        posteriors, index=post_synapse_features.index, columns=model.classes_
    )
    posteriors = posteriors.astype("float16")
    posteriors.index.name = "synapse_id"

    if not DataFrameTensorStore.exists(post_prediction_path):
        post_prediction_store = DataFrameTensorStore.initialize_from_sample(
            post_prediction_path, posteriors, n_rows=target_n_rows
        )
    post_prediction_store = DataFrameTensorStore(post_prediction_path, verbose=VERBOSE)
    post_prediction_store.write_dataframe(posteriors)

    del post_synapse_features
    del posteriors

    with tqdm_joblib(
        desc="Getting features on morphs", total=len(morphs), disable=not VERBOSE
    ):
        pre_synapse_features = Parallel(**parallel_kwargs)(
            delayed(get_features_from_morphs)(morph, side="pre") for morph in morphs
        )
    pre_synapse_features = pd.concat(pre_synapse_features, axis=0)
    pre_synapse_features.index.name = "synapse_id"

    if not DataFrameTensorStore.exists(pre_path):
        pre_store = DataFrameTensorStore.initialize_from_sample(
            pre_path, pre_synapse_features, n_rows=target_n_rows, verbose=VERBOSE
        )
    pre_store = DataFrameTensorStore(pre_path, verbose=VERBOSE)

    pre_store.write_dataframe(pre_synapse_features)
    del pre_synapse_features

    print()
    print(f"{time.time() - currtime:.3f} seconds elapsed for batch.")
    print()

    return True


# %%
if REQUEST:
    tq.insert(partial(run_batch, i) for i in range(N_BATCHES))


# %%


def stop_fn(executed):
    if executed >= MAX_RUNS:
        quit()


if RUN:
    tq.poll(lease_seconds=LEASE_SECONDS, verbose=False, tally=False, stop_fn=stop_fn)

#%%