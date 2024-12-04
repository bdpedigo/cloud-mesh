# %%
import datetime
import os
import time
from functools import partial
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import toml
from caveclient import CAVEclient, set_session_defaults
from cloudfiles import CloudFiles
from fast_simplification import simplify
from joblib import load
from taskqueue import TaskQueue, queueable

from cloudigo import put_dataframe
from meshmash import chunked_hks_pipeline, get_label_components, project_points_to_mesh


def replace_none(parameters):
    for key, value in parameters.items():
        if isinstance(value, dict):
            parameters[key] = replace_none(value)
        elif value == "None":
            parameters[key] = None
    return parameters


REQUEST = False
TEST = False
RUN = os.getenv("RUN", "true").lower() == "true"
QUEUE_NAME = os.getenv("QUEUE_NAME", "ben-skedit")
REPLICAS = int(os.environ.get("REPLICAS", 1))
N_JOBS = 1 if REPLICAS > 1 else -1
VERBOSE = int(os.environ.get("VERBOSE", 1))


MODEL_NAME = "obtuse-dog-search"
model_folder = Path(__file__).parent.parent / "models" / MODEL_NAME

parameters = toml.load(model_folder / "parameters.toml")
parameters = replace_none(parameters)

cloud_folder = "gs://allen-minnie-phase3/ben-ssa/{MODEL_NAME}"
cloud_path = Path(cloud_folder)
cf = CloudFiles("gs://allen-minnie-phase3/ben-ssa/{MODEL_NAME}")

set_session_defaults(max_retries=5, backoff_factor=0.5)
client = CAVEclient("minnie65_phase3_v1", version=1181)
cv = client.info.segmentation_cloudvolume(progress=False)


# %%
def load_synapses(
    root_id,
    mesh,
    client,
    labeled=True,
    distance_threshold=None,
    mapping_column="ctr_pt_position",
):
    ts = client.chunkedgraph.get_root_timestamps(root_id, latest=True)[0]
    post_synapses = client.materialize.query_table(
        "synapses_pni_2",
        filter_equal_dict={"post_pt_root_id": root_id},
        timestamp=ts,
        log_warning=False,
        split_positions=True,
        desired_resolution=[1, 1, 1],
    )
    post_synapses.set_index("id", inplace=True)

    synapse_locs = post_synapses[
        [f"{mapping_column}_x", f"{mapping_column}_y", f"{mapping_column}_z"]
    ].values

    indices, distances = project_points_to_mesh(
        synapse_locs, mesh, distance_threshold=distance_threshold, return_distances=True
    )

    post_synapses["mesh_index"] = indices
    post_synapses["distance_to_mesh"] = distances

    post_synapses.query("mesh_index != -1", inplace=True)

    if isinstance(mesh, tuple):
        vertices = mesh[0]
    else:
        vertices = mesh.vertices

    mesh_pts = vertices[post_synapses["mesh_index"]]
    post_synapses["mesh_pt_position_x"] = mesh_pts[:, 0]
    post_synapses["mesh_pt_position_y"] = mesh_pts[:, 1]
    post_synapses["mesh_pt_position_z"] = mesh_pts[:, 2]

    if labeled:
        post_synapses = post_synapses.query("label.notnull()")
    return post_synapses


def load_model(model_name):
    current = Path(__file__).parent.parent
    model_path = current / f"models/{model_name}/model.joblib"
    return load(model_path)


@queueable
def run_prediction_for_root(root_id):
    model = load_model(MODEL_NAME)

    currtime = time.time()
    raw_mesh = cv.mesh.get(root_id, **parameters["cv-mesh-get"])[root_id]
    pull_mesh_time = time.time() - currtime
    mesh = (raw_mesh.vertices, raw_mesh.faces)
    raw_n_vertices = len(raw_mesh.vertices)

    currtime = time.time()
    mesh = simplify(mesh[0], mesh[1], **parameters["simplify"])
    processed_n_vertices = len(mesh[0])
    simplify_time = time.time() - currtime

    currtime = time.time()
    synapses = load_synapses(
        root_id,
        mesh,
        client,
        labeled=False,
        distance_threshold=parameters["project_points_to_mesh"]["distance_threshold"],
    )
    pull_synapses_time = time.time() - currtime

    hks, timings = chunked_hks_pipeline(
        mesh,
        return_timing=True,
        verbose=VERBOSE,
        n_jobs=N_JOBS,
        **parameters["chunked_hks_pipeline"],
    )

    currtime = time.time()
    predictions_df = pd.DataFrame(index=synapses.index)
    indices = synapses["mesh_index"]

    hks_by_synapse = np.log(hks[indices])
    posteriors = model.predict_proba(hks_by_synapse)
    classes = model.classes_
    max_inds = np.argmax(posteriors, axis=1)
    predictions = classes[max_inds]
    predictions_df["pred_label"] = predictions
    max_posteriors = posteriors[np.arange(len(posteriors)), max_inds]
    predictions_df["pred_label_posterior"] = max_posteriors

    # put_dataframe(
    #     predictions_df, cf, f"predictions/{root_id}_synapse_predictions.csv.gz"
    # )
    put_dataframe(
        predictions_df, cloud_path / f"predictions/{root_id}_synapse_predictions.csv.gz"
    )

    predict_time = time.time() - currtime

    currtime = time.time()
    pred_labels = model.predict(np.log(hks))
    label_components = get_label_components(mesh, pred_labels)
    synapses["pred_label"] = predictions_df["pred_label"]
    synapses["component_id"] = synapses["mesh_index"].map(label_components.__getitem__)
    component_counts = (
        synapses.query("pred_label == 'spine'").groupby("component_id").size()
    )
    synapses["spine_component_count"] = (
        synapses["component_id"].map(component_counts).fillna(0).astype(int)
    )
    multispine_synapses = synapses[synapses["spine_component_count"] > 1]
    multispine_synapses = multispine_synapses[["component_id", "spine_component_count"]]
    multispine_synapses["root_id"] = root_id
    put_dataframe(
        multispine_synapses,
        cloud_path / f"components/{root_id}_multispine_synapses.csv.gz",
    )
    component_label_time = time.time() - currtime

    # timings = {
    #     "root_id": root_id,
    #     "pull_mesh_time": pull_mesh_time,
    #     "preprocess_time": preprocess_time,
    #     "split_time": split_time,
    #     "hks_time": hks_time,
    #     "pull_synapses_time": pull_synapses_time,
    #     "predict_time": predict_time,
    #     "raw_n_vertices": raw_n_vertices,
    #     "processed_n_vertices": processed_n_vertices,
    #     "replicas": REPLICAS,
    #     "timestamp": time.time(),
    # }

    timings["root_id"] = root_id
    timings["pull_mesh_time"] = pull_mesh_time
    timings["simplify_time"] = simplify_time
    timings["pull_synapses_time"] = pull_synapses_time
    timings["predict_time"] = predict_time
    timings["component_label_time"] = component_label_time
    timings["raw_n_vertices"] = raw_n_vertices
    timings["processed_n_vertices"] = processed_n_vertices
    timings["replicas"] = REPLICAS
    timings["n_jobs"] = N_JOBS
    timings["timestamp"] = time.time()

    if cf.exists(f"timings/{root_id}_timings.json"):
        cf.delete(f"timings/{root_id}_timings.json")
    cf.put_json(f"timings/{root_id}_timings.json", timings, cache_control="no-cache")

    # if generate_plot:
    #     from meshrep.colors import color_weights, predict_proba_colors
    #     pv.set_jupyter_backend("html")
    #     posteriors = model.predict_proba(np.log(hks))
    #     plotter = pv.Plotter()
    #     plotter.add_mesh(
    #         pv.make_tri_mesh(*mesh), scalars=color_weights(model, posteriors), rgb=True
    #     )
    #     plotter.add_points(
    #         synapses[
    #             ["mesh_pt_position_x", "mesh_pt_position_y", "mesh_pt_position_z"]
    #         ].values,
    #         scalars=predict_proba_colors(model, np.log(hks_by_synapse)),
    #         point_size=10,
    #         rgb=True,
    #     )
    #     # TODO


# %%

TEST = True
if TEST:
    root_id = 864691136237725199
    run_prediction_for_root(root_id)

    def load_dataframe(cf, path, **kwargs):
        bytes_out = cf.get(path)
        with BytesIO(bytes_out) as f:
            df = pd.read_csv(f, **kwargs)
        return df

    back_predictions = load_dataframe(
        cf, f"predictions/{root_id}_synapse_predictions.csv.gz", compression="gzip"
    )

    back_components = load_dataframe(
        cf, f"components/{root_id}_multispine_synapses.csv.gz", compression="gzip"
    )

# %%


vCPU_spot_price = 0.00668  # in dollars
memory_spot_price = 0.000898  # in dollars
# currently using c2d-standard-32 which has 32 vCPUs and 128 GB of memory
total_spot_rate = 32 * vCPU_spot_price + 128 * memory_spot_price  # in dollars per hour
# let's say i can use c2d-highcpu-32 which has 32 vCPUs and 64 GB of memory
# total_spot_rate = 32 * vCPU_spot_price + 64 * memory_spot_price # in dollars per hour

TIMING = False
if TIMING:
    timing_rows = []
    # for file in cf.list("timings"):
    paths = list(cf.list("timings"))
    timing_rows = cf.get_json(paths)
    timing_df = pd.DataFrame(timing_rows)
    timing_df["timestamp"] = timing_df["timestamp"].apply(
        lambda x: datetime.datetime.fromtimestamp(x) if not pd.isnull(x) else x
    )

    timing_cols = [
        col for col in timing_df.columns if "time" in col and "timestamp" not in col
    ]
    timing_df["total"] = timing_df[timing_cols].sum(axis=1)
    timing_df["total_effective"] = timing_df["total"] / timing_df["replicas"]
    relevant_timing_df = timing_df.query("replicas == 32")
    mean_seconds_per_root = relevant_timing_df["total_effective"].mean()
    # price_per_hour = 0.5997 * 0.25
    # https://cloud.google.com/compute/all-pricing#compute-optimized_machine_types
    # price_per_hour = 1.4527 * 0.25
    price_per_hour = total_spot_rate
    neurons_per_hour = 3600 / mean_seconds_per_root
    price_per_root = price_per_hour / neurons_per_hour
    n_roots = 100_000
    total_projected_price = n_roots * price_per_root
    total_projected_price


# %%
tq = TaskQueue(f"https://sqs.us-west-2.amazonaws.com/629034007606/{QUEUE_NAME}")

# %%
REQUEST = True
if REQUEST:
    n_roots = "unfinished"
    # tq.purge()
    types_table = client.materialize.query_table(
        "allen_v1_column_types_slanted_ref",
        # "allen_column_mtypes_v2"
    )
    types_table.query("pt_root_id != 0", inplace=True)
    if n_roots == "all":
        root_ids = types_table["pt_root_id"].tolist()
    elif n_roots == "unfinished":
        root_ids = types_table["pt_root_id"].tolist()
        print(len(root_ids))
        done_files = list(cf.list("predictions"))
        processed_roots = np.array(
            [int(file.split("_")[0].split("/")[1]) for file in done_files]
        )
        print(len(processed_roots))
        root_ids = np.setdiff1d(root_ids, processed_roots)
        print(len(root_ids))
    else:
        counts = types_table["cell_type"].value_counts()
        props = counts / counts.sum()
        weights = types_table["cell_type"].map(props)
        root_ids = types_table.sample(n_roots, weights=weights)["pt_root_id"].tolist()
    tasks = [partial(run_prediction_for_root, root_id) for root_id in root_ids]
    tq.insert(tasks)

# %%


def stop_fn(elapsed_time):
    if elapsed_time > 3600 * 2:
        print("Timed out")
        return True


lease_seconds = 2 * 3600
RUN = True
if RUN:
    tq.poll(lease_seconds=lease_seconds, verbose=False, tally=False)
