# %%
import datetime
import os
import time
from functools import partial
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from caveclient import CAVEclient, set_session_defaults
from cloudfiles import CloudFiles
from fast_simplification import simplify_mesh
from joblib import load
from pymeshfix import MeshFix
from sklearn.neighbors import NearestNeighbors
from taskqueue import TaskQueue, queueable

from meshspice import (
    MeshStitcher,
    compute_hks,
    interpret_mesh,
)

RUN = os.getenv("RUN", "true").lower() == "true"
REQUEST = False
TEST = False
QUEUE_NAME = os.getenv("QUEUE_NAME", "ben-skedit")

REPLICAS = int(os.getenv("REPLICAS", 1))
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"

smooth_n_iter = None
target_reduction = 0.7
agg = 7
n_scales = 64
t_min = 5e4
t_max = 2e7
max_eval = 5e-6
generate_plot = False
overlap_distance = 20_000
vertex_threshold = 20_000
n_jobs = 1

cf = CloudFiles("gs://allen-minnie-phase3/hks")

set_session_defaults(max_retries=5, backoff_factor=0.9)
client = CAVEclient("minnie65_phase3_v1", version=1181)
cv = client.info.segmentation_cloudvolume(progress=False)


def write_dataframe(df, cf, path):
    with BytesIO() as f:
        df.to_csv(f, index=True, compression="gzip")
        cf.put(path, f)


def preprocess_mesh(
    mesh, smooth_n_iter=None, target_reduction=None, agg=False, verbose=False
):
    mesh = interpret_mesh(mesh)
    mesh_poly = pv.make_tri_mesh(mesh[0], mesh[1])
    assert mesh_poly.is_all_triangles
    # hoping to smooth out the mesh a bit and also reduce the number of faces
    if verbose:
        print("Cleaning mesh")
    mesh_poly = mesh_poly.clean()
    if smooth_n_iter is not None:
        if verbose:
            print("Smoothing mesh")
        mesh_poly = mesh_poly.smooth(n_iter=smooth_n_iter)
    if verbose:
        print("Extracting largest connected component")
    mesh_poly = mesh_poly.extract_largest().triangulate()
    if target_reduction is not None:
        if verbose:
            print("Simplifying mesh")
        mesh_poly = simplify_mesh(
            mesh_poly, target_reduction=target_reduction, agg=agg
        ).extract_largest()

    if verbose:
        print("Repairing mesh")
    # was finding that some meshes got eigendecomposition errors before I did this
    mesh_fix = MeshFix(mesh_poly)
    mesh_fix.repair()
    vertices = mesh_fix.points
    faces = mesh_fix.faces
    return vertices, faces


def project_points_to_mesh(
    points, mesh, distance_threshold=None, return_distances=False
):
    if isinstance(mesh, tuple):
        vertices = mesh[0]
    else:
        vertices = mesh.vertices
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(vertices)

    distances, indices = nn.kneighbors(points)
    indices = indices.reshape(-1)
    distances = distances.reshape(-1)
    if distance_threshold is not None:
        indices[distances > distance_threshold] = -1

    if return_distances:
        return indices, distances
    else:
        return indices


def load_synapses(
    root_id,
    mesh,
    client,
    labeled=True,
    distance_threshold=300,
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
    model_path = current / f"models/{model_name}.pkl"
    return load(model_path)


@queueable
def run_prediction_for_root(root_id):
    model = load_model("rf_2024-10-08")

    currtime = time.time()
    raw_mesh = cv.mesh.get(
        root_id, remove_duplicate_vertices=False, deduplicate_chunk_boundaries=False
    )[root_id]
    pull_mesh_time = time.time() - currtime
    mesh = (raw_mesh.vertices, raw_mesh.faces)
    raw_n_vertices = len(raw_mesh.vertices)

    currtime = time.time()
    mesh = preprocess_mesh(
        mesh,
        smooth_n_iter=smooth_n_iter,
        target_reduction=target_reduction,
        agg=agg,
        verbose=VERBOSE,
    )
    processed_n_vertices = len(mesh[0])
    preprocess_time = time.time() - currtime

    stitcher = MeshStitcher(mesh, n_jobs=n_jobs, verbose=VERBOSE)
    currtime = time.time()
    stitcher.split_mesh(
        overlap_distance=overlap_distance, vertex_threshold=vertex_threshold
    )
    split_time = time.time() - currtime

    currtime = time.time()
    hks = stitcher.apply(
        compute_hks,
        n_scales=n_scales,
        t_min=t_min,
        t_max=t_max,
        max_eigenvalue=max_eval,
    )
    hks_time = time.time() - currtime

    currtime = time.time()
    synapses = load_synapses(
        root_id, mesh, client, labeled=False, distance_threshold=300
    )
    pull_synapses_time = time.time() - currtime

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

    write_dataframe(
        predictions_df, cf, f"predictions/{root_id}_synapse_predictions.csv.gz"
    )

    predict_time = time.time() - currtime

    timings = {
        "root_id": root_id,
        "pull_mesh_time": pull_mesh_time,
        "preprocess_time": preprocess_time,
        "split_time": split_time,
        "hks_time": hks_time,
        "pull_synapses_time": pull_synapses_time,
        "predict_time": predict_time,
        "raw_n_vertices": raw_n_vertices,
        "processed_n_vertices": processed_n_vertices,
        "replicas": REPLICAS,
        "timestamp": time.time(),
    }
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

vCPU_spot_price = 0.00668  # in dollars
memory_spot_price = 0.000898  # in dollars
# currently using c2d-standard-32 which has 32 vCPUs and 128 GB of memory
total_spot_rate = 32 * vCPU_spot_price + 128 * memory_spot_price  # in dollars per hour
# let's say i can use c2d-highcpu-32 which has 32 vCPUs and 64 GB of memory
# total_spot_rate = 32 * vCPU_spot_price + 64 * memory_spot_price # in dollars per hour

if TEST:
    root_id = 864691136237725199
    # run_prediction_for_root(root_id)

    def load_dataframe(cf, path, **kwargs):
        bytes_out = cf.get(path)
        with BytesIO(bytes_out) as f:
            df = pd.read_csv(f, **kwargs)
        return df

    # load_dataframe(
    #     cf, f"predictions/{root_id}_synapse_predictions.csv.gz", compression="gzip"
    # )

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


# %%
# QUEUE_NAME = "ben-skedit"
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
