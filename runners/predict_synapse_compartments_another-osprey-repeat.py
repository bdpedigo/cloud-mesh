# %%
import logging
import warnings

import urllib3

urllib3.disable_warnings()

warnings.filterwarnings("ignore", module="urllib3")

# supress warnings for WARNING:urllib3.connectionpool:Connection pool is full...

logging.getLogger("urllib3").setLevel(logging.CRITICAL)

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
from nglui import site_utils
from nglui.statebuilder import (
    AnnotationLayerConfig,
    ChainedStateBuilder,
    PointMapper,
    StateBuilder,
    from_client,
)
from scipy.sparse import csr_array
from scipy.sparse.csgraph import depth_first_order
from taskqueue import TaskQueue, queueable

from cloudigo import put_dataframe
from meshmash import (
    MeshStitcher,
    component_size_transform,
    compute_hks,
    cotangent_laplacian,
    get_label_components,
    project_points_to_mesh,
)


def replace_none(parameters):
    for key, value in parameters.items():
        if isinstance(value, dict):
            parameters[key] = replace_none(value)
        elif value == "None":
            parameters[key] = None
    return parameters


RUN = os.getenv("RUN", "true").lower() == "true"
SAVE_LINK = os.getenv("SAVE_LINK", "false").lower() == "true"
QUEUE_NAME = os.getenv("QUEUE_NAME", "ben-skedit")
REPLICAS = int(os.environ.get("REPLICAS", 1))
N_JOBS = 1 if REPLICAS > 1 else -1
VERBOSE = int(os.environ.get("VERBOSE", 1))
VERSION = 1181

MODEL_NAME = "another-osprey-repeat"
model_folder = Path(__file__).parent.parent / "models" / MODEL_NAME

parameters = toml.load(model_folder / "parameters.toml")
parameters = replace_none(parameters)

cloud_folder = f"gs://allen-minnie-phase3/ben-ssa/{MODEL_NAME}"
cloud_path = Path(cloud_folder)
cf = CloudFiles(f"gs://allen-minnie-phase3/ben-ssa/{MODEL_NAME}")

set_session_defaults(max_retries=5, backoff_factor=0.5, pool_maxsize=20)
client = CAVEclient("minnie65_phase3_v1", version=VERSION)
cv = client.info.segmentation_cloudvolume(progress=False)


feature_columns = [f"hks_{i}" for i in range(parameters["compute_hks"]["n_scales"])] + [
    "component_size",
    "mass",
    "distance_to_nucleus",
]


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
    post_synapses.query("pre_pt_root_id != post_pt_root_id", inplace=True)
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


def compute_distance_to_nucleus(points, root_id):
    nuc_table = client.materialize.query_table(
        "nucleus_detection_v0",
        filter_equal_dict={"pt_root_id": root_id},
        split_positions=True,
        desired_resolution=[1, 1, 1],
    )

    nuc_coords = nuc_table[["pt_position_x", "pt_position_y", "pt_position_z"]].values
    if nuc_coords.shape != (1, 3):
        raise ValueError(f"nuc_coords shape is {nuc_coords.shape}")
    distances = np.linalg.norm(points - nuc_coords, axis=1)
    return distances


def compute_mass(mesh, indices):
    _, M = cotangent_laplacian(
        mesh, robust=True, mollify_factor=parameters["compute_hks"]["mollify_factor"]
    )
    mass = M.diagonal()
    return mass[indices]


def transform(X_df):
    # indices = np.arange(1, 64, 2)
    # X = X[:, indices]

    features = X_df.columns
    hks_features = [f for f in features if f.startswith("hks_")]
    non_hks_features = [f for f in features if not f.startswith("hks_")]
    non_hks_features.remove("mass")

    non_hks_X = X_df[non_hks_features].values
    X = X_df[hks_features].values.copy()

    # TODO unsure if this normalization by mass is a good idea
    # X = X / X_df["mass"].values[:, None]
    X = np.log(X)

    # TODO the diff thing seemed to hurt, but that was also with corrupted data, maybe
    # should retry
    # diff = X[:, 1:] - X[:, :-1]
    X = np.column_stack([X, non_hks_X])
    return X


def render_root(root_id, post_synapses):
    edges = np.array(client.chunkedgraph.level2_chunk_graph(root_id))
    nodes = pd.Index(np.unique(edges))
    # convert to positional indices for the adjacency matrix
    sources = nodes.get_indexer(edges[:, 0]).astype(np.intc)
    targets = nodes.get_indexer(edges[:, 1]).astype(np.intc)
    adjacency = csr_array(
        (np.ones(len(edges)), (sources, targets)), shape=(len(nodes), len(nodes))
    )

    # anchor the DFS on the soma node
    nuc_row = client.materialize.query_table(
        "nucleus_detection_v0", filter_equal_dict={"pt_root_id": root_id}
    )
    sv_id = nuc_row["pt_supervoxel_id"].values[0]
    l2_id = client.chunkedgraph.get_roots(sv_id, stop_layer=2)[0]
    soma_index = nodes.get_loc(l2_id)

    # run the DFS
    order, _ = depth_first_order(adjacency, soma_index, directed=False)
    level2_id_order = nodes[order]

    site_utils.set_default_config(
        target_site="spelunker", datastack_name="minnie65_phase3_v1"
    )

    tag_order = dict(zip(["soma", "shaft", "spine"], range(3)))

    post_synapses["ctr_pt_position"] = post_synapses[
        ["ctr_pt_position_x", "ctr_pt_position_y", "ctr_pt_position_z"]
    ].values.tolist()

    # sorting
    post_synapses["tag_order"] = post_synapses["pred_label"].map(tag_order)
    supervoxels = post_synapses["post_pt_supervoxel_id"]
    post_synapses["post_level2_id"] = client.chunkedgraph.get_roots(
        supervoxels, stop_layer=2
    )
    post_synapses["depth_first_order"] = post_synapses["post_level2_id"].map(
        lambda x: level2_id_order.get_loc(x)
    )

    # post_synapses = post_synapses.sort_values(["tag_order", "depth_first_order"])
    post_synapses = post_synapses.sort_values(
        [
            "depth_first_order",
            "ctr_pt_position_x",
            "ctr_pt_position_y",
            "ctr_pt_position_z",
        ]
    )

    img_layer, seg_layer = from_client(client)
    seg_layer.add_selection_map(fixed_ids=[root_id])

    # component_layer = AnnotationLayerConfig(
    #     name="multispine_components",
    #     mapping_rules=LineMapper(point_column_a="pt_a", point_column_b="pt_b"),
    #     data_resolution=[1, 1, 1],
    #     color=COMPARTMENT_PALETTE_HEX["spine"],
    # )
    # sb = StateBuilder(layers=[img_layer, seg_layer], client=client)

    anno_layer = AnnotationLayerConfig(
        name="labeled_synapses",
        mapping_rules=PointMapper(
            point_column="ctr_pt_position",
            description_column="id",
            tag_column="pred_label",
            set_position=True,
            linked_segmentation_column="pre_pt_root_id",
        ),
        tags=["soma", "shaft", "spine"],
        data_resolution=[1, 1, 1],
    )
    # sb2 = StateBuilder(layers=[anno_layer], client=client)

    sb = StateBuilder(layers=[img_layer, seg_layer, anno_layer], client=client)

    corrections_layer = AnnotationLayerConfig(
        name="corrected_labels",
        mapping_rules=PointMapper(
            point_column="ctr_pt_position",
            description_column="id",
            # tag_column="pred_label",
            set_position=True,
            linked_segmentation_column="pre_pt_root_id",
        ),
        tags=["soma", "shaft", "spine"],
        data_resolution=[1, 1, 1],
    )
    sb2 = StateBuilder(layers=[corrections_layer], client=client)

    cb = ChainedStateBuilder([sb, sb2])

    state_dict = cb.render_state(
        [post_synapses.reset_index(), post_synapses.reset_index()],
        return_as="dict",
        client=client,
    )

    shader1 = """
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
    shader2 = """
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
    setPointMarkerSize(5.0);
    }
    """
    state_dict["layers"][-2]["shader"] = shader1
    state_dict["layers"][-1]["shader"] = shader2
    state_dict["layers"][1]["objectAlpha"] = 0.9
    state_dict["layout"] = "3d"

    annotation_id = state_dict["layers"][-1]["annotations"][0]["id"]
    state_dict["selection"] = {
        "layers": {
            "corrected_labels": {
                "annotationId": annotation_id,
                "annotationSource": 0,
                "annotationSubsource": "default",
            }
        }
    }
    state_id = client.state.upload_state_json(state_dict)
    base_url = "https://spelunker.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/"
    url = base_url + str(state_id)
    return url


@queueable
def run_prediction_for_root(root_id):
    info = {}
    info["root_id"] = root_id
    info["model"] = MODEL_NAME

    model = load_model(MODEL_NAME)

    if VERBOSE:
        print("Pulling mesh...")
    currtime = time.time()
    raw_mesh = cv.mesh.get(root_id, **parameters["cv-mesh-get"])[root_id]
    info["pull_mesh_time"] = time.time() - currtime
    mesh = (raw_mesh.vertices, raw_mesh.faces)
    info["raw_n_vertices"] = len(raw_mesh.vertices)

    if VERBOSE:
        print("Simplifying mesh...")
    currtime = time.time()
    mesh = simplify(mesh[0], mesh[1], **parameters["simplify"])
    info["processed_n_vertices"] = len(mesh[0])
    info["simplify_time"] = time.time() - currtime

    currtime = time.time()
    if VERBOSE:
        print("Loading synapses...")
    synapses = load_synapses(
        root_id,
        mesh,
        client,
        labeled=False,
        distance_threshold=parameters["project_points_to_mesh"]["distance_threshold"],
    )
    info["pull_synapses_time"] = time.time() - currtime

    currtime = time.time()
    if VERBOSE:
        print("Splitting mesh...")
    stitcher = MeshStitcher(mesh, n_jobs=N_JOBS, verbose=VERBOSE)
    stitcher.split_mesh(
        **parameters["split_mesh"],
    )
    info["split_time"] = time.time() - currtime

    currtime = time.time()
    if VERBOSE:
        print("Computing HKS...")
    X = stitcher.subset_apply(
        compute_hks,
        synapses["mesh_index"],
        reindex=False,
        **parameters["compute_hks"],
    )
    info["hks_time"] = time.time() - currtime

    X_df = pd.DataFrame(X, columns=[f"hks_{i}" for i in range(X.shape[1])])

    currtime = time.time()
    aux_X = []
    aux_X_features = []
    component_sizes = component_size_transform(mesh, np.arange(len(mesh[0])))
    aux_X.append(component_sizes)
    aux_X_features.append("component_size")

    mass = compute_mass(mesh, np.arange(len(mesh[0])))
    aux_X.append(mass)
    aux_X_features.append("mass")

    distances_to_nuc = compute_distance_to_nucleus(mesh[0], root_id)
    aux_X.append(distances_to_nuc)
    aux_X_features.append("distance_to_nucleus")

    aux_X = np.column_stack(aux_X)
    aux_X_df = pd.DataFrame(aux_X, columns=aux_X_features)

    info["aux_features_time"] = time.time() - currtime

    X_df = pd.concat([X_df, aux_X_df], axis=1)
    X_df = X_df[feature_columns]

    transformed_X = transform(X_df)

    # hks, timings = chunked_hks_pipeline(
    #     mesh,
    #     mesh_indices=synapses["mesh_index"],
    #     return_timing=True,
    #     verbose=VERBOSE,
    #     n_jobs=N_JOBS,
    #     **parameters["chunked_hks_pipeline"],
    # )

    if VERBOSE:
        print("Computing predictions...")
    currtime = time.time()
    predictions_df = pd.DataFrame(index=synapses.index)
    indices = synapses["mesh_index"]

    posteriors = model.predict_proba(transformed_X[indices])
    classes = model.classes_
    max_inds = np.argmax(posteriors, axis=1)
    predictions = classes[max_inds]
    predictions_df["pred_label"] = predictions
    max_posteriors = posteriors[np.arange(len(posteriors)), max_inds]
    predictions_df["pred_label_posterior"] = max_posteriors

    if VERBOSE:
        print("Saving predictions...")
    put_dataframe(
        predictions_df, cloud_path / f"predictions/{root_id}_synapse_predictions.csv.gz"
    )

    info["predict_time"] = time.time() - currtime

    if VERBOSE:
        print("Labeling components...")
    currtime = time.time()
    pred_labels = model.predict(transformed_X)
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

    if VERBOSE:
        print("Saving components...")
    put_dataframe(
        multispine_synapses,
        cloud_path / f"components/{root_id}_multispine_synapses.csv.gz",
    )
    info["component_label_time"] = time.time() - currtime

    # try:
    #     replicas = get_replicas_on_node()
    #     print("Found replicas:", replicas)
    # except Exception as e:
    #     print("Could not find replicas, using default:", REPLICAS)
    #     print(e)
    replicas = REPLICAS
    info["replicas"] = replicas
    info["n_jobs"] = N_JOBS
    info["timestamp"] = time.time()

    if SAVE_LINK:
        currtime = time.time()
        try:
            url = render_root(root_id, synapses)
            info["url"] = url
        except Exception as e:
            print(e)
            info["url"] = None
        info["render_time"] = time.time() - currtime

    if VERBOSE:
        print("Saving timings...")
    if cf.exists(f"timings/{root_id}_timings.json"):
        cf.delete(f"timings/{root_id}_timings.json")
    cf.put_json(f"timings/{root_id}_timings.json", info, cache_control="no-cache")

    return info


# %%
TEST = False
if TEST:
    root_id = 864691136237725199
    info = run_prediction_for_root(root_id)

    def load_dataframe(cf, path, **kwargs):
        bytes_out = cf.get(path)
        with BytesIO(bytes_out) as f:
            df = pd.read_csv(f, **kwargs)
        return df

    print("reading predictions")
    back_predictions = load_dataframe(
        cf, f"predictions/{root_id}_synapse_predictions.csv.gz", compression="gzip"
    )

    print("reading components")
    back_components = load_dataframe(
        cf, f"components/{root_id}_multispine_synapses.csv.gz", compression="gzip"
    )

    print("done")

# %%
tq = TaskQueue(f"https://sqs.us-west-2.amazonaws.com/629034007606/{QUEUE_NAME}")

# %%

REQUEST = False
if REQUEST and not RUN:
    n_roots = "all"
    # tq.purge()
    types_table = client.materialize.query_table(
        "aibs_metamodel_mtypes_v661_v2",
        # "allen_v1_column_types_slanted_ref",
        # "allen_column_mtypes_v2"
    )
    types_table.query("pt_root_id != 0", inplace=True)
    types_table.drop_duplicates("pt_root_id", inplace=True)
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

if RUN:
    tq.poll(lease_seconds=lease_seconds, verbose=False, tally=False)

# %%
