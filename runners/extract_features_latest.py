# %%
import json
import os
from pathlib import Path

import numpy as np
import toml
from caveclient import CAVEclient
from cloudfiles import CloudFiles

from meshmash import (
    chunked_hks_pipeline,
    find_nucleus_point,
    project_points_to_mesh,
    save_condensed_edges,
    save_condensed_features,
)


def replace_none(parameters):
    for key, value in parameters.items():
        if isinstance(value, dict):
            parameters[key] = replace_none(value)
        elif value == "None":
            parameters[key] = None
    return parameters


VERBOSE = 10
N_JOBS = -1
REPLICAS = 1

MODEL_NAME = "foggy-forest-call"

# %%

client = CAVEclient("minnie65_phase3_v1", version=1300)

cv = client.info.segmentation_cloudvolume(progress=False)


model_folder = Path(__file__).parent.parent / "models" / MODEL_NAME

parameters = toml.load(model_folder / "parameters.toml")
parameters = replace_none(parameters)

path = Path(f"bdp-ssa/minnie/{MODEL_NAME}")


# %%
feature_out_path = path / "features"
edge_out_path = path / "edges"
timing_out_path = path / "timings"

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

cf = CloudFiles("gs://" + str(path))

root_id = 1

cf.exists(f"features/{root_id}.npz")

# %%


def load_synapses(
    root_id,
    mesh,
    client,
    distance_threshold=None,
    mapping_column="ctr_pt_position",
):
    post_synapses = client.materialize.query_table(
        "synapses_pni_2",
        filter_equal_dict={"post_pt_root_id": root_id},
        log_warning=False,
        split_positions=True,
        desired_resolution=[1, 1, 1],
    )
    post_synapses.query("pre_pt_root_id != post_pt_root_id", inplace=True)
    post_synapses.set_index("id", inplace=True)

    synapse_locs = post_synapses[
        [f"{mapping_column}_x", f"{mapping_column}_y", f"{mapping_column}_z"]
    ].values

    indices = project_points_to_mesh(
        synapse_locs,
        mesh,
        distance_threshold=distance_threshold,
        return_distances=False,
    )

    post_synapses["mesh_index"] = indices.astype("int32")
    post_synapses.query("mesh_index != -1", inplace=True)

    out = post_synapses["mesh_index"]
    return out


#%%
root_id = 864691137021382510
raw_mesh = cv.mesh.get(root_id, **parameters["cv-mesh-get"])[root_id]
raw_mesh = (raw_mesh.vertices, raw_mesh.faces)

# %%
out = load_synapses(root_id, raw_mesh, client, **parameters["project_points_to_mesh"])
out

#%%

synapse_mapping_data = out.to_frame().reset_index().values

np.savez_compressed("test_synapses.npz", id_to_mesh_index=synapse_mapping_data)

# %%


def run_for_root(root_id):
    print(f"Processing {root_id}")
    out_file = feature_out_path / f"{root_id}.npz"
    edge_file = edge_out_path / f"{root_id}.npz"

    if cf.exists(edge_file):
        print(f"Features already extracted for {root_id}")
        print()
        return

    raw_mesh = cv.mesh.get(root_id, **parameters["cv-mesh-get"])[root_id]
    raw_mesh = (raw_mesh.vertices, raw_mesh.faces)

    indices = project_points_to_mesh(synapse_points, raw_mesh, distance_threshold=1000)

    try:
        nuc_point = find_nucleus_point(root_id, client)
    except Exception as e:
        print(f"Error finding nucleus point for {root_id}: {e}")
        nuc_point = None

    result = chunked_hks_pipeline(
        raw_mesh,
        query_indices=None,
        nuc_point=nuc_point,
        verbose=VERBOSE,
        n_jobs=N_JOBS,
        **parameters["chunked_hks_pipeline"],
    )

    labels = result.labels
    condensed_features = result.condensed_features

    # for consistency with the rest of the pipeline, make sure null label is present
    assert -1 in condensed_features.index

    # log transform to avoid overflow
    condensed_features[non_hks_features] = np.log(condensed_features[non_hks_features])

    save_condensed_features(
        out_file, condensed_features, labels, **parameters["save_condensed_features"]
    )
    print(f"Saved features for {root_id}")

    n_mb = os.path.getsize(out_file) / 1024**2
    print(f"File size: {n_mb:.2f} MB")

    save_condensed_edges(edge_file, result.condensed_edges)

    timing_dict = result.timing_info
    print(timing_dict)
    timing_dict["root_id"] = str(root_id)
    timing_dict["n_mb"] = n_mb
    timing_dict["n_vertices"] = raw_mesh[0].shape[0]
    timing_dict["n_faces"] = raw_mesh[1].shape[0]
    timing_dict["n_condensed_nodes"] = condensed_features.shape[0] - 1
    timing_dict["n_condensed_edges"] = result.condensed_edges.shape[0]

    with open(timing_out_path / f"{root_id}.json", "w") as f:
        json.dump(timing_dict, f)

    print()
