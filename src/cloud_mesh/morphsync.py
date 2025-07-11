from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from caveclient import CAVEclient
from cloudfiles import CloudFiles
from tqdm import tqdm

from morphsync import MorphSync


def path_to_cloudfiles(path: Union[str, Path], **kwargs) -> Path:
    if isinstance(path, str):
        path = Path(path)

    if path.suffix == "":
        path = path
    else:
        raise ValueError("Path has a file suffix, should be a directory.")

    path_str = str(path)
    splits = path_str.split(":/")
    path_str = splits[0] + "://" + splits[1]
    cf = CloudFiles(path_str, **kwargs)

    return cf


def read_header(cf: CloudFiles, header_file_name="header.txt") -> list[str]:
    header_bytes = cf.get([header_file_name])
    if isinstance(header_bytes, list):
        header_bytes = header_bytes[0]["content"]
    if header_bytes is None:
        return None
    header = header_bytes.decode()
    columns = header.split("\t")
    return columns


def listify(morphs):
    if isinstance(morphs, MorphSync):
        return [morphs]
    else:
        return morphs


class MorphClient:
    """
    A client for interacting with morphology data.
    """

    def __init__(self, datastack: str, hks_parameters: Optional[str] = None):
        self.datastack = datastack
        self.hks_parameters = hks_parameters
        self.caveclient = CAVEclient(datastack)

        if hks_parameters is not None:
            self.hks_base_path = f"gs://bdp-ssa/{datastack}/{hks_parameters}/"
            self.hks_cloudfiles = path_to_cloudfiles(
                self.hks_base_path,
                progress=True,
                green=False,
                num_threads=20,
                parallel=20,
            )

    def add_synapses(
        self,
        morphs: list[MorphSync],
        side="pre",
        timestamp=None,
    ):
        """Add pre or post synapses to the MorphSync object."""
        morphs = listify(morphs)
        kwargs = {
            "desired_resolution": [1, 1, 1],
            "split_positions": True,
            "timestamp": timestamp,
        }
        for morph in morphs:
            root_id = morph.name
            if side == "pre":
                kwargs["pre_ids"] = root_id
            elif side == "post":
                kwargs["post_ids"] = root_id
            synapses = (
                self.caveclient.materialize.synapse_query(**kwargs)
                .set_index("id")
                .sort_index()
            )
            keep_cols = [
                "ctr_pt_position_x",
                "ctr_pt_position_y",
                "ctr_pt_position_z",
                "post_pt_root_id",
                "post_pt_supervoxel_id",
                "pre_pt_root_id",
                "pre_pt_supervoxel_id",
                "size",
            ]
            morph.add_points(
                synapses[keep_cols],
                spatial_columns=[
                    "ctr_pt_position_x",
                    "ctr_pt_position_y",
                    "ctr_pt_position_z",
                ],
                name=f"{side}_synapses",
            )

    def get_hks(self, root_ids):
        hks_columns = read_header(
            self.hks_cloudfiles, header_file_name="features/header.txt"
        )
        feature_datas = self.hks_cloudfiles.get(
            [f"features/{root_id}.npz" for root_id in root_ids],
            raise_errors=False,
        )
        features_by_post_root = {}
        for back_data in tqdm(feature_datas):
            this_root_id = int(back_data["path"].split("/")[-1].split(".")[0])

            content = back_data["content"]
            if content is None:
                continue
            if len(content) == 0:
                continue
            with BytesIO(content) as bio:
                data = np.load(bio)
                X = data["X"]
                labels = data["labels"]
            index = np.arange(-1, len(X) - 1)
            features = pd.DataFrame(X, columns=hks_columns, index=index)
            features_by_post_root[this_root_id] = (features, labels)
        return features_by_post_root

    def add_hks(self, morphs: list[MorphSync]):
        """
        Add HKS features to the MorphSync object.
        """
        morphs = listify(morphs)
        root_ids = [morph.name for morph in morphs]
        features_by_root = self.get_hks(root_ids)

        for morph in morphs:
            root_id = morph.name
            if root_id not in features_by_root:
                continue
            features, labels = features_by_root[root_id]

            morph.add_table(features, name="hks_features")

            n_mesh_nodes = len(labels)
            dummy_mesh = (
                pd.DataFrame(index=pd.RangeIndex(n_mesh_nodes)),
                pd.DataFrame(),
            )
            morph.add_mesh(dummy_mesh, name="mesh", relation_columns=[])
            morph.add_link("mesh", "hks_features", mapping=labels, reciprocal=True)

    def get_synapse_mesh_mappings(self, root_ids, side="pre"):
        """
        Get synapse mesh mappings for the given root IDs.
        """
        synapse_mappings = self.hks_cloudfiles.get(
            [f"{side}-synapse-mappings/{root_id}.npz" for root_id in root_ids],
            raise_errors=False,
        )
        id_to_mesh_maps_by_root = {}
        for back_data in tqdm(synapse_mappings):
            this_root_id = int(back_data["path"].split("/")[-1].split(".")[0])
            content = back_data["content"]
            if content is None:
                continue
            if len(content) == 0:
                continue
            with BytesIO(content) as bio:
                data = np.load(bio)
                id_to_mesh_map = data["id_to_mesh_map"]
            id_to_mesh_maps_by_root[this_root_id] = id_to_mesh_map
        return id_to_mesh_maps_by_root

    def add_synapse_mesh_mappings(
        self,
        morphs: list[MorphSync],
        side="pre",
    ):
        """
        Add synapse mesh mappings to the MorphSync object.
        """
        morphs = listify(morphs)
        root_ids = [morph.name for morph in morphs]

        id_to_mesh_maps_by_root = self.get_synapse_mesh_mappings(root_ids, side=side)
        for morph in morphs:
            root_id = morph.name

            if root_id not in id_to_mesh_maps_by_root:
                continue

            id_to_mesh_map = id_to_mesh_maps_by_root[root_id]
            if len(id_to_mesh_map) == 0:
                continue

            id_to_mesh_map = pd.Series(
                index=id_to_mesh_map[:, 0], data=id_to_mesh_map[:, 1]
            )
            if not hasattr(morph, f"{side}_synapses"):
                # add dummy synapses in case we aren't going to look them up later
                morph.add_points(
                    pd.DataFrame(index=id_to_mesh_map.index),
                    name=f"{side}_synapses",
                )

            morph.add_link(
                f"{side}_synapses", "mesh", mapping=id_to_mesh_map, reciprocal=True
            )

    def get_supermoxel_graphs(self, root_ids):
        """
        Get supermoxel graphs for the given root IDs.
        """
        node_columns = read_header(
            self.hks_cloudfiles, header_file_name="graphs/nodes_header.txt"
        )
        edge_columns = read_header(
            self.hks_cloudfiles, header_file_name="graphs/edges_header.txt"
        )
        supermoxel_graphs = self.hks_cloudfiles.get(
            [f"graphs/{root_id}.npz" for root_id in root_ids],
            raise_errors=False,
        )
        graphs_by_root = {}
        for back_data in tqdm(supermoxel_graphs):
            this_root_id = int(back_data["path"].split("/")[-1].split(".")[0])
            content = back_data["content"]
            if content is None:
                continue
            if len(content) == 0:
                continue
            with BytesIO(content) as bio:
                data = np.load(bio)
                nodes = data["nodes"]
                nodes = pd.DataFrame(nodes, columns=node_columns)
                edges = data["edges"]
                edges = pd.DataFrame(edges, columns=edge_columns)
            graphs_by_root[this_root_id] = (nodes, edges)
        return graphs_by_root

    def add_supermoxel_graph(self, morphs: list[MorphSync]):
        """
        Add supermoxel graph to the MorphSync object.
        """
        morphs = listify(morphs)
        root_ids = [morph.name for morph in morphs]

        graphs_by_root = self.get_supermoxel_graphs(root_ids)
        for morph in morphs:
            root_id = morph.name

            if root_id not in graphs_by_root:
                continue

            nodes, edges = graphs_by_root[root_id]
            morph.add_graph(
                (nodes, edges),
                name="supermoxel_graph",
                spatial_columns=["x", "y", "z"],
                relation_columns=["source", "target"],
            )
            morph.add_link(
                "supermoxel_graph",
                "hks_features",
                mapping=nodes.index.values,
                reciprocal=True,
            )

    def add_synapse_skeleton_mappings(
        self, morphs: list[MorphSync], side="pre", timestamp=None
    ):
        morphs = listify(morphs)
        # TODO parallelize / chunk root lookups
        for morph in morphs:
            synapses = morph.__getattribute__(f"{side}_synapses").nodes
            synapses[f"{side}_pt_level2_id"] = self.caveclient.chunkedgraph.get_roots(
                synapses[f"{side}_pt_supervoxel_id"], timestamp=timestamp, stop_layer=2
            )
            morph.add_link(
                f"{side}_synapses",
                "level2_nodes",
                mapping=synapses[f"{side}_pt_level2_id"].values,
            )

    def add_level2_skeleton(self, morphs: list[MorphSync]):
        """
        Add the level 2 skeleton to the MorphSync object.
        """
        morphs = listify(morphs)
        for morph in morphs:
            root_id = morph.name
            # TODO way of getting multiple skeletons at once?
            skel_dict = self.caveclient.skeleton.get_skeleton(root_id)
            edges = skel_dict["edges"]
            vertices = skel_dict["vertices"]
            l2_node_indices = skel_dict["mesh_to_skel_map"]
            l2_node_ids = skel_dict["lvl2_ids"]

            morph.add_graph(
                (vertices, edges),
                name="level2_skeleton",
            )
            morph.add_table(pd.DataFrame(index=l2_node_ids), name="level2_nodes")
            morph.add_link("level2_nodes", "level2_skeleton", mapping=l2_node_indices)

    def add_mesh(self, morphs: list[MorphSync]):
        """
        Add mesh to the MorphSync object.
        """
        cv = self.caveclient.info.segmentation_cloudvolume(progress=False)
        morphs = listify(morphs)
        # TODO use cloudvolume for getting many meshes at once
        for morph in morphs:
            root_id = morph.name
            mesh = cv.mesh.get(
                root_id,
                deduplicate_chunk_boundaries=False,
                remove_duplicate_vertices=True,
            )[root_id]
            mesh = (mesh.vertices, mesh.faces)
            morph.add_mesh(mesh, name="mesh")

