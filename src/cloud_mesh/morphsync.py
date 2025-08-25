from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from caveclient import CAVEclient
from cloudfiles import CloudFiles
from joblib import Parallel, cpu_count, delayed
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

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


def int_listify(morphs):
    if isinstance(morphs, int):
        return [morphs]
    else:
        return morphs


def get_parallel(n_jobs: Optional[int] = None) -> int:
    if n_jobs == 1:
        return False
    elif n_jobs is None:
        return cpu_count()
    elif n_jobs < 1:
        return cpu_count() + n_jobs + 1
    else:
        return n_jobs


class MorphClient:
    """
    A client for interacting with morphology data and applying to MorphSync objects.
    """

    def __init__(
        self,
        datastack: str,
        hks_parameters: Optional[str] = None,
        model_name: Optional[str] = None,
        model_target: Optional[str] = None,
        verbose: bool = True,
        n_jobs: int = 1,
    ):
        self.datastack = datastack
        self.hks_parameters = hks_parameters
        self.model_name = model_name
        self.model_target = model_target
        self.caveclient = CAVEclient(datastack)
        self.verbose = verbose
        self.n_jobs = n_jobs
        self._parallel = None
        self._cloudvolume = None

        if hks_parameters is not None:
            self.hks_base_path = f"gs://bdp-ssa/{datastack}/{hks_parameters}/"
            self.hks_cloudfiles = path_to_cloudfiles(
                self.hks_base_path,
                progress=self.verbose > 0,
                green=False,
                # num_threads=20,
                parallel=self.parallel,
            )

    @property
    def parallel(self):
        if self._parallel is not None:
            return self._parallel
        else:
            self._parallel = get_parallel(self.n_jobs)
            return self._parallel

    @property
    def cloudvolume(self):
        """
        Get the cloudvolume for the current datastack.
        """
        if self._cloudvolume is None:
            self._cloudvolume = self.caveclient.info.segmentation_cloudvolume(
                progress=self.verbose > 0, parallel=self.parallel
            )
        return self._cloudvolume

    def _get_synapses(self, root_id, side="pre", timestamp=None):
        kwargs = {
            "desired_resolution": [1, 1, 1],
            "split_positions": True,
            "timestamp": timestamp,
        }
        if side == "pre":
            kwargs["pre_ids"] = root_id
        elif side == "post":
            kwargs["post_ids"] = root_id
        synapses = (
            self.caveclient.materialize.synapse_query(**kwargs)
            .set_index("id")
            .sort_index()
        )
        return synapses

    def get_synapses(self, root_ids, side="pre", timestamp=None):
        """
        Get pre or post synapses for the given root IDs.
        """
        root_ids = int_listify(root_ids)

        if self.n_jobs == 1:
            synapses_by_root = {}
            for root_id in tqdm(
                root_ids, disable=not self.verbose, desc=f"Getting {side} synapses"
            ):
                synapses = self._get_synapses(root_id, side=side, timestamp=timestamp)
                synapses_by_root[root_id] = synapses
        else:
            with tqdm_joblib(
                desc=f"Getting {side} synapses",
                total=len(root_ids),
                disable=not self.verbose,
            ):
                synapses_by_root = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._get_synapses)(root_id, side=side, timestamp=timestamp)
                    for root_id in root_ids
                )
            synapses_by_root = dict(zip(root_ids, synapses_by_root))

        return synapses_by_root

    def add_synapses(
        self,
        morphs: list[MorphSync],
        side="pre",
        timestamp=None,
        synapses: Optional[pd.DataFrame] = None,
    ):
        """Add pre or post synapses to the MorphSync object."""
        # keep_cols = [
        #     "ctr_pt_position_x",
        #     "ctr_pt_position_y",
        #     "ctr_pt_position_z",
        #     "post_pt_root_id",
        #     "post_pt_supervoxel_id",
        #     "pre_pt_root_id",
        #     "pre_pt_supervoxel_id",
        #     "size",
        # ]
        drop_cols = ["created", "updated", "valid"]

        morphs = listify(morphs)
        root_ids = [morph.name for morph in morphs]
        if synapses is None:
            synapses_by_root = self.get_synapses(
                [morph.name for morph in morphs], side=side, timestamp=timestamp
            )
        else:
            synapses_by_root = {}
            for root_id, sub_synapses in synapses.groupby("post_pt_root_id"):
                synapses_by_root[root_id] = sub_synapses
            for root_id in root_ids:
                if root_id not in synapses_by_root:
                    synapses_by_root[root_id] = pd.DataFrame(columns=synapses.columns)
        for morph in morphs:
            root_id = morph.name
            synapses = synapses_by_root[root_id]
            synapses = synapses.drop(columns=drop_cols, errors="ignore")
            morph.add_points(
                synapses,
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
        for back_data in tqdm(
            feature_datas, disable=not self.verbose, desc="Getting HKS features"
        ):
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

    def has_hks(self, root_ids):
        """
        Check if HKS features are available for the given root IDs.
        """
        queries = [f"features/{root_id}.npz" for root_id in root_ids]
        does_exist = self.hks_cloudfiles.exists(queries)
        exists_indicator = np.vectorize(does_exist.get)(queries)
        return exists_indicator

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
            if not hasattr(morph, "mesh"):
                dummy_mesh = (
                    pd.DataFrame(index=pd.RangeIndex(n_mesh_nodes)),
                    pd.DataFrame(),
                )
                morph.add_mesh(dummy_mesh, name="mesh", relation_columns=[])
            morph.add_link("mesh", "hks_features", mapping=labels, reciprocal=True)

    def has_synapse_mesh_mappings(self, root_ids, side="pre"):
        """
        Check if synapse mesh mappings are available for the given root IDs.
        """
        queries = [f"{side}-synapse-mappings/{root_id}.npz" for root_id in root_ids]
        does_exist = self.hks_cloudfiles.exists(queries)
        exists_indicator = np.vectorize(does_exist.get)(queries)
        return exists_indicator

    def get_synapse_mesh_mappings(self, root_ids, side="pre"):
        """
        Get synapse mesh mappings for the given root IDs.
        """
        synapse_mappings = self.hks_cloudfiles.get(
            [f"{side}-synapse-mappings/{root_id}.npz" for root_id in root_ids],
            raise_errors=False,
        )
        id_to_mesh_maps_by_root = {}
        for back_data in tqdm(
            synapse_mappings,
            disable=not self.verbose,
            desc=f"Getting {side} synapse mesh mappings",
        ):
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
        for back_data in tqdm(
            supermoxel_graphs,
            disable=not self.verbose,
            desc="Getting supermoxel graphs",
        ):
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

    def get_synapse_skeleton_mappings(self, supervoxel_ids, timestamp=None):
        batch_size = 10_000
        supervoxel_batches = np.array_split(
            supervoxel_ids, len(supervoxel_ids) // batch_size + 1
        )
        supervoxel_mappings = {}
        if self.n_jobs == 1:
            for supervoxel_batch in tqdm(
                supervoxel_batches,
                disable=not self.verbose,
                desc="Getting synapse skeleton mappings",
            ):
                batch_level2_ids = self.caveclient.chunkedgraph.get_roots(
                    supervoxel_batch, timestamp=timestamp, stop_layer=2
                )
                for sv_id, l2_id in zip(supervoxel_batch, batch_level2_ids):
                    supervoxel_mappings[sv_id] = l2_id
        else:
            with tqdm_joblib(
                desc="Getting synapse skeleton mappings",
                total=len(supervoxel_batches),
                disable=not self.verbose,
            ):
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(self.caveclient.chunkedgraph.get_roots)(
                        supervoxel_batch, timestamp=timestamp, stop_layer=2
                    )
                    for supervoxel_batch in supervoxel_batches
                )
            for supervoxel_batch, batch_level2_ids in zip(supervoxel_batches, results):
                for sv_id, l2_id in zip(supervoxel_batch, batch_level2_ids):
                    supervoxel_mappings[sv_id] = l2_id

        supervoxel_mappings = pd.Series(supervoxel_mappings, name="level2_id")
        supervoxel_mappings.index.name = "supervoxel_id"
        return supervoxel_mappings

    # def _add_synapse_skeleton_mappings(
    #     self, morph: MorphSync, side="pre", timestamp=None
    # ):
    #     """
    #     Add synapse skeleton mappings to a single MorphSync object.
    #     """
    #     synapses = morph.get_layer(f"{side}_synapses").nodes
    #     synapses[f"{side}_pt_level2_id"] = self.caveclient.chunkedgraph.get_roots(
    #         synapses[f"{side}_pt_supervoxel_id"], timestamp=timestamp, stop_layer=2
    #     )
    #     morph.add_link(
    #         f"{side}_synapses",
    #         "level2_nodes",
    #         mapping=synapses[f"{side}_pt_level2_id"].values,
    #     )

    def add_synapse_skeleton_mappings(
        self, morphs: list[MorphSync], side="pre", timestamp=None
    ):
        morphs = listify(morphs)
        all_supervoxel_ids = []
        for morph in morphs:
            supervoxel_ids = (
                morph.get_layer(f"{side}_synapses")
                .nodes[f"{side}_pt_supervoxel_id"]
                .values
            )
            all_supervoxel_ids.extend(supervoxel_ids)
        all_supervoxel_ids = np.unique(all_supervoxel_ids)

        supervoxel_mappings = self.get_synapse_skeleton_mappings(
            all_supervoxel_ids, timestamp=timestamp
        )
        for morph in morphs:
            synapses = morph.get_layer(f"{side}_synapses").nodes
            synapses[f"{side}_pt_level2_id"] = synapses[f"{side}_pt_supervoxel_id"].map(
                supervoxel_mappings
            )
            mapping = synapses[f"{side}_pt_level2_id"].values
            morph.add_link(
                f"{side}_synapses",
                "level2_nodes",
                mapping=mapping,
            )

        # if self.n_jobs == 1:
        #     for morph in morphs:
        #         self._add_synapse_skeleton_mappings(
        #             morph, side=side, timestamp=timestamp
        #         )
        # else:
        #     with tqdm_joblib(
        #         desc=f"Adding {side} synapse skeleton mappings",
        #         total=len(morphs),
        #         disable=not self.verbose,
        #     ):
        #         Parallel(n_jobs=self.n_jobs)(
        #             delayed(self._add_synapse_skeleton_mappings)(
        #                 morph, side=side, timestamp=timestamp
        #             )
        #             for morph in morphs
        #         )

    def get_level2_skeleton(self, root_ids: list):
        """
        Get the level 2 skeleton for the given MorphSync objects.
        """
        if self.n_jobs == 1:
            skel_dicts = {}
            for root_id in tqdm(
                root_ids, disable=not self.verbose, desc="Getting level 2 skeletons"
            ):
                skel_dict = self.caveclient.skeleton.get_skeleton(root_id)
                skel_dicts[root_id] = skel_dict
        else:
            with tqdm_joblib(
                desc="Getting level 2 skeletons",
                total=len(root_ids),
                disable=not self.verbose,
            ):
                skel_dicts = Parallel(n_jobs=self.n_jobs)(
                    delayed(self.caveclient.skeleton.get_skeleton)(root_id)
                    for root_id in root_ids
                )
            skel_dicts = dict(zip(root_ids, skel_dicts))

        return skel_dicts

    def add_level2_skeleton(self, morphs: list[MorphSync]):
        """
        Add the level 2 skeleton to the MorphSync object.
        """
        morphs = listify(morphs)
        root_ids = [morph.name for morph in morphs]
        skel_dicts_by_root = self.get_level2_skeleton(root_ids)
        for morph in morphs:
            root_id = morph.name
            if root_id not in skel_dicts_by_root:
                continue

            skel_dict = skel_dicts_by_root[root_id]
            vertices = skel_dict["vertices"]
            edges = skel_dict["edges"]
            l2_node_indices = skel_dict["mesh_to_skel_map"]
            l2_node_ids = skel_dict["lvl2_ids"]
            root = skel_dict["root"]

            nodes = pd.DataFrame(vertices, columns=["x", "y", "z"])
            nodes["compartment"] = skel_dict["compartment"]
            nodes["radius"] = skel_dict["radius"]
            nodes["is_root"] = False
            nodes.loc[root, "is_root"] = True

            edge_lengths = np.linalg.norm(
                vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1
            )
            edges = pd.DataFrame(edges, columns=["source", "target"])
            edges["length"] = edge_lengths
            edges["radius"] = (
                nodes.loc[edges["source"], "radius"].values
                + nodes.loc[edges["target"], "radius"].values
            ) / 2

            morph.add_graph(
                (nodes, edges),
                name="level2_skeleton",
                spatial_columns=["x", "y", "z"],
                relation_columns=["source", "target"],
            )
            morph.add_table(pd.DataFrame(index=l2_node_ids), name="level2_nodes")
            morph.add_link("level2_nodes", "level2_skeleton", mapping=l2_node_indices)

    def get_level2_nodes(self, root_ids, root_l2_map=None):
        # TODO this needs to be chunked for large numbers of l2_ids and parallelized
        if root_l2_map is None:
            l2_ids_by_root: dict[int, np.ndarray] = (
                self.caveclient.chunkedgraph.get_leaves_many(root_ids, stop_layer=2)
            )

            l2_ids = []
            root_ids = []
            for root_id, sub_l2_ids in l2_ids_by_root.items():
                l2_ids.extend(sub_l2_ids)
                root_ids.extend([root_id] * len(sub_l2_ids))
            l2_ids = np.array(l2_ids)
            root_l2_map = pd.Series(index=root_ids, data=l2_ids, name="level2_id")
            root_l2_map.index.name = "root_id"

        l2_root_map = root_l2_map.reset_index().set_index("level2_id")
        l2_ids = np.unique(l2_ids)
        l2_data_table = self.caveclient.l2cache.get_l2data_table(
            l2_ids, split_columns=True
        ).rename_axis(index="level2_id")
        l2_data_table.drop(
            columns=["chunk_intersect_count", "pca_val"], inplace=True, errors="ignore"
        )
        l2_data_by_root = {}
        for root_id, root_data in l2_data_table.groupby(
            l2_data_table.index.map(l2_root_map["root_id"])
        ):
            l2_data_by_root[root_id] = root_data

        return l2_data_by_root

    def add_level2_nodes(self, morphs: list[MorphSync]):
        """
        Add level 2 nodes to the MorphSync object.
        """
        morphs = listify(morphs)
        root_ids = [morph.name for morph in morphs]
        # TODO should not have to look up l2 IDS if already have skeleton
        l2_data_by_root = self.get_level2_nodes(root_ids)

        for morph in morphs:
            root_id = morph.name
            if root_id not in l2_data_by_root:
                continue
            l2_data = l2_data_by_root[root_id]

            morph.add_points(
                l2_data,
                name="level2_nodes",
                spatial_columns=["rep_coord_nm_x", "rep_coord_nm_y", "rep_coord_nm_z"],
            )

    def get_mesh(self, root_ids) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """
        Get the mesh for the given root ID.
        """
        cv = self.cloudvolume
        # with warnings.catch_warnings():
        #     # ignore warnings that are "WARNING:urllib3.connectionpool:Connection pool is full"
        meshes = cv.mesh.get(
            root_ids, deduplicate_chunk_boundaries=False, remove_duplicate_vertices=True
        )
        meshes = {
            root_id: (mesh.vertices, mesh.faces) for root_id, mesh in meshes.items()
        }

        return meshes

    def add_mesh(self, morphs: list[MorphSync]):
        """
        Add mesh to the MorphSync object.
        """
        morphs = listify(morphs)
        meshes_by_root = self.get_mesh([morph.name for morph in morphs])
        for morph in morphs:
            mesh = meshes_by_root.get(morph.name, None)
            if mesh is None:
                continue
            morph.add_mesh(mesh, name="mesh")

    def get_component_mappings(self, root_ids):
        root_ids = int_listify(root_ids)
        prefix = f"{self.model_name}/{self.model_target}-component-mappings/"
        queries = [prefix + f"{root_id}.npz" for root_id in root_ids]
        component_datas = self.hks_cloudfiles.get(
            queries,
            raise_errors=False,
        )
        components_by_root = {}
        for back_data in tqdm(
            component_datas, disable=not self.verbose, desc="Getting component mappings"
        ):
            this_root_id = int(back_data["path"].split("/")[-1].split(".")[0])

            content = back_data["content"]
            if content is None:
                continue
            if len(content) == 0:
                continue
            with BytesIO(content) as bio:
                data = np.load(bio)
                id_to_component_map = data["array"]
            components_by_root[this_root_id] = id_to_component_map

        return components_by_root

    def add_component_mappings(self, morphs: list[MorphSync]):
        """
        Add component mappings to the MorphSync object.
        """
        morphs = listify(morphs)
        root_ids = [morph.name for morph in morphs]
        components_by_root = self.get_component_mappings(root_ids)
        for morph in morphs:
            root_id = morph.name
            if root_id not in components_by_root:
                continue
            id_to_component_map = components_by_root[root_id]

            if len(id_to_component_map) == 0:
                continue

            morph.add_link(
                "mesh",
                f"{self.model_target}_morphometry",
                mapping=id_to_component_map,
                reciprocal=True,
            )

    def get_morphometry(self, root_ids):
        root_ids = int_listify(root_ids)
        prefix = f"{self.model_name}/{self.model_target}-morphometry/"
        queries = [prefix + f"{root_id}.csv.gz" for root_id in root_ids]
        feature_datas = self.hks_cloudfiles.get(
            queries,
            raise_errors=False,
        )
        features_by_root = {}
        for back_data in tqdm(
            feature_datas, disable=not self.verbose, desc="Getting morphometry features"
        ):
            this_root_id = int(back_data["path"].split("/")[-1].split(".")[0])

            content = back_data["content"]
            if content is None:
                continue
            if len(content) == 0:
                continue
            with BytesIO(content) as bio:
                df = pd.read_csv(
                    bio,
                    # header=None,
                    compression="gzip",
                ).set_index("component_id")
                features_by_root[this_root_id] = df

        return features_by_root

    def add_morphometry(self, morphs: list[MorphSync]):
        """
        Add morphometry features to the MorphSync object.
        """
        morphs = listify(morphs)
        root_ids = [morph.name for morph in morphs]
        features_by_root = self.get_morphometry(root_ids)

        for morph in morphs:
            root_id = morph.name
            if root_id not in features_by_root:
                continue
            features = features_by_root[root_id]

            morph.add_points(
                features, name="spine_morphometry", spatial_columns=["x", "y", "z"]
            )
