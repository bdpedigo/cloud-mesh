# %%
import json
import logging
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional, Union

import attrs
import numpy as np
import pandas as pd
import requests
import toml
import urllib3
from cave_mapper import map_points
from caveclient import CAVEclient, set_session_defaults
from cloudfiles import CloudFiles
from cloudvolume import CloudVolume
from joblib import load

from cloudigo import exists, get_dataframe, put_dataframe
from meshmash import (
    component_morphometry_pipeline,
    condensed_hks_pipeline,
    find_nucleus_point,
    get_synapse_mapping,
    interpret_path,
    read_array,
    read_condensed_features,
    read_condensed_graph,
    read_id_to_mesh_map,
    save_array,
    save_condensed_features,
    save_condensed_graph,
    save_id_to_mesh_map,
    scale_mesh,
)

SYSTEM = platform.system()

urllib3.disable_warnings()

# suppress warnings for WARNING:urllib3.connectionpool:Connection pool is full...

logging.getLogger("urllib3").setLevel(logging.CRITICAL)

DATASTACK = os.environ.get("DATASTACK", "minnie65_phase3_v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "absolute-solo-yak")
VERBOSE = str(os.environ.get("VERBOSE", "True")).lower() == "true"
N_JOBS = int(os.environ.get("N_JOBS", -2))
REPLICAS = int(os.environ.get("REPLICAS", 1))
MATERIALIZATION_VERSION = int(os.environ.get("MATERIALIZATION_VERSION", 1300))
QUEUE_NAME = os.environ.get("QUEUE_NAME", "ben-skedit")
RUN = os.environ.get("RUN", False)
REQUEST = os.environ.get("REQUEST", not RUN)
LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL", "ERROR")
LEASE_SECONDS = int(os.environ.get("LEASE_SECONDS", 7200))
RECOMPUTE = os.environ.get("RECOMPUTE", False)
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", 5))
BACKOFF_FACTOR = int(os.environ.get("BACKOFF_FACTOR", 4))
BACKOFF_MAX = int(os.environ.get("BACKOFF_MAX", 240))
MAX_RUNS = int(os.environ.get("MAX_RUNS", 5))

logging.basicConfig(level="ERROR")

logging.getLogger("meshmash").setLevel(level=LOGGING_LEVEL)

if SYSTEM == "Darwin":
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

# %%


# decorator for protecting any method in a try/except with logging logic
def loggable(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = f"Error in {func.__name__} with experiment {args[0]}"
            msg += "\n\n" + str(e)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            exp = traceback.format_exception(
                exc_type, exc_value, exc_traceback, limit=50
            )
            tb = "".join(exp)
            msg += "\n" + str(tb)
            logging.error(msg)
            requests.post(URL, json={"content": msg})
            raise e

    return wrapper


@attrs.define
class Experiment:
    root_id: int = attrs.field()
    datastack: str = attrs.field()
    version: int = attrs.field()
    scale: float = attrs.field(default=1.0)
    model: Optional[Any] = attrs.field(
        default=None, repr=False, init=True
    )  # Placeholder for the model, can be set later
    select_label: Optional[Union[str, int]] = attrs.field(
        default=None,
        repr=False,
        init=True,
    )

    _path: Path = attrs.field(init=False, repr=False)
    _cf: CloudFiles = attrs.field(init=False, repr=False)
    _client: CAVEclient = attrs.field(init=False, default=None, repr=False)
    _mesh: tuple = attrs.field(init=False, default=None, repr=False)
    _pre_synapse_mapping: np.ndarray = attrs.field(init=False, default=None, repr=False)
    _post_synapse_mappings: pd.Series = attrs.field(init=False, default=None, repr=False)
    _nuc_point: np.ndarray = attrs.field(init=False, default=None, repr=False)
    _checked_nuc_point: bool = attrs.field(init=False, default=False, repr=False)
    _labels: np.ndarray = attrs.field(init=False, default=None, repr=False)
    _condensed_features: np.ndarray = attrs.field(init=False, default=None, repr=False)
    _condensed_nodes: np.ndarray = attrs.field(init=False, default=None, repr=False)
    _condensed_edges: np.ndarray = attrs.field(init=False, default=None, repr=False)
    _condensed_posteriors: pd.DataFrame = attrs.field(
        init=False, default=None, repr=False
    )
    _condensed_predictions: pd.Series = attrs.field(
        init=False, default=None, repr=False
    )
    _condensed_posterior_entropy: pd.Series = attrs.field(
        init=False, default=None, repr=False
    )

    # _mesh_predictions: Optional[np.ndarray] = attrs.field(init=False, default=None, repr=False)
    _synapse_prediction_summary: pd.DataFrame = attrs.field(
        init=False, default=None, repr=False
    )
    _morphometry_summary: pd.DataFrame = attrs.field(
        init=False, default=None, repr=False
    )
    _components: np.ndarray = attrs.field(init=False, default=None, repr=False)

    def __attrs_post_init__(self):
        path = Path(f"gs://bdp-ssa//{self.datastack}/{MODEL_NAME}")
        self._path = path
        cf, _ = interpret_path(path)
        self._cf = cf

    @property
    def client(self):
        if (
            self._client is not None
            and self._client.datastack_name == self.datastack
            and self._client.version == self.version
        ):
            return self._client
        else:
            set_session_defaults(
                max_retries=MAX_RETRIES,
                backoff_factor=BACKOFF_FACTOR,
                backoff_max=BACKOFF_MAX,
            )
            self._client = CAVEclient(self.datastack, version=self.version)
            return self._client

    @property
    @loggable
    def mesh(self):
        if self._mesh is None:
            root_id = self.root_id
            datastack = self.datastack
            scale = self.scale
            client = self.client

            logging.info(f"Loading mesh for {root_id}")
            if datastack == "zheng_ca3":
                cv = CloudVolume(
                    "gs://zheng_mouse_hippocampus_production/v2/seg_m195",
                    progress=False,
                )
                raw_mesh = cv.mesh.get(root_id)[root_id]
                raw_mesh = raw_mesh.deduplicate_vertices(is_chunk_aligned=True)
                raw_mesh = (raw_mesh.vertices, raw_mesh.faces)
            else:
                cv = client.info.segmentation_cloudvolume(progress=False)
                raw_mesh = cv.mesh.get(root_id, **parameters["cv-mesh-get"])[root_id]
                raw_mesh = (raw_mesh.vertices, raw_mesh.faces)
            logging.info(
                f"Loaded mesh for {root_id}, has {raw_mesh[0].shape[0]} vertices"
            )

            if scale != 1.0:
                raw_mesh = scale_mesh(raw_mesh, scale)

            self._mesh = raw_mesh
        return self._mesh

    @property
    @loggable
    def pre_synapse_mapping(self):
        if self._pre_synapse_mapping is None:
            root_id = self.root_id
            client = self.client

            pre_synapse_mapping_file = (
                self._path / "pre-synapse-mappings" / f"{root_id}.npz"
            )

            if not exists(pre_synapse_mapping_file) or RECOMPUTE:
                logging.info(
                    f"Searched for pre-synapse mapping at {pre_synapse_mapping_file}"
                )
                logging.info(f"Pre-synapse mapping not found for {root_id}")
                pre_synapse_mapping = get_synapse_mapping(
                    root_id,
                    self.mesh,
                    client,
                    side="pre",
                    **parameters["project_points_to_mesh"],
                )
                save_id_to_mesh_map(pre_synapse_mapping_file, pre_synapse_mapping)
                logging.info(
                    f"Saved {len(pre_synapse_mapping)} synapse mappings for {root_id}"
                )
                return pre_synapse_mapping
            else:
                self._pre_synapse_mapping = read_id_to_mesh_map(
                    pre_synapse_mapping_file
                )
        return self._pre_synapse_mapping

    @property
    @loggable
    def post_synapse_mappings(self):
        if self._post_synapse_mappings is None:
            root_id = self.root_id
            client = self.client

            post_synapse_mappings_file = (
                self._path / "post-synapse-mappings" / f"{root_id}.npz"
            )

            if not exists(post_synapse_mappings_file) or RECOMPUTE:
                logging.info(
                    f"Searched for post-synapse mapping at {post_synapse_mappings_file}"
                )
                logging.info(f"Post-synapse mapping not found for {root_id}")
                post_synapse_mappings = get_synapse_mapping(
                    root_id,
                    self.mesh,
                    client,
                    side="post",
                    **parameters["project_points_to_mesh"],
                )
                save_id_to_mesh_map(post_synapse_mappings_file, post_synapse_mappings)
                logging.info(
                    f"Saved {len(post_synapse_mappings)} synapse mappings for {root_id}"
                )
            else:
                post_synapse_mappings = read_id_to_mesh_map(post_synapse_mappings_file)
            post_synapse_mappings = pd.Series(
                index=post_synapse_mappings[:, 0], data=post_synapse_mappings[:, 1]
            )
            self._post_synapse_mappings = post_synapse_mappings
        return self._post_synapse_mappings

    @property
    @loggable
    def nuc_point(self):
        if self._nuc_point is None and not self._checked_nuc_point:
            root_id = self.root_id
            client = self.client

            try:
                nuc_point = find_nucleus_point(root_id, client, update_root_id="check")
                self._checked_nuc_point = True
                if nuc_point is None:
                    logging.info(f"No nucleus point found for {root_id}")
                else:
                    logging.info(f"Found nucleus point for {root_id} at {nuc_point[0]}")
                    self._nuc_point = nuc_point
            except Exception as e:
                logging.info(f"Error finding nucleus point for {root_id}: {e}")
                self._nuc_point = None
                self._checked_nuc_point = True
        return self._nuc_point

    @loggable
    def _hks_pipeline(self):
        mesh = self.mesh
        nuc_point = self.nuc_point
        verbose = VERBOSE
        n_jobs = N_JOBS
        path = self._path
        scale = self.scale
        root_id = self.root_id

        result = condensed_hks_pipeline(
            mesh,
            nuc_point=nuc_point,
            verbose=verbose,
            n_jobs=n_jobs,
            **parameters["condensed_hks_pipeline"],
        )
        labels = result.labels
        condensed_features = result.condensed_features

        feature_out_path = path / "features"
        if scale != 1.0:
            feature_out_path = feature_out_path / str(scale)
            out_file = feature_out_path / f"{root_id}.npz"
        else:
            out_file = feature_out_path / f"{root_id}.npz"
        save_condensed_features(
            out_file,
            condensed_features,
            labels,
            **parameters["save_condensed_features"],
        )
        logging.info(f"Saved features for {root_id}")
        self._condensed_features = condensed_features
        self._labels = labels

        if scale == 1.0:
            graph_out_path = path / "graphs"
            graph_file = graph_out_path / f"{root_id}.npz"
            save_condensed_graph(
                graph_file,
                result.condensed_nodes,
                result.condensed_edges,
                **parameters["save_condensed_graph"],
            )
            logging.info(f"Saved edges for {root_id}")

            self._condensed_nodes = result.condensed_nodes
            self._condensed_edges = result.condensed_edges

    @property
    def labels(self):
        if self._labels is None:
            feature_file = self._path / "features" / f"{self.root_id}.npz"
            if not exists(feature_file) or RECOMPUTE:
                self._hks_pipeline()
            else:
                features, labels = read_condensed_features(feature_file)
                self._condensed_features = features
                self._labels = labels
        return self._labels

    @property
    def condensed_features(self):
        if self._condensed_features is None:
            feature_file = self._path / "features" / f"{self.root_id}.npz"
            if not exists(feature_file) or RECOMPUTE:
                self._hks_pipeline()
            else:
                features, _ = read_condensed_features(feature_file)
                self._condensed_features = features
        return self._condensed_features

    @property
    def condensed_nodes(self):
        if self._condensed_nodes is None:
            graph_file = self._path / "graphs" / f"{self.root_id}.npz"
            if not exists(graph_file) or RECOMPUTE:
                self._hks_pipeline()
            else:
                _, nodes, edges = read_condensed_graph(graph_file)
                self._condensed_nodes = nodes
                self._condensed_edges = edges
        return self._condensed_nodes

    @property
    def condensed_edges(self):
        if self._condensed_edges is None:
            graph_file = self._path / "graphs" / f"{self.root_id}.npz"
            if not exists(graph_file) or RECOMPUTE:
                self._hks_pipeline()
            else:
                _, nodes, edges = read_condensed_graph(graph_file)
                self._condensed_nodes = nodes
                self._condensed_edges = edges
        return self._condensed_edges

    @loggable
    def _condensed_prediction(self):
        condensed_features = self.condensed_features
        model = self.model

        condensed_features = condensed_features[model.feature_names_in_]

        condensed_posteriors = model.predict_proba(condensed_features)
        condensed_posteriors = pd.DataFrame(
            condensed_posteriors, index=condensed_features.index, columns=model.classes_
        )
        condensed_posteriors.loc[-1] = None
        self._condensed_posteriors = condensed_posteriors

        condensed_predictions = condensed_posteriors.idxmax(axis=1)
        condensed_predictions.loc[-1] = None
        self._condensed_predictions = condensed_predictions

        condensed_posterior_entropy = -np.sum(
            condensed_posteriors * np.log(condensed_posteriors), axis=1
        )
        self._condensed_posterior_entropy = condensed_posterior_entropy

    @property
    def mesh_predictions(self):
        out = self.condensed_predictions.loc[self.labels]
        return out.values

    @property
    def condensed_posteriors(self):
        if self._condensed_posteriors is None:
            self._condensed_prediction()
        return self._condensed_posteriors

    @property
    def condensed_predictions(self):
        if self._condensed_predictions is None:
            self._condensed_prediction()
        return self._condensed_predictions

    @property
    def condensed_posterior_entropy(self):
        if self._condensed_posterior_entropy is None:
            self._condensed_prediction()
        return self._condensed_posterior_entropy

    @loggable
    def _post_target_prediction(self):
        root_id = self.root_id
        condensed_ids = self.labels
        datastack = self.datastack

        condensed_predictions = self.condensed_predictions
        condensed_posteriors = self.condensed_posteriors
        condensed_posterior_entropy = self.condensed_posterior_entropy
        synapse_mapping = self.post_synapse_mappings

        synapse_predictions = condensed_predictions.loc[
            condensed_ids[synapse_mapping]
        ].values
        synapse_predictions = pd.Series(
            synapse_predictions, index=synapse_mapping.index, name="pred_label"
        )

        synapse_posterior_max = (
            condensed_posteriors.loc[condensed_ids[synapse_mapping]].max(axis=1).values
        )
        synapse_posterior_max = pd.Series(
            synapse_posterior_max, index=synapse_mapping.index, name="posterior_max"
        )

        synapse_posterior_entropy = condensed_posterior_entropy.loc[
            condensed_ids[synapse_mapping]
        ].values
        synapse_posterior_entropy = pd.Series(
            synapse_posterior_entropy,
            index=synapse_mapping.index,
            name="posterior_entropy",
        )
        synapse_prediction_summary = pd.concat(
            [
                synapse_predictions,
                synapse_posterior_max,
                synapse_posterior_entropy,
            ],
            axis=1,
        )
        synapse_prediction_summary.rename_axis("id", inplace=True)
        synapse_prediction_summary.dropna(how="any", inplace=True)

        synapse_prediction_path = f"gs://bdp-ssa/{datastack}/{PARAMETER_NAME}/post-synapse-predictions/{root_id}.csv.gz"
        put_dataframe(synapse_prediction_summary, synapse_prediction_path)

        self._synapse_prediction_summary = synapse_prediction_summary

    @property
    def synapse_prediction_summary(self):
        if self._synapse_prediction_summary is None:
            synapse_prediction_path = (
                self._path / "post-synapse-predictions" / f"{self.root_id}.csv.gz"
            )
            if not exists(synapse_prediction_path) or RECOMPUTE:
                self._post_target_prediction()
            else:
                self._synapse_prediction_summary = get_dataframe(
                    synapse_prediction_path, index_col=0
                )
        return self._synapse_prediction_summary

    @loggable
    def _morphometry_pipeline(self):
        morphometry_summary, components = component_morphometry_pipeline(
            self.mesh,
            self.mesh_predictions,
            select_label=self.select_label,
            post_synapse_mappings=self.post_synapse_mappings,
            verbose=True,

        )
        morphometry_summary.drop(columns=["n_interior_samples"])

        points = morphometry_summary[["x", "y", "z"]].values
        mask = np.isfinite(points).all(axis=1)
        points = points[mask]
        mapping_info = map_points(
            points,
            self.root_id,
            self.client,
            initial_distance=0,
            max_distance=4,
            verbose=VERBOSE,
        )
        mapping_info.index = morphometry_summary.index[mask]
        morphometry_summary = morphometry_summary.join(
            mapping_info[["voxel_pt_x", "voxel_pt_y", "voxel_pt_z"]],
            how="left",
        )

        put_dataframe(
            morphometry_summary,
            self._path / f"{self.select_label}-morphometry" / f"{self.root_id}.csv.gz",
        )

        component_mapping_path = (
            self._path
            / f"{self.select_label}-component-mappings"
            / f"{self.root_id}.npz"
        )
        save_array(
            component_mapping_path,
            components,
        )
        self._morphometry_summary = morphometry_summary
        self._components = components

    @property
    def morphometry_summary(self):
        morphometry_summary_path = (
            self._path / f"{self.select_label}-morphometry" / f"{self.root_id}.csv.gz"
        )
        if self._morphometry_summary is None:
            if not exists(morphometry_summary_path) or True:
                self._morphometry_pipeline()
            else:
                self._morphometry_summary = get_dataframe(
                    morphometry_summary_path, index_col=0
                )
        return self._morphometry_summary

    @property
    def components(self):
        component_path = (
            self._path
            / f"{self.select_label}-component-mappings"
            / f"{self.root_id}.npz"
        )
        if self._components is None:
            if not exists(component_path) or RECOMPUTE:
                self._morphometry_pipeline()
            else:
                self._components = read_array(component_path)
        return self._components

    @property
    def post_synapse_components(self):
        out = self.components[self.post_synapse_mappings]
        return out


PARAMETER_NAME = "absolute-solo-yak"
model_folder = Path(__file__).parent.parent / "models" / PARAMETER_NAME
model_path = model_folder / "hks_model_calibrated.joblib"
model = load(model_path)

RECOMPUTE = False
root_id = 648518346428852915
datastack = "zheng_ca3"
version = 195
e = Experiment(root_id, datastack, version, model=model, select_label="spine")

# e.labels
# e.condensed_features
# e._post_target_prediction()
# e.synapse_prediction_summary

# e.mesh_predictions
# RECOMPUTE = True

currtime = time.time()
morphometry_summary = e.morphometry_summary
print(f"{time.time() - currtime:.3f} seconds elapsed.")

# %%
# mesh = e.mesh
morphometry_summary


# %%
morphometry_summary["n_synapses"] = 0
for component in e.post_synapse_components:
    if component != -1:
        morphometry_summary.loc[component, "n_synapses"] += 1

# %%

# col = "x"
# col = 'area_nm2'
# x = morphometry_summary.reindex(e.components)[col].isna().values & (
#     e.components > 0
# )
# # x[x < 0] = morphometry_summary['fiedler_eval'].max()
# x = np.log(x)

# clim = [morphometry_summary[col].min(), morphometry_summary[col].max()]

import pyvista as pv

plotter = pv.Plotter()

plotter.add_mesh(pv.make_tri_mesh(*e.mesh), color="lightgray", opacity=0.5)
plotter.add_points(
    morphometry_summary[["x", "y", "z"]].values,
    scalars=morphometry_summary["n_post_synapses"].values,
    point_size=10,
    cmap="Reds",
    render_points_as_spheres=True,
    # scalar_bar_args={"title": "Size (nm³)"},
)

plotter.enable_fly_to_right_click()
plotter.show()

# %%

plotter = pv.Plotter()
morphometry_summary["graph_fiedler_eval_rank"] = morphometry_summary[
    "graph_fiedler_eval"
].rank(ascending=True, method="first")
plotter.add_mesh(
    pv.make_tri_mesh(*e.mesh),
    scalars=morphometry_summary["graph_fiedler_eval_rank"]
    .reindex(e.components)
    .fillna(morphometry_summary["graph_fiedler_eval_rank"].max())
    .values,
    cmap="Reds_r",
)
plotter.enable_fly_to_right_click()
plotter.show()

# %%
import seaborn as sns

sns.histplot(data=morphometry_summary, x="size_nm3", log_scale=True)

# %%
