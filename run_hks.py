"""
run_hks.py - Task definition for condensed HKS feature extraction.

Defines run_hks(), a queueable task that:
  1. Loads a neuron mesh from a CAVE-linked CloudVolume
  2. Runs the condensed HKS pipeline on the full mesh, OR - when a
     query_point_nm + query_radius_nm are provided - runs the chunked
     HKS pipeline restricted to a neighborhood around that point.
  3. Saves the output to GCS via export_hks_output
"""

import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Optional, Sequence
from cloudpathlib import GSPath

import tomllib
from caveclient import CAVEclient, set_session_defaults
from meshmash import chunked_hks_pipeline, condensed_hks_pipeline
from fast_simplification import replay_simplification, simplify
from scipy.spatial import cKDTree
from taskqueue import queueable

from io import BytesIO
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import requests
from time import time

import warnings
warnings.filterwarnings("ignore", message=".*backoff_max.*")  # silence urllib3 v2 warning
warnings.filterwarnings("ignore", message=".*use_all_points.*")

# Load config and parameters
_config_path = Path(os.environ.get("CONFIG_PATH", Path(__file__).parent / "config.toml"))
_hks_path = Path(os.environ.get("HKS_PARAMS_PATH", Path(__file__).parent / "hks_parameters.toml"))

with open(_config_path, "rb") as _f:
    _config = tomllib.load(_f)["job"]

with open(_hks_path, "rb") as _f:
    _hks = tomllib.load(_f)

_filter = _hks.get("mesh_filter", {})
VERT_MIN = int(_filter.get("min_total_vertices", 300))
VERT_MAX = int(_filter.get("max_total_vertices", 5_000_000))


def _cfg(key, env_var, cast=str):
    """Return env var if set, otherwise config.toml value."""
    raw = os.environ.get(env_var)
    if raw is not None:
        return cast(raw)
    return _config[key]


OUTPUT_BUCKET = _cfg("output_bucket", "CLOUD_MESH_OUTPUT_BUCKET")
N_JOBS = _cfg("n_jobs", "CLOUD_MESH_N_JOBS", int)
RECOMPUTE = (
    _cfg("recompute", "CLOUD_MESH_RECOMPUTE", lambda v: v.lower() == "true")
    if os.environ.get("CLOUD_MESH_RECOMPUTE")
    else bool(_config["recompute"])
)
LOGGING_LEVEL = _cfg("logging_level", "CLOUD_MESH_LOGGING_LEVEL")
WEBHOOK = os.environ.get("SLACK_WEBHOOK_URL")
CAVE_AUTH_TOKEN = os.environ.get("CAVE_AUTH_TOKEN")

# Logging
logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("run_hks")


# Helpers
def _output_path(
    output_bucket: str,
    datastack: str,
    root_id: int,
    synapse_id: Optional[int] = None,
) -> str:
    """Build the GCS (or local) output path for an HKS result.

    Full-mesh runs go to ``features/{root_id}.npz``.
    Chunked (per-synapse) runs go to
    ``features/synapse/{root_id}_syn_{synapse_id}.npz`` so synapse-targeted
    outputs are kept in their own subfolder, separate from the full-mesh
    outputs but still trivially identifiable by their synapse of origin.
    """
    base = f"{output_bucket.rstrip('/')}/{datastack}/features"
    if synapse_id is None:
        return f"{base}/{root_id}.npz"
    return f"{base}/synapse/{root_id}_syn_{int(synapse_id)}.npz"


def _resolve_query_indices(
    mesh_vertices: np.ndarray, point_nm: Sequence[float], radius_nm: float
) -> np.ndarray:
    """Return indices of mesh vertices within ``radius_nm`` of ``point_nm``.

    Uses Euclidean distance in the mesh vertex coordinate space (nm for CAVE
    meshes). Returns at least the single nearest vertex even if the ball is
    empty, so that chunked_hks_pipeline always has something to work with.
    """
    tree = cKDTree(mesh_vertices)
    idxs = np.asarray(
        tree.query_ball_point(np.asarray(point_nm, dtype=float), r=float(radius_nm)),
        dtype=np.int64,
    )
    if idxs.size == 0:
        # fallback: nearest single vertex, so the caller gets a non-empty query
        _, nearest = tree.query(np.asarray(point_nm, dtype=float), k=1)
        idxs = np.asarray([int(nearest)], dtype=np.int64)
    return idxs


def _prune_mesh_to_ball(
    mesh: tuple,
    point_nm: Sequence[float],
    radius_nm: float,
) -> tuple[tuple, int]:
    """Prune a mesh to the connected-face neighborhood within ``radius_nm`` of
    ``point_nm``.

    Returns ``(pruned_mesh, n_kept_vertices)`` where ``pruned_mesh`` is a
    ``(vertices, faces)`` tuple with re-indexed faces. We keep any face whose
    *centroid* is within ``radius_nm`` of ``point_nm``, then keep all vertices
    referenced by those faces. This avoids the holes you'd get from a strict
    vertex-only mask: a face survives even if only its centroid falls inside
    the ball, so the resulting submesh stays watertight along the boundary
    and the chunked HKS pipeline's overlap region has real geometry to work
    with.

    If the radius doesn't pick up any faces (rare — synapse outside the cell
    mesh, or radius too small), falls back to keeping a small neighborhood
    around the single nearest vertex so the caller always gets a non-empty
    submesh.

    Parameters
    ----------
    mesh : (vertices (V, 3) float32, faces (F, 3) int)
    point_nm : (3,) sequence
        Query point in nm (same coordinate space as mesh vertices).
    radius_nm : float
        Ball radius in nm.

    Returns
    -------
    pruned_mesh : (verts_p (Vp, 3), faces_p (Fp, 3))
    n_kept_vertices : int
        Equals ``Vp``; returned for logging convenience.
    """
    verts, faces = mesh
    verts = np.asarray(verts)
    faces = np.asarray(faces, dtype=np.int64)
    point = np.asarray(point_nm, dtype=float)
    r = float(radius_nm)

    # Use face centroids to decide membership: more robust than per-vertex
    # masks at the boundary, and cheap (one mean per face).
    centroids = verts[faces].mean(axis=1)
    d2 = np.sum((centroids - point) ** 2, axis=1)
    keep_face = d2 <= (r * r)

    if not keep_face.any():
        # Fallback: keep all faces touching the nearest few vertices to the
        # query point so we never hand chunked_hks_pipeline an empty mesh.
        tree = cKDTree(verts)
        _, nearest = tree.query(point, k=min(64, verts.shape[0]))
        nearest = np.atleast_1d(nearest).astype(np.int64)
        keep_vert_mask = np.zeros(verts.shape[0], dtype=bool)
        keep_vert_mask[nearest] = True
        # Keep faces with any of those vertices.
        keep_face = keep_vert_mask[faces].any(axis=1)
        if not keep_face.any():
            # Truly degenerate: hand back the original mesh untouched.
            return (verts, faces), verts.shape[0]

    kept_faces = faces[keep_face]
    used_vert_idx = np.unique(kept_faces.reshape(-1))

    # Build old->new index remap for faces.
    remap = np.full(verts.shape[0], -1, dtype=np.int64)
    remap[used_vert_idx] = np.arange(used_vert_idx.size, dtype=np.int64)
    new_faces = remap[kept_faces]
    new_verts = verts[used_vert_idx]

    # Preserve dtype on the vertex array (the rest of the pipeline expects
    # float32 throughout).
    if verts.dtype != new_verts.dtype:
        new_verts = new_verts.astype(verts.dtype)

    return (new_verts, new_faces), int(new_verts.shape[0])


def post_slack_message(text: str, channel: Optional[str] = None) -> bool:
    """Post to Slack. With ``channel`` unset the webhook uses its own channel."""
    if not WEBHOOK:
        print("SLACK_WEBHOOK_URL not set; skipping Slack post.", file=sys.stderr)
        return False

    payload = {"text": text}
    if channel:
        payload["channel"] = channel

    try:
        resp = requests.post(WEBHOOK, json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print("Slack post failed:", repr(e), file=sys.stderr)
        return False


def export_hks_output(output_path, root_id, hks_output=None):
    """
    Repackage the hks results for saving, or load saved outputs and reassemble the dataframes.
    Supports local paths and GCS URIs (gs://...) via google-cloud-storage.
    Returns a dict with keys matching the previous behavior or None if no data.
    """
    output_path = str(output_path)
    parsed = urlparse(output_path)
    use_gcs = parsed.scheme == "gs"

    if parsed.path and parsed.path not in ("/", ""):
        path_suffix = parsed.path.lstrip("/")
    else:
        path_suffix = ""

    file_name = (
        Path(path_suffix).name
        if path_suffix and not output_path.endswith("/")
        else f"{root_id}_hks_output.npz"
    )

    if use_gcs:
        from google.cloud import storage

        bucket_name = parsed.netloc
        prefix = parsed.path.lstrip("/")
        if output_path.endswith("/") or prefix == "" or prefix.endswith("/"):
            blob_name = (prefix.rstrip("/") + "/" + file_name).lstrip("/")
        else:
            blob_name = prefix

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        file_exists = blob.exists()
    else:
        p = Path(output_path)
        if p.is_dir() or output_path.endswith("/"):
            dest = p / file_name
        else:
            dest = p if p.suffix else (p / file_name)
        file_exists = dest.exists()

    if not file_exists or (file_exists and hks_output is not None):
        if hks_output is None:
            raise ValueError("No hks_output provided to save.")
        hks_export = {
            "mapping": hks_output.mapping.copy(),
            "labels": hks_output.labels.copy(),
            "simple_labels": hks_output.simple_labels.copy(),
            "condensed_features_values": hks_output.condensed_features.values.copy(),
            "condensed_features_columns": np.array(
                hks_output.condensed_features.columns.astype(str)
            ),
            "simple_mesh_vertices": hks_output.simple_mesh[0],
            "simple_mesh_faces": hks_output.simple_mesh[1],
        }

        with BytesIO() as bio:
            np.savez_compressed(bio, **hks_export)
            data_bytes = bio.getvalue()

            if use_gcs:
                blob.upload_from_string(data_bytes, content_type="application/octet-stream")
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as f:
                    f.write(data_bytes)
        return hks_export

    if file_exists:
        if use_gcs:
            raw = blob.download_as_bytes()
        else:
            with open(dest, "rb") as f:
                raw = f.read()

        with BytesIO(raw) as bio:
            data = np.load(bio, allow_pickle=True)
            hks_export = {
                "mapping": data["mapping"],
                "labels": data["labels"],
                "simple_labels": data["simple_labels"],
                "condensed_features": pd.DataFrame(
                    data["condensed_features_values"],
                    columns=data["condensed_features_columns"],
                ),
                "simple_mesh_vertices": data["simple_mesh_vertices"],
                "simple_mesh_faces": data["simple_mesh_faces"],
            }
        return hks_export

    return None


# Task
@queueable
def run_hks(
    root_id: int,
    datastack: str,
    synapse_id: Optional[int] = None,
    query_point_nm: Optional[Sequence[float]] = None,
    query_radius_nm: Optional[float] = None,
) -> None:
    """Extract HKS features for one neuron and write to GCS (or local path).

    When synapse_id, query_point_nm, and query_radius_nm are all provided,
    the analysis is restricted to a neighborhood of mesh vertices within
    query_radius_nm of query_point_nm (Euclidean, in nm) - i.e. the local
    region around one synapse. The chunked HKS pipeline is used in that
    case, and the output goes to features/{root_id}_syn_{synapse_id}.npz.
    Otherwise the full-mesh condensed pipeline is run and the output goes
    to features/{root_id}.npz as before.

    All three targeted-region args must be supplied together or not at all.
    """
    any_targeted = (
        synapse_id is not None
        or query_point_nm is not None
        or query_radius_nm is not None
    )
    all_targeted = (
        synapse_id is not None
        and query_point_nm is not None
        and query_radius_nm is not None
    )
    if any_targeted and not all_targeted:
        raise ValueError(
            "synapse_id, query_point_nm, and query_radius_nm must be supplied "
            f"together (got synapse_id={synapse_id!r}, "
            f"point={query_point_nm!r}, radius={query_radius_nm!r})"
        )
    chunked = all_targeted

    if chunked:
        synapse_id = int(synapse_id)
        log.info(
            "root_id=%s datastack=%s - starting CHUNKED run (synapse_id=%s, point=%s, r=%s nm)",
            root_id, datastack, synapse_id, list(query_point_nm), query_radius_nm,
        )
    else:
        log.info("root_id=%s datastack=%s - starting", root_id, datastack)

    out_path = _output_path(
        OUTPUT_BUCKET, datastack, root_id, synapse_id=synapse_id if chunked else None
    )
    is_gs = str(out_path).startswith("gs://")

    if is_gs:
        gp = GSPath(out_path)
        exists = gp.exists()
    else:
        p = Path(out_path)
        if p.is_dir() or str(out_path).endswith("/"):
            suffix = (
                f"{root_id}_syn_{synapse_id}.npz" if chunked else f"{root_id}.npz"
            )
            p = Path(out_path) / suffix
        exists = p.exists()

    if not RECOMPUTE and exists:
        log.info("root_id=%s datastack=%s - already done, skipping", root_id, datastack)
        return

    try:
        set_session_defaults(max_retries=5, backoff_factor=4, backoff_max=240)
        client = CAVEclient(datastack, auth_token=CAVE_AUTH_TOKEN)

        log.info("root_id=%s datastack=%s - loading mesh", root_id, datastack)
        logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR) # silence "Connection pool is full" warning
        cv = client.info.segmentation_cloudvolume(progress=False)
        mesh = cv.mesh.get(root_id, **_hks.get("cv_mesh_get", {}))[root_id]
        n_verts = mesh.vertices.shape[0]
        mesh = (mesh.vertices.astype(np.float32), mesh.faces) # convert to float32

        log.info(
            "root_id=%s datastack=%s - mesh loaded: %d vertices, %d faces",
            root_id, datastack, n_verts, mesh[1].shape[0],
        )

        if chunked:
            # NOTE: meshmash.chunked_hks_pipeline simplifies the mesh internally
            # (when simplify_target_reduction is not None) and then constructs a
            # MeshStitcher whose submesh_mapping is sized to the *simplified*
            # vertex count. However, it never remaps the user-supplied
            # ``query_indices`` from original vertex space into simplified space,
            # so for any vertex whose simplified-space index would exceed the
            # simplified vertex count, the pipeline raises IndexError inside
            # MeshStitcher.subset_apply (out of bounds on submesh_mapping). This
            # accounts for the bulk of dead-lettered chunked tasks.
            #
            # Workaround: do the simplification ourselves, resolve query indices
            # against the *simplified* mesh, and call the pipeline with
            # simplify_target_reduction=None so it doesn't re-simplify. Resolving
            # by nearest-vertex-to-point in simplified space (rather than
            # remapping original indices through thresh_to_simple_mapping) is
            # robust to the many-to-one collapses that simplification produces.
            chunked_kwargs = dict(_hks.get("chunked_hks_pipeline", {}))
            simplify_target_reduction = chunked_kwargs.pop(
                "simplify_target_reduction", 0.7
            )
            simplify_agg = chunked_kwargs.pop("simplify_agg", 7)

            # Diagnostic: confirm the query point is actually inside (or near)
            # the loaded mesh's coordinate frame. If they're in different
            # spaces — e.g. mesh-in-nm vs query-in-voxel, or wrong root_id —
            # the prune will return ~0 vertices and the run will fail
            # confusingly. Print the mesh bbox, the distance from the query
            # point to the nearest mesh vertex, and the vertex count at a few
            # ring radii so you can spot a unit/coord mismatch in one log line.
            _verts_all = mesh[0]
            _qp = np.asarray(query_point_nm, dtype=float)
            _bbox_lo = _verts_all.min(axis=0)
            _bbox_hi = _verts_all.max(axis=0)
            _tree_diag = cKDTree(_verts_all)
            _nn_dist, _ = _tree_diag.query(_qp, k=1)
            _ring_radii = (10000.0, 50000.0, 100000.0, 500000.0, 2_000_000.0)
            _ring_counts = []
            for _r in _ring_radii:
                _idxs = _tree_diag.query_ball_point(_qp, r=_r)
                _ring_counts.append(len(_idxs))
            log.info(
                "root_id=%s syn=%s - DIAG mesh bbox nm: x=[%.0f, %.0f] "
                "y=[%.0f, %.0f] z=[%.0f, %.0f]; query_point_nm=%s; "
                "nearest mesh vertex is %.0f nm away; vertex counts within "
                "%s nm: %s",
                root_id, synapse_id,
                _bbox_lo[0], _bbox_hi[0], _bbox_lo[1], _bbox_hi[1],
                _bbox_lo[2], _bbox_hi[2], _qp.tolist(), float(_nn_dist),
                list(_ring_radii), _ring_counts,
            )

            # Hard sanity guard. If the query point is well outside the
            # mesh bbox plus a generous margin, bail with a unit-mismatch-
            # specific error rather than wasting cycles on a doomed prune.
            # Gotcha worth knowing: CAVE stores ctr_pt_position at the
            # segmentation's MIP-0 voxel size, which is what
            # client.info.viewer_resolution() returns. Where a datastack's
            # image MIP-0 is finer than its segmentation MIP-0, using the
            # image resolution instead puts the query point short by that
            # ratio in x/y.
            _margin = float(query_radius_nm) + 2.0 * float(
                chunked_kwargs.get("overlap_distance", 20000.0)
            )
            _outside = (
                (_qp < _bbox_lo - _margin) | (_qp > _bbox_hi + _margin)
            )
            if _outside.any():
                raise ValueError(
                    f"query_point_nm={_qp.tolist()} is outside the mesh "
                    f"bbox (lo={_bbox_lo.tolist()}, hi={_bbox_hi.tolist()}) "
                    f"by more than the chunked HKS margin ({_margin:.0f} nm). "
                    f"Likely cause: voxel->nm conversion used the wrong "
                    f"resolution. CAVE ctr_pt_position is in SEGMENTATION "
                    f"MIP-0 voxels — use client.info.viewer_resolution(), "
                    f"or pass --voxel-size explicitly. Otherwise verify the "
                    f"synapse is actually on this root_id."
                )

            # Prune the full mesh down to a ball around the query point *before*
            # simplifying. This delivers the actual point of the chunked design:
            # for huge presynaptic neurons the simplify+HKS cost scales with the
            # local neighborhood size, not the whole cell. We keep a generous
            # margin so the chunked pipeline's overlap region (controlled by
            # `overlap_distance` in chunked_kwargs) still has real geometry to
            # walk through; without margin you risk the pipeline's neighbor
            # search trying to reach beyond the kept region and returning
            # truncated/biased eigenfunctions near the boundary.
            overlap_distance = float(chunked_kwargs.get("overlap_distance", 20000.0))
            prune_radius_nm = float(query_radius_nm) + 2.0 * overlap_distance
            mesh, n_pruned = _prune_mesh_to_ball(
                mesh, query_point_nm, prune_radius_nm
            )
            log.info(
                "root_id=%s syn=%s - pruned mesh to %d vertices "
                "(ball radius=%.1f nm = query_radius %.1f + 2 * overlap %.1f) "
                "from full %d vertices",
                root_id, synapse_id, n_pruned, prune_radius_nm,
                float(query_radius_nm), overlap_distance, n_verts,
            )

            # Sanity guard: meshmash's split_mesh requires at least
            # ``min_vertex_threshold`` vertices to produce any valid submesh.
            # If the pruned region is below that threshold, the pipeline will
            # crash inside fit_mesh_split with an opaque
            # ``zero-size array to reduction operation maximum`` error.
            # Bail out early with a clear message instead.
            chunked_min_verts = int(chunked_kwargs.get("min_vertex_threshold", 200))
            chunked_max_verts = int(chunked_kwargs.get("max_vertex_threshold", 20000))
            if n_pruned < chunked_min_verts:
                log.warning(
                    "root_id=%s syn=%s - pruned mesh has only %d vertices, "
                    "below chunked min_vertex_threshold=%d. The synapse may be "
                    "on a thin or distal process; try a larger query_radius_nm "
                    "or verify the synapse position. Skipping.",
                    root_id, synapse_id, n_pruned, chunked_min_verts,
                )
                return

            # When the pruned mesh is small, simplification can drop us under
            # the chunked pipeline's min_vertex_threshold (e.g. 94 -> 28 with
            # target_reduction=0.7). Skip the explicit simplify pass in that
            # regime — the pruned mesh already fits inside a single chunk, so
            # there's nothing for the pipeline's stitcher to gain from a
            # smaller vertex count.
            should_simplify = (
                simplify_target_reduction is not None
                and n_pruned > chunked_max_verts
            )
            if simplify_target_reduction is not None and not should_simplify:
                log.info(
                    "root_id=%s syn=%s - skipping simplification: pruned mesh "
                    "(%d verts) already fits in one chunk (max=%d).",
                    root_id, synapse_id, n_pruned, chunked_max_verts,
                )

            if should_simplify:
                _, _, collapses = simplify(
                    mesh[0],
                    mesh[1],
                    agg=simplify_agg,
                    target_reduction=simplify_target_reduction,
                    return_collapses=True,
                )
                verts_s, faces_s, _ = replay_simplification(
                    points=mesh[0],
                    triangles=mesh[1],
                    collapses=collapses,
                )
                pipeline_mesh = (verts_s.astype(np.float32), faces_s)
                log.info(
                    "root_id=%s syn=%s - simplified pruned mesh from %d -> %d vertices "
                    "(target_reduction=%.2f, agg=%d)",
                    root_id, synapse_id, n_pruned, verts_s.shape[0],
                    simplify_target_reduction, simplify_agg,
                )
                if verts_s.shape[0] < chunked_min_verts:
                    log.warning(
                        "root_id=%s syn=%s - simplified mesh has %d vertices, "
                        "below chunked min_vertex_threshold=%d. Falling back to "
                        "the un-simplified pruned mesh (%d vertices).",
                        root_id, synapse_id, verts_s.shape[0],
                        chunked_min_verts, n_pruned,
                    )
                    pipeline_mesh = mesh
            else:
                pipeline_mesh = mesh

            query_indices = _resolve_query_indices(
                pipeline_mesh[0], query_point_nm, float(query_radius_nm)
            )
            log.info(
                "root_id=%s syn=%s - query resolved to %d vertices within %.1f nm of %s",
                root_id, synapse_id, query_indices.size,
                float(query_radius_nm), list(query_point_nm),
            )

            # Keep only query vertices that live in the LARGEST connected
            # component of the simplified mesh. meshmash's split_mesh drops
            # chunks below min_vertex_threshold, which can leave us with a
            # submesh_mapping smaller than our pipeline_mesh; if any of our
            # query_indices reference those dropped vertices, the pipeline
            # crashes with "IndexError: index N is out of bounds for axis 0
            # with size M" inside MeshStitcher.subset_apply. Restricting
            # query_indices to the dominant component sidesteps this.
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import connected_components as _cc

            _verts_p, _faces_p = pipeline_mesh
            _n_v = int(_verts_p.shape[0])
            _e0 = np.concatenate([_faces_p[:, 0], _faces_p[:, 1], _faces_p[:, 2]])
            _e1 = np.concatenate([_faces_p[:, 1], _faces_p[:, 2], _faces_p[:, 0]])
            _adj = csr_matrix(
                (np.ones(_e0.size, dtype=np.int8), (_e0, _e1)),
                shape=(_n_v, _n_v),
            )
            _n_comp, _comp_labels = _cc(_adj, directed=False)
            if _n_comp > 1:
                _sizes = np.bincount(_comp_labels)
                _largest = int(np.argmax(_sizes))
                _keep_mask = _comp_labels[query_indices] == _largest
                _kept = int(_keep_mask.sum())
                _dropped = int(query_indices.size - _kept)
                if _dropped > 0:
                    log.info(
                        "root_id=%s syn=%s - simplified mesh has %d connected "
                        "components (largest=%d verts); dropping %d/%d query "
                        "vertices that fall in smaller components.",
                        root_id, synapse_id, _n_comp, int(_sizes[_largest]),
                        _dropped, query_indices.size,
                    )
                    query_indices = query_indices[_keep_mask]
                if query_indices.size == 0:
                    # Fallback: take the nearest vertex in the largest
                    # component so chunked_hks_pipeline still has something
                    # to work with.
                    _in_largest = np.where(_comp_labels == _largest)[0]
                    _tree = cKDTree(_verts_p[_in_largest])
                    _, _nn = _tree.query(np.asarray(query_point_nm, dtype=float), k=1)
                    query_indices = np.asarray([int(_in_largest[int(_nn)])], dtype=np.int64)
                    log.info(
                        "root_id=%s syn=%s - all original query vertices were in "
                        "small components; falling back to nearest vertex in the "
                        "largest component (idx=%d).",
                        root_id, synapse_id, int(query_indices[0]),
                    )

            start_time = time()
            result = chunked_hks_pipeline(
                pipeline_mesh,
                query_indices=query_indices,
                # We've already simplified above (or chose not to); tell the
                # pipeline to skip its own simplification pass.
                simplify_target_reduction=None,
                verbose=False,
                n_jobs=N_JOBS,
                **chunked_kwargs,
            )
            elapsed = time() - start_time
            log.info(
                "root_id=%s syn=%s - chunked HKS finished in %.1f seconds",
                root_id, synapse_id, elapsed,
            )
        else:
            if not (VERT_MIN < n_verts < VERT_MAX):
                log.info(
                    "root_id=%s datastack=%s - %d vertices outside limits [%d, %d], skipping",
                    root_id, datastack, n_verts, VERT_MIN, VERT_MAX,
                )
                return

            log.info("root_id=%s datastack=%s - extracting features", root_id, datastack)
            start_time = time()
            result = condensed_hks_pipeline(
                mesh,
                verbose=False,
                n_jobs=N_JOBS,
                **_hks["condensed_hks_pipeline"],
            )
            elapsed = time() - start_time
            log.info(
                "root_id=%s datastack=%s - HKS finished in %.1f seconds (%d vertices)",
                root_id, datastack, elapsed, n_verts,
            )

        log.info("root_id=%s datastack=%s - features extracted", root_id, datastack)
        log.info("root_id=%s datastack=%s - saving -> %s", root_id, datastack, out_path)
        export_hks_output(out_path, root_id, result)
        log.info("root_id=%s datastack=%s - saved -> %s", root_id, datastack, out_path)

    except Exception:
        #tag = f"{root_id} (syn={synapse_id})" if chunked else str(root_id)
        #post_slack_message(f"HKS: {tag} failed!")
        log.error(
            "root_id=%s datastack=%s - failed\n%s",
            root_id, datastack, traceback.format_exc(),
        )
