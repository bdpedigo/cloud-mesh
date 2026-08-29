"""
run_targeted.py — Run the *chunked* (synapse-targeted) HKS pipeline locally for
one synapse, bypassing the task queue and writing the result to a local folder.

Useful for smoke-testing the chunked path on a single big presynapse before
re-deploying the cluster. Mirrors how the worker invokes ``run_hks`` from a
Pub/Sub task, but with no Pub/Sub, no Docker, and no GCS upload — output goes
straight to disk.

Usage
-----
From the command line::

    uv run run_targeted.py --datastack minnie65_phase3_v1 \\
        --root-id 864691135436446706 --synapse-id 12345678 \\
        --query-point 200008 148538 2959 --query-units voxel

Output lands at ``{--out-dir}/{root_id}_syn_{synapse_id}.npz`` (flat, unlike
the datastack/features/ nesting used in the GCS bucket).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Sequence


logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("run_targeted")


# ── defaults ───────────────────────────────────────────────────────────────────

# Where finished .npz files land. The script writes
# ``{DEFAULT_OUT_DIR}/{root_id}_syn_{synapse_id}.npz`` (flat, no datastack/
# features/ nesting — that layout is for the GCS bucket).
DEFAULT_OUT_DIR = Path(__file__).parent / "targeted_output"
DEFAULT_QUERY_RADIUS_NM = 50.0  # mirrors enqueue_pubsub default


# ── core ───────────────────────────────────────────────────────────────────────

def run_one(
    root_id: int,
    synapse_id: int,
    query_point: Sequence[float],
    datastack: str,
    *,
    query_units: str = "voxel",
    query_radius_nm: float = DEFAULT_QUERY_RADIUS_NM,
    voxel_size_nm: Optional[Sequence[float]] = None,
    out_dir: Path = DEFAULT_OUT_DIR,
    recompute: bool = True,
) -> Path:
    """Run the chunked HKS pipeline for a single synapse and return the
    written .npz path.

    Parameters
    ----------
    root_id : int
        Pre-synaptic root id whose mesh is sliced and analysed (the same
        root id the chunked output is tagged by).
    synapse_id : int
        Synapse id, used to name the output file.
    query_point : (3,) sequence
        Synapse center in either voxel or nm units (see ``query_units``).
    datastack : str
        CAVE datastack — also used for client.info.viewer_resolution()
        when ``voxel_size_nm`` is None and ``query_units`` is "voxel".
    query_units : {"voxel", "nm"}
        Units of ``query_point``. CAVE ``ctr_pt_position`` columns are in
        voxels; the chunked HKS pipeline wants nm.
    query_radius_nm : float
        Radius (nm) of the ball around ``query_point`` to compute HKS over.
    voxel_size_nm : (3,) sequence, optional
        Voxel size in nm. Only used when ``query_units == "voxel"``. If
        omitted, pulled from CAVE via ``client.info.viewer_resolution()``.
    out_dir : path
        Folder for the output .npz. The file lands at
        ``{out_dir}/{root_id}_syn_{synapse_id}.npz``.
    recompute : bool
        If True (default), force run_hks to recompute even when an output
        file already exists.

    Returns
    -------
    Path
        Local path to the written .npz file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_path = out_dir / f"{int(root_id)}_syn_{int(synapse_id)}.npz"

    # 1. Resolve query point to nm.
    qp = [float(c) for c in query_point]
    units = query_units.lower()
    if units == "nm":
        query_point_nm = qp
    elif units == "voxel":
        if voxel_size_nm is None:
            # CAVE ``ctr_pt_position`` is at segmentation MIP-0 voxels,
            # which is what ``viewer_resolution()`` returns. Coordinates
            # read off neuroglancer's cursor display are at image MIP-0
            # instead; where those scales differ, pass --voxel-size.
            from caveclient import CAVEclient
            client = CAVEclient(datastack_name=datastack)
            voxel_size_nm = tuple(
                float(v) for v in client.info.viewer_resolution()
            )
            log.info(
                "voxel_size auto-detected from viewer_resolution: %s nm",
                voxel_size_nm,
            )
        vs = [float(v) for v in voxel_size_nm]
        if len(vs) != 3:
            raise ValueError(f"voxel_size_nm must be length-3, got {voxel_size_nm!r}")
        query_point_nm = [qp[i] * vs[i] for i in range(3)]
        log.info(
            "converted voxel %s × %s nm/voxel -> nm %s",
            qp, vs, query_point_nm,
        )
    else:
        raise ValueError(f"query_units must be 'voxel' or 'nm', got {query_units!r}")

    # 2. Import run_hks and override its output-path machinery so files land
    #    at out_dir/{root_id}_syn_{synapse_id}.npz (a flat layout the user
    #    asked for, rather than the GCS bucket nesting). We patch the
    #    module-level _output_path function and the OUTPUT_BUCKET constant
    #    after import.
    import run_hks as _rh

    _rh.OUTPUT_BUCKET = str(out_dir)
    _rh.RECOMPUTE = bool(recompute)

    def _flat_output_path(output_bucket, datastack, root_id, synapse_id=None):
        base = Path(output_bucket)
        if synapse_id is None:
            return str(base / f"{int(root_id)}.npz")
        return str(base / f"{int(root_id)}_syn_{int(synapse_id)}.npz")

    _rh._output_path = _flat_output_path  # type: ignore[attr-defined]

    log.info(
        "running chunked HKS: root_id=%s synapse_id=%s query_point_nm=%s "
        "radius=%.1f nm -> %s",
        root_id, synapse_id, query_point_nm, float(query_radius_nm), expected_path,
    )

    _rh.run_hks(
        int(root_id),
        datastack,
        synapse_id=int(synapse_id),
        query_point_nm=query_point_nm,
        query_radius_nm=float(query_radius_nm),
    )

    if not expected_path.exists():
        log.warning(
            "run_hks completed but %s does not exist — likely the chunked "
            "pipeline hit a min-size filter or pruned mesh was empty; check "
            "the log lines above.", expected_path,
        )
    else:
        log.info(
            "wrote %s (%.1f MB)",
            expected_path, expected_path.stat().st_size / (1024 * 1024),
        )
    return expected_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run targeted (chunked) HKS for one synapse locally.",
    )
    p.add_argument("--root-id", required=True, type=int,
                   help="Pre-synaptic root id to slice the mesh from.")
    p.add_argument("--synapse-id", required=True, type=int,
                   help="Synapse id (used to name the output file).")
    p.add_argument(
        "--query-point", nargs=3, type=float, required=True,
        metavar=("X", "Y", "Z"),
        help="Synapse center; units controlled by --query-units.",
    )
    p.add_argument(
        "--query-units", default="voxel", choices=("voxel", "nm"),
        help="Units of --query-point (default: voxel, matching CAVE "
             "ctr_pt_position).",
    )
    p.add_argument(
        "--query-radius-nm", type=float, default=DEFAULT_QUERY_RADIUS_NM,
        help=f"Ball radius (nm) for chunked HKS (default: {DEFAULT_QUERY_RADIUS_NM}).",
    )
    p.add_argument(
        "--datastack", required=True,
        help="CAVE datastack name, e.g. minnie65_phase3_v1",
    )
    p.add_argument(
        "--voxel-size", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"),
        help="Voxel size in nm for voxel->nm conversion. If omitted, "
             "auto-detected from CAVE.",
    )
    p.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help=f"Output folder (default: {DEFAULT_OUT_DIR}).",
    )
    p.add_argument(
        "--no-recompute", action="store_true",
        help="If the output file already exists, skip the run.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_one(
        root_id=args.root_id,
        synapse_id=args.synapse_id,
        query_point=args.query_point,
        datastack=args.datastack,
        query_units=args.query_units,
        query_radius_nm=args.query_radius_nm,
        voxel_size_nm=args.voxel_size,
        out_dir=args.out_dir,
        recompute=not args.no_recompute,
    )


if __name__ == "__main__":
    main()
