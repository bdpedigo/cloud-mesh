"""
generate_queue.py — CAVE synapse-table loaders shared by the enqueue and
status-reporting CLIs.

Each loader either queries CAVE through an authenticated ``CAVEclient`` or
reads a previously saved dataframe off disk, so a batch can be re-enqueued
or re-checked without re-hitting the materialization service.
"""

from pathlib import Path

import pandas as pd


def get_synaptic_partners(root_id, client, output_dir=None):
    """
    Query synaptic partners of a root_id and save them to a text file.

    Parameters
    ----------
    root_id : int
        The root ID of the neuron to query.
    client : CAVEclient
        An authenticated CAVEclient instance.
    output_dir : Path
        Directory to save the output text file.

    Returns
    -------
    syn_id : set
        Set of all synaptic partner root IDs (including the query neuron).
    """
    root_id = int(root_id)
    syn_id = set([root_id]) #  int(root_id) #

    pre_syn = client.materialize.synapse_query(post_ids=[root_id], remove_autapses=True)
    pre_id = pre_syn["pre_pt_root_id"].to_list()
    syn_id = syn_id.union(set(pre_id))

    post_syn = client.materialize.synapse_query(pre_ids=[root_id], remove_autapses=True)
    post_id = post_syn["post_pt_root_id"].to_list()
    syn_id = syn_id.union(set(post_id))

    if output_dir:
        syn_txt_path = output_dir / f"{root_id}_syn_partners.txt"
        if not syn_txt_path.exists():
            with open(syn_txt_path, "w") as f:
                f.write("\n".join(str(item) for item in syn_id))
            print(f"Saved {len(syn_id)} partners to {syn_txt_path}")
        else:
            print(f"File already exists: {syn_txt_path}")

    return syn_id


def _read_synapse_file(file_path):
    """Load a saved synapse dataframe. Supported: .parquet/.csv/.feather/.ftr/.pkl/.pickle."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Synapse file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".feather", ".ftr"):
        return pd.read_feather(path)
    if suffix in (".pkl", ".pickle"):
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported synapse file extension: {suffix}")


def get_all_synapse_df(root_id, client=None, file_path=None):
    """
    Return a dataframe of *all* synapses touching ``root_id``, in BOTH
    directions: those where ``root_id`` is the post-synaptic partner and
    those where it is the pre-synaptic partner.

    Each row is one synapse and carries at least:
      - ``id``               — the synapse id
      - ``pre_pt_root_id``   — the pre-synaptic neuron
      - ``post_pt_root_id``  — the post-synaptic neuron
      - ``ctr_pt_position``  — synapse center, in voxel units

    Parameters
    ----------
    root_id : int
        Root id whose synapses (in either direction) we want.
    client : CAVEclient, optional
        Authenticated CAVEclient. Required when ``file_path`` is None.
    file_path : str | Path, optional
        If provided, load the dataframe from disk instead of querying CAVE.
        Supported extensions: ``.parquet``, ``.csv``, ``.feather``/``.ftr``,
        ``.pkl``/``.pickle``.

    Returns
    -------
    pandas.DataFrame
    """
    root_id = int(root_id)

    if file_path is not None:
        return _read_synapse_file(file_path)

    if client is None:
        raise ValueError(
            "Either an authenticated CAVEclient or a file_path must be provided."
        )

    post_syn = client.materialize.synapse_query(
        post_ids=[root_id], remove_autapses=True
    )
    pre_syn = client.materialize.synapse_query(
        pre_ids=[root_id], remove_autapses=True
    )
    return pd.concat([post_syn, pre_syn], ignore_index=True)


def get_post_synapse_df(root_id, client=None, file_path=None):
    """
    Return a dataframe of synapses where ``root_id`` is the post-synaptic
    partner.

    Each row is one synapse and is expected to carry at least:
      - ``id``               — the synapse id (used to tag chunked HKS outputs)
      - ``pre_pt_root_id``   — the pre-synaptic neuron (the analysis target)
      - ``ctr_pt_position``  — synapse center, in voxel units (length-3 sequence)

    Parameters
    ----------
    root_id : int
        Post-synaptic root id whose partners we want.
    client : CAVEclient, optional
        Authenticated CAVEclient. Required when ``file_path`` is None.
    file_path : str | Path, optional
        If provided, load the dataframe from disk instead of querying CAVE.
        Supported extensions: ``.parquet``, ``.csv``, ``.feather``/``.ftr``,
        ``.pkl``/``.pickle``.

    Returns
    -------
    pandas.DataFrame
    """
    root_id = int(root_id)

    if file_path is not None:
        return _read_synapse_file(file_path)

    if client is None:
        raise ValueError(
            "Either an authenticated CAVEclient or a file_path must be provided."
        )
    return client.materialize.synapse_query(
        post_ids=[root_id], remove_autapses=True
    )
