from pathlib import Path
from typing import Literal, NamedTuple, Union

import numpy as np
import pandas as pd
import tensorstore as ts
from cloudfiles import CloudFiles
from tqdm.auto import tqdm

# TODO make these configurable
MINISHARD_TARGET_COUNT = 1000
SHARD_TARGET_SIZE = 50000000
ID_DTYPE = ">u8"


class _ShardSpec(NamedTuple):
    type: str
    hash: Literal["murmurhash3_x86_128", "identity_hash"]
    preshift_bits: int
    shard_bits: int
    minishard_bits: int
    data_encoding: Literal["raw", "gzip"]
    minishard_index_encoding: Literal["raw", "gzip"]

    def to_json(self):
        return {
            "@type": self.type,
            "hash": self.hash,
            "preshift_bits": self.preshift_bits,
            "shard_bits": self.shard_bits,
            "minishard_bits": self.minishard_bits,
            "data_encoding": str(self.data_encoding),
            "minishard_index_encoding": str(self.minishard_index_encoding),
        }


def _choose_output_spec(
    total_count,
    total_bytes,
    hashtype: Literal["murmurhash3_x86_128", "identity_hash"] = "murmurhash3_x86_128",
    gzip_compress=True,
):
    if total_count == 1:
        return None
    if ts is None:
        return None

    # test if hashtype is valid
    if hashtype not in ["murmurhash3_x86_128", "identity_hash"]:
        raise ValueError(
            f"Invalid hashtype {hashtype}."
            "Must be one of 'murmurhash3_x86_128' "
            "or 'identity_hash'"
        )

    total_minishard_bits = 0
    while (total_count >> total_minishard_bits) > MINISHARD_TARGET_COUNT:
        total_minishard_bits += 1

    shard_bits = 0
    while (total_bytes >> shard_bits) > SHARD_TARGET_SIZE:
        shard_bits += 1

    preshift_bits = 0
    while MINISHARD_TARGET_COUNT >> preshift_bits:
        preshift_bits += 1

    minishard_bits = total_minishard_bits - min(total_minishard_bits, shard_bits)
    data_encoding: Literal["raw", "gzip"] = "raw"
    minishard_index_encoding: Literal["raw", "gzip"] = "raw"

    if gzip_compress:
        data_encoding = "gzip"
        minishard_index_encoding = "gzip"

    return _ShardSpec(
        type="neuroglancer_uint64_sharded_v1",
        hash=hashtype,
        preshift_bits=preshift_bits,
        shard_bits=shard_bits,
        minishard_bits=minishard_bits,
        data_encoding=data_encoding,
        minishard_index_encoding=minishard_index_encoding,
    )


def _interpret_path_str(path: Union[str, Path]) -> str:
    path_str = str(path).strip("/")
    if "gs:/" in path_str:
        path_str = path_str.replace("gs:/", "gs://")
    if "://" not in path_str and ":/" not in path_str:
        path_str = "file:///" + path_str
    return path_str


def _path_to_cloudfiles(path: Union[str, Path], **kwargs) -> CloudFiles:
    if isinstance(path, str):
        path = Path(path)

    if path.suffix == "":
        path = path
    else:
        raise ValueError("Path has a file suffix, should be a directory.")

    path_str = _interpret_path_str(path)
    cf = CloudFiles(path_str, **kwargs)

    return cf


def _bytes_per_element(dtypes: Union[dict, pd.Series, pd.DataFrame]) -> dict:
    """Returns a dictionary mapping column names to the number of bytes per element."""
    if isinstance(dtypes, pd.DataFrame):
        dtypes = dtypes.dtypes
    return {col: np.dtype(dtype).itemsize for col, dtype in dtypes.items()}


def _bytes_per_row(dtypes: Union[dict, pd.Series]) -> int:
    bytes_per_element = _bytes_per_element(dtypes)
    total = 0
    for _, n_bytes in bytes_per_element.items():
        total += n_bytes
    return total


class DataFrameTensorStore:
    def __init__(self, path: Union[str, Path], verbose: bool = False):
        self.cloudfiles = _path_to_cloudfiles(path)
        self.spec = self._read_spec()
        self.dtypes = self._read_dtypes()
        self.dataset = ts.KvStore.open(self.spec).result()
        self.verbose = verbose
        self.columns = list(self.dtypes.keys())
        self._n_bytes_per_element = {
            col: np.dtype(dtype).itemsize for col, dtype in self.dtypes.items()
        }

    def __repr__(self):
        return f"DataFrameTensorStore(path={self.path}, n_columns={len(self.columns)})"

    @property
    def path(self) -> str:
        return str(self.cloudfiles.cloudpath)

    def _read_spec(self) -> dict:
        if self.cloudfiles.exists("spec.json"):
            spec = self.cloudfiles.get_json("spec.json")
        else:
            raise FileNotFoundError(f"Spec file not found at {self.path}/spec.json")
        return spec

    def _read_dtypes(self) -> dict:
        if self.cloudfiles.exists("dtypes.json"):
            dtypes = self.cloudfiles.get_json("dtypes.json")
            dtypes = {key: np.dtype(value) for key, value in dtypes.items()}
        else:
            raise FileNotFoundError(f"Dtypes file not found at {self.path}/dtypes.json")
        return {key: np.dtype(value) for key, value in dtypes.items()}

    def encode_ids(self, row_id: Union[int, list[int]]) -> list[bytes]:
        if isinstance(row_id, int):
            row_id = [row_id]
        return [np.ascontiguousarray(id_, dtype=ID_DTYPE).tobytes() for id_ in row_id]

    def decode_ids(self, encoded_ids: Union[bytes, list[bytes]]) -> np.ndarray[int]:
        if isinstance(encoded_ids, bytes):
            encoded_ids = [encoded_ids]
        out = [
            np.frombuffer(encoded_id, dtype=ID_DTYPE)[0] for encoded_id in encoded_ids
        ]
        return np.array(out, dtype=ID_DTYPE)

    def write_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        dtypes = dataframe.dtypes.to_dict()
        bytes_by_column = {
            col: np.ascontiguousarray(dataframe[col].values, dtype=dtype).tobytes()
            for col, dtype in dtypes.items()
        }
        n_bytes_per_element = {
            col: np.dtype(dtype).itemsize for col, dtype in dtypes.items()
        }

        txn = ts.Transaction()
        encoded_index = self.encode_ids(dataframe.index)

        for i, key in tqdm(
            enumerate(encoded_index),
            total=len(dataframe),
            desc="Writing dataframe rows",
            disable=not self.verbose,
        ):
            value_bytes = b""
            for col in dataframe.columns:
                n_bytes = n_bytes_per_element[col]
                value_bytes += bytes_by_column[col][i * n_bytes : (i + 1) * n_bytes]
            self.dataset.with_transaction(txn)[key] = value_bytes

        txn.commit_async().result()

    def get_by_id(self, ids: Union[int, list[int]]) -> pd.DataFrame:
        if isinstance(ids, int):
            ids = [ids]
        encoded_ids = self.encode_ids(ids)
        futures = [self.dataset.read(id_) for id_ in encoded_ids]
        all_bytes = [
            future.result().value
            for future in tqdm(futures, desc="Reading rows", disable=not self.verbose)
        ]

        anns = []
        index = []
        for i, row_bytes in enumerate(
            tqdm(all_bytes, desc="Decoding rows", disable=not self.verbose)
        ):
            if len(row_bytes) == 0:
                continue
            else:
                values = []
                offset = 0
                for col, dtype in self.dtypes.items():
                    value_size = self._n_bytes_per_element[col]
                    value_bytes = row_bytes[offset : offset + value_size]
                    value = np.frombuffer(value_bytes, dtype=dtype)[0]
                    values.append(value)
                    offset += value_size
                anns.append(values)
                index.append(ids[i])

        df = pd.DataFrame(anns, index=index, columns=self.columns)
        return df

    def get_ids(self) -> list[int]:
        all_ids_bytes = self.dataset.list().result()
        all_ids = self.decode_ids(all_ids_bytes)
        return all_ids

    @classmethod
    def initialize_from_sample(
        cls,
        path: Union[str, Path],
        sample_data: Union[pd.DataFrame, dict, pd.Series],
        n_rows: int,
        hashtype="murmurhash3_x86_128",
        gzip_compress=True,
        **kwargs,
    ) -> "DataFrameTensorStore":
        """Initializes a DataFrameTensorStore from a sample DataFrame.

        Parameters
        ----------
        path :
            The path where the TensorStore will be created.
        sample_df :
            A sample DataFrame, or dict or Series of numpy types from which to infer the
            schema.
        n_rows :
            An estimate of the number of rows expected to be stored in the entire
            TensorStore (NOT the number of rows in the sample DataFrame).
        hashtype :
            The type of hash to use for sharding. See TensorStore documentation for
            details.
        gzip_compress :
            Whether to use gzip compression for the data.

        Returns
        -------
        DataFrameTensorStore
            An instance of DataFrameTensorStore initialized with the given path and
            sample data.
        """
        if isinstance(sample_data, pd.DataFrame):
            dtypes = sample_data.dtypes.to_dict()
        if not isinstance(dtypes, dict):
            dtypes = dtypes.to_dict()
        bytes_per_row = _bytes_per_row(dtypes)
        total_expected_bytes = bytes_per_row * n_rows
        shard_spec = _choose_output_spec(
            total_count=n_rows,
            total_bytes=total_expected_bytes,
            hashtype=hashtype,
            gzip_compress=gzip_compress,
        )
        path_str = _interpret_path_str(path)
        spec = {
            "driver": "neuroglancer_uint64_sharded",
            "metadata": shard_spec.to_json(),
            "base": path_str,
        }

        cloudfiles = _path_to_cloudfiles(path)
        if cloudfiles.exists("spec.json"):
            raise FileExistsError(
                f"Spec file already exists at {path_str}/spec.json.\n"
                "Choose a different path or clear the existing TensorStore."
            )

        cloudfiles.put_json("spec.json", spec)

        dtypes_strs = {key: dt.name for key, dt in dtypes.items()}
        if cloudfiles.exists("dtypes.json"):
            raise FileExistsError(
                f"Dtypes file already exists at {path_str}/dtypes.json.\n"
                "Choose a different path or clear the existing TensorStore."
            )
        cloudfiles.put_json("dtypes.json", dtypes_strs)

        return cls(path, **kwargs)

    @classmethod
    def exists(cls, path: Union[str, Path]) -> bool:
        """Checks if a DataFrameTensorStore exists at the given path."""
        cloudfiles = _path_to_cloudfiles(path)
        return cloudfiles.exists("spec.json") and cloudfiles.exists("dtypes.json")
