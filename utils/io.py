# ruff: noqa: T201
from __future__ import annotations

__all__ = ["serialize_fp32", "serialize_int8", "write_tensors_by_group"]

from typing import IO, TYPE_CHECKING

import numpy as np
import torch
from utils.general import extract_layer_number
from utils.quantization import quantize_q40, quantize_q80

if TYPE_CHECKING:
    from collections.abc import Sequence

    from safetensors import safe_open


def serialize_fp32(
    file: IO[bytes],
    tensor: torch.Tensor,
    chunk_size: int = 1024 * 1024,
) -> None:
    local_tensor = tensor.detach().cpu().to(torch.float32).numpy().ravel()

    for i in range(0, len(local_tensor), chunk_size):
        chunk = local_tensor[i : i + chunk_size]
        file.write(chunk.tobytes())


def serialize_int8(
    file: IO[bytes], tensor: torch.Tensor, chunk_size: int = 1024 * 1024
) -> None:
    local_tensor = tensor.detach().cpu().numpy().astype(np.int8).ravel()

    for i in range(0, len(local_tensor), chunk_size):
        chunk = local_tensor[i : i + chunk_size]
        file.write(chunk.tobytes())


def write_tensors_by_group(  # noqa: PLR0917
    files: Sequence[safe_open],
    layer_pattern: str,
    out_file: IO[bytes],
    m_type: str = "model.layers",
    quantize_type: int = 0,
    group_size: int = 128,
    splits: int = 0,
    split_idx: int = -1,
) -> list[float]:
    ew = []

    for f in files:
        layers_keys = [
            key
            for key in f.keys()  # noqa: SIM118
            if layer_pattern in key and m_type in key
        ]

        for layer in sorted(layers_keys, key=extract_layer_number):
            w = f.get_tensor(layer)

            weights = []

            if splits > 0:
                w_size = w.shape[0] // splits

                weights.append(w[split_idx * w_size : split_idx * w_size + w_size])
            else:
                weights.append(w)

            for w in weights:
                print(f"Writing: {layer} {tuple(w.shape)}")

                if quantize_type == 0:
                    serialize_fp32(out_file, w)
                else:
                    if quantize_type == 1:
                        q, s, err = quantize_q80(w, group_size)
                    elif quantize_type == 2:
                        q, s, err = quantize_q40(w, group_size)

                    serialize_int8(out_file, q)
                    serialize_fp32(out_file, s)

                    ew.append(err)
                    print(
                        f"{layer} quantized {tuple(w.shape)} to {'Q8_0' if quantize_type == 1 else 'Q4_0'} with max error {err}"
                    )

    return ew
