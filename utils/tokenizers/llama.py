from __future__ import annotations

__all__ = ["Tokenizer"]

import struct

from transformers import AutoTokenizer

_PACK_Ix4 = struct.Struct("IIII").pack
_PACK_fI = struct.Struct("fI").pack


class Tokenizer:
    __slots__ = [
        "bos_id",
        "eos_id",
        "model",
        "n_words",
    ]

    def __init__(self, model_id: str) -> None:
        self.model = AutoTokenizer.from_pretrained(model_id, use_fast=False)

        self.n_words = len(self.model)

        self.bos_id = self.model.bos_token_id
        self.eos_id = self.model.eos_token_id

    def export(self) -> None:
        tokens, scores = [], []
        for i in range(self.n_words):
            t = self.model.decode([i])

            # just for easier compatibility with sentencepiece tokenizers
            s = 1.0

            b = t.encode("utf-8")

            tokens.append(b)
            scores.append(s)

        # record the max token length
        max_token_length = max(len(t) for t in tokens)

        with open("tokenizer.bin", "wb") as f:
            f.write(_PACK_Ix4(self.n_words, max_token_length, self.bos_id, self.eos_id))
            for bytes, score in zip(tokens, scores, strict=True):
                f.write(_PACK_fI(score, len(bytes)))
                f.write(bytes)
