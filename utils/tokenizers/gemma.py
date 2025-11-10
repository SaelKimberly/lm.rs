from __future__ import annotations

__all__ = ["Tokenizer"]

import struct

from transformers import AutoTokenizer

# Convert sentencepiece tokenizer model into lmrs format (slight modification from karpathy's code)
_PACK_Ix4 = struct.Struct("IIII").pack
_PACK_fI = struct.Struct("fI").pack


class Tokenizer:
    __slots__ = [
        "bos_id",
        "eos_id",
        "n_words",
        "pad_id",
        "sp_model",
    ]

    def __init__(self, model_id: str) -> None:
        self.sp_model = AutoTokenizer.from_pretrained(model_id, use_fast=False).sp_model

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()

        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def export(self) -> None:

        # get all the tokens (postprocessed) and their scores as floats
        tokens, scores = [], []
        for i in range(self.n_words):
            # decode the token and light postprocessing
            t = self.sp_model.id_to_piece(i)
            s = self.sp_model.get_score(i)
            if i == self.bos_id:
                t = "\n<s>\n"
            elif i == self.eos_id:
                t = "\n</s>\n"
            t = t.replace("▁", " ")  # sentencepiece uses this character as whitespace
            b = t.encode("utf-8")  # bytes of this token, utf-8 encoded

            tokens.append(b)
            scores.append(s)

        # record the max token length
        max_token_length = max(len(t) for t in tokens)

        with open("tokenizer.bin", "wb") as f:
            f.write(_PACK_Ix4(self.n_words, max_token_length, self.bos_id, self.eos_id))
            for bytes, score in zip(tokens, scores, strict=True):
                f.write(_PACK_fI(score, len(bytes)))
                f.write(bytes)
