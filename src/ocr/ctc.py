from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class Charset:
    chars: str

    @property
    def blank_index(self) -> int:
        return len(self.chars)

    @property
    def num_classes(self) -> int:
        return len(self.chars) + 1

    @property
    def char_to_idx(self) -> Dict[str, int]:
        return {c: i for i, c in enumerate(self.chars)}

    @property
    def idx_to_char(self) -> Dict[int, str]:
        return {i: c for i, c in enumerate(self.chars)}


def encode_label(text: str, charset: Charset, max_len: int) -> Tuple[np.ndarray, int]:
    text = text.strip().upper()
    encoded = [charset.char_to_idx[c] for c in text if c in charset.char_to_idx]
    length = len(encoded)
    if length > max_len:
        encoded = encoded[:max_len]
        length = max_len
    padded = np.zeros((max_len,), dtype=np.int32)
    if length > 0:
        padded[:length] = np.asarray(encoded, dtype=np.int32)
    return padded, length


def greedy_decode(batch_logits: np.ndarray, charset: Charset) -> List[str]:
    # batch_logits: [batch, time, classes]
    blank = charset.blank_index
    preds = np.argmax(batch_logits, axis=-1)
    results = []
    for seq in preds:
        dedup = []
        prev = None
        for idx in seq.tolist():
            if idx == prev:
                continue
            prev = idx
            if idx == blank:
                continue
            dedup.append(charset.idx_to_char.get(idx, ""))
        results.append("".join(dedup))
    return results


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    dp = np.zeros((len(a) + 1, len(b) + 1), dtype=np.int32)
    dp[0, :] = np.arange(len(b) + 1)
    dp[:, 0] = np.arange(len(a) + 1)

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )
    return int(dp[len(a), len(b)])


def cer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _levenshtein(reference, hypothesis) / float(len(reference))


def batch_cer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    if not references:
        return 0.0
    errors = [cer(r, h) for r, h in zip(references, hypotheses)]
    return float(np.mean(errors))


def exact_match_accuracy(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    if not references:
        return 0.0
    matches = [1.0 if r == h else 0.0 for r, h in zip(references, hypotheses)]
    return float(np.mean(matches))
