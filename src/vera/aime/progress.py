from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ProgressEntry:
    timestamp: float
    seed_id: str
    stage: str
    message: str
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["iso_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp))
        return out


class ProgressLogger:
    """
    Collects fine-grained progress updates and optionally streams them to stdout.
    """

    def __init__(self, stream: Any = None, capture: bool = True):
        self._stream = stream if stream is not None else sys.stdout
        self._capture = capture
        self._entries: List[ProgressEntry] = []

    @property
    def entries(self) -> List[ProgressEntry]:
        return list(self._entries)

    def log(self, seed_id: str, stage: str, message: str, **payload: Any) -> None:
        now = time.time()
        entry = ProgressEntry(timestamp=now, seed_id=seed_id, stage=stage, message=message, payload=dict(payload))
        line = f"[{time.strftime('%H:%M:%S', time.gmtime(now))}] [{seed_id}] [{stage}] {message}"
        if payload:
            extras = ", ".join(f"{k}={payload[k]}" for k in sorted(payload))
            line += f" ({extras})"
        if self._stream:
            print(line, file=self._stream, flush=True)
        if self._capture:
            self._entries.append(entry)

    def dump_json(self, path: str) -> None:
        data = [entry.to_dict() for entry in self._entries]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
