"""Small JSONL reader for benchmark tasks."""

import json
from pathlib import Path


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{number}: invalid JSON: {exc}') from exc
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{number}: expected an object')
            yield row
