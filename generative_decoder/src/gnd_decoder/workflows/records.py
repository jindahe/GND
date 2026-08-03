import json
from datetime import datetime, timezone
from pathlib import Path


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json_record(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
