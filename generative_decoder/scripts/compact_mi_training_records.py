#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def sequence_summary(values):
    if values is None:
        return None
    encoded = json.dumps(values, separators=(",", ":")).encode("utf-8")
    return {
        "length": len(values),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def partition_summary(partition):
    if partition is None:
        return None
    return {
        "axis": partition["axis"],
        "cut": partition["cut"],
        "len_A": len(partition["idx_A"]),
        "len_B": len(partition["idx_B"]),
        "n_coords": len(partition.get("coords", [])),
        "idx_A": sequence_summary(partition["idx_A"]),
        "idx_B": sequence_summary(partition["idx_B"]),
        "order_AB": sequence_summary(partition["order_AB"]),
        "order_BA": sequence_summary(partition["order_BA"]),
    }


def compact_dataset(dataset):
    if not isinstance(dataset, dict):
        return dataset, False

    had_verbose = "partition" in dataset or "applied_order" in dataset
    compact = {
        "path": dataset.get("path"),
        "meta": dataset.get("meta"),
        "partition_summary": dataset.get("partition_summary"),
        "applied_order_summary": dataset.get("applied_order_summary"),
        "shape": dataset.get("shape"),
    }
    if compact["partition_summary"] is None and "partition" in dataset:
        compact["partition_summary"] = partition_summary(dataset["partition"])
    if compact["applied_order_summary"] is None and "applied_order" in dataset:
        compact["applied_order_summary"] = sequence_summary(dataset["applied_order"])
    return compact, had_verbose


def compact_record(path, dry_run):
    original_text = path.read_text(encoding="utf-8")
    payload = json.loads(original_text)
    if payload.get("record_type") != "syndrome_training":
        return False, len(original_text), len(original_text)

    compact_dataset_payload, changed = compact_dataset(payload.get("dataset"))
    if not changed:
        return False, len(original_text), len(original_text)

    payload["schema_version"] = max(int(payload.get("schema_version", 1)), 2)
    payload["dataset"] = compact_dataset_payload
    new_text = json.dumps(payload, indent=2) + "\n"

    if not dry_run:
        path.write_text(new_text, encoding="utf-8")
    return True, len(original_text), len(new_text)


def iter_record_paths(root):
    yield from sorted(root.glob("**/models/records/*.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("net/mi_scaling"))
    parser.add_argument("--apply", action="store_true", help="Rewrite records in place.")
    args = parser.parse_args()

    changed = 0
    old_total = 0
    new_total = 0
    for path in iter_record_paths(args.root):
        did_change, old_size, new_size = compact_record(path, dry_run=not args.apply)
        old_total += old_size
        new_total += new_size
        if did_change:
            changed += 1
            print(f"compact {path} {old_size}->{new_size}")

    mode = "APPLIED" if args.apply else "DRY_RUN"
    saved = old_total - new_total
    print(f"{mode} records_changed={changed} bytes_before={old_total} bytes_after={new_total} bytes_saved={saved}")


if __name__ == "__main__":
    main()
