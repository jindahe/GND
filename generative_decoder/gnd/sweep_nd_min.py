import argparse
import csv
import json
from pathlib import Path

from .records import utc_timestamp, write_json_record


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate model MI records into n_d^min(L).")
    parser.add_argument("--true-result", action="append", default=[], required=True)
    parser.add_argument("--model-result", action="append", default=[], required=True)
    parser.add_argument("--relative-tolerance", type=float, default=0.1)
    parser.add_argument("--capacity-key", default="parameter_count")
    parser.add_argument("--output-json", default="net/gnd/nd_min/nd_min.json")
    parser.add_argument("--output-csv", default="net/gnd/nd_min/nd_min.csv")
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_json(path):
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def result_map(payload):
    return {item["cut"]["name"]: item for item in payload["results"]}


def model_capacity(payload, capacity_key):
    capacity = payload.get("model_capacity", {})
    if capacity_key not in capacity:
        raise ValueError(
            f"Model result {payload.get('source_path')} lacks model_capacity.{capacity_key}; "
            "record capacity before nd_min aggregation"
        )
    return capacity[capacity_key]


def main():
    args = parse_args()
    true_by_l = {}
    for path in args.true_result:
        payload = load_json(path)
        true_by_l[payload["code"]["d"]] = payload

    candidates = []
    for path in args.model_result:
        payload = load_json(path)
        capacity = model_capacity(payload, args.capacity_key)
        candidates.append((payload["code"]["d"], capacity, payload))

    rows = []
    for l_value, true_payload in sorted(true_by_l.items()):
        true_results = result_map(true_payload)
        for cut_name, true_result in true_results.items():
            passing = []
            for candidate_l, capacity, candidate in candidates:
                if candidate_l != l_value:
                    continue
                model_results = result_map(candidate)
                if cut_name not in model_results:
                    continue
                true_mi = true_result["mi"]
                model_mi = model_results[cut_name]["mi"]
                relative_error = abs(model_mi - true_mi) / max(abs(true_mi), 1e-12)
                if relative_error <= args.relative_tolerance:
                    passing.append((capacity, relative_error, candidate))
            passing.sort(key=lambda item: (item[0], item[1]))
            best = passing[0] if passing else None
            rows.append(
                {
                    "L": l_value,
                    "cut": cut_name,
                    "true_mi": true_result["mi"],
                    "relative_tolerance": args.relative_tolerance,
                    "capacity_key": args.capacity_key,
                    "nd_min": best[0] if best else None,
                    "relative_error": best[1] if best else None,
                    "model_source": best[2]["source_path"] if best else None,
                }
            )

    output_json = resolve_path(args.output_json)
    output_csv = resolve_path(args.output_csv)
    write_json_record(
        output_json,
        {
            "record_type": "gnd_nd_min_summary",
            "schema_version": 1,
            "created_at_utc": utc_timestamp(),
            "capacity_key": args.capacity_key,
            "relative_tolerance": args.relative_tolerance,
            "rows": rows,
        },
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print(f"rows: {len(rows)}")
    print(f"saved: {output_json}")
    print(f"saved: {output_csv}")


if __name__ == "__main__":
    main()
