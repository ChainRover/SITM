"""Score the selected pairs, calculate statistics, and write experiment tables."""

import hashlib
import json
import csv
import math
from collections import Counter
from importlib.metadata import PackageNotFoundError, version

from analysis import METHODS, external_scores, fit_alpha, fit_gate, fit_sf, fuse, score_pairs, summarize
from data import read_jsonl, write_jsonl
from measurements import field_traceability, measure_cost, noise_correlations
from mutations import verify_replays
from tables import clear_previous_reports, core_coverage, save_core_tables, save_experiment_tables, save_table


def json_file(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


# def preview_pair_scores(pairs, model_records, count=3):
#     selected = pairs[:count]
#     identifiers = {row["id"] for row in selected}
#     records = [row for row in model_records if row["id"] in identifiers]
#     scores = external_scores(records, selected)
#     rows = score_pairs(selected, scores)
#     for row in rows:
#         print(row["id"], row.get("scores", {}))
#         if "sitm_components" in row:
#             print(row["sitm_components"])
#     return rows


def evaluate(pairs, args, sources):
    identifiers = {row["id"] for row in pairs}
    # print("input pairs / unique IDs:", len(pairs), len(identifiers))
    # print("dataset counts:", Counter(row["dataset"] for row in pairs))
    # print("output directory:", args.out, "details:", args.details)
    model_records = []
    for path in args.scores:
        records = read_jsonl(path)
        matched = [row for row in records if row["id"] in identifiers]
        # print("model score file:", path.name, "records:", len(records), "matched:", len(matched))
        # print("unmatched score IDs:", [row["id"] for row in records if row["id"] not in identifiers][:5])
        if records and not matched:
            raise ValueError(f"No score IDs from {path} match the selected pair IDs")
        model_records.extend(matched)
    rows = score_pairs(pairs, external_scores(model_records, pairs))
    # 接入模型分数后，先看有没有按 ID 合并进来。
    # for method in ("E2V", "SmartTrans"):
    #     print(method, sum(method in row.get("scores", {}) for row in rows))
    mixed = [row for row in rows if row["dataset"] == "mixed"]
    # print("mixed label counts:", Counter(row["label"] for row in mixed))

    calibration = {}
    if args.sf_weights is not None:
        weights = args.sf_weights
        if not all(math.isfinite(weight) and weight >= 0 for weight in weights) or abs(sum(weights) - 1) > 1e-8:
            raise ValueError("SF weights must be nonnegative and sum to one")
        calibration["sf"] = {"weights": weights, "origin": "explicit"}
    elif {row["label"] for row in mixed} == {0, 1}:
        calibration["sf"] = fit_sf(mixed)
        calibration["sf"]["origin"] = "mixed"
    if "sf" in calibration:
        # print("SF weights:", calibration["sf"]["weights"], "origin:", calibration["sf"]["origin"])
        for row in rows:
            for field in ("scores", "clean_scores"):
                if field in row:
                    row[field]["SF"] = fuse(row[field], calibration["sf"]["weights"])
                    # print("SF check:", row["id"], field, row[field]["SF"])

    if args.calibrate:
        equivalent = [row for row in pairs if row["dataset"] == "calibration"]
        # print("equivalent calibration pairs:", len(equivalent))
        if equivalent:
            calibration["alpha"] = fit_alpha(equivalent)
        if {row["label"] for row in mixed} == {0, 1}:
            calibration["gate_at_published_alpha"] = fit_gate(mixed)

    summary = summarize(pairs, rows, args.threshold, args.bootstrap, args.seed)
    # print("summary sections:", sorted(summary))
    # print("threshold / bootstrap / seed:", args.threshold, args.bootstrap, args.seed)
    summary["calibration"] = calibration
    summary["experiment"] = args.exp
    summary["generated_mutation_positives"] = args.generated_positives
    summary["mut_label_counts"] = {
        "equivalent": sum(row["dataset"] == "mut" and row.get("label") == 1 for row in pairs),
        "mutated": sum(row["dataset"] == "mut" and row.get("label") == 0 for row in pairs),
    }
    summary["pairs_without_address"] = sum("contract_id_source" in row for row in pairs)
    summary["sample_origins"] = dict(Counter(row.get("sample_origin", "provided") for row in pairs))
    summary["input_files"] = []
    for path in dict.fromkeys([*sources, *args.scores, *args.replays]):
        summary["input_files"].append({"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    summary["dependencies"] = {}
    for name in ("py-solc-x", "numpy", "scipy", "scikit-learn", "networkx", "cbor2", "rapidfuzz"):
        try:
            summary["dependencies"][name] = version(name)
        except PackageNotFoundError:
            summary["dependencies"][name] = "unavailable"

    negative = {row["id"]: row["mutation"] for row in pairs
                if row["dataset"] == "mut" and row.get("label") == 0}
    replay_records = []
    for path in args.replays:
        records = read_jsonl(path)
        matched = [row for row in records if row["id"] in negative]
        # print("replay file:", path.name, "records:", len(records), "matched negatives:", len(matched))
        if records and not matched:
            raise ValueError(f"No replay IDs from {path} match the selected mutation negatives")
        replay_records.extend(matched)
    if replay_records:
        validation = verify_replays(replay_records, set(negative))
        validation["by_operator"] = {}
        for kind in sorted(set(negative.values())):
            group = [row for row in replay_records if negative[row["id"]] == kind]
            # print("replay operator coverage:", kind, len(group))
            if group:
                validation["by_operator"][kind] = verify_replays(group, set(negative))
        if not args.details:
            validation.pop("pairs")
            for group in validation["by_operator"].values():
                group.pop("pairs")
        summary["rq3_mut"]["differential_replay"] = validation

    args.out.mkdir(parents=True, exist_ok=True)
    clear_previous_reports(args.out, [*sources, *args.scores, *args.replays])
    opt = [row for row in pairs if row["dataset"] == "opt"]
    if args.exp in {"all", "core", "smoke", "6"} and opt:
        summary["cost"] = measure_cost(opt, args.cost_samples, args.warmup, args.seed, args.out, args.details)
    if args.exp in {"all", "core", "smoke", "9", "10"} and opt:
        summary["field_traceability"] = field_traceability(opt, args.out, args.details)
    if args.details:
        noise = [row for row in rows if row["dataset"] == "noise"]
        summary["noise_spearman"] = noise_correlations(noise)
        save_table(args.out / "noise_spearman.csv", summary["noise_spearman"])
    save_experiment_tables(summary, args.out, args.details)
    save_core_tables(summary, args.out, args.details)
    summary["table_coverage"] = core_coverage(summary, args.out)
    # for item in summary["table_coverage"]:
    #     print("table coverage:", item)
    save_table(args.out / "table_status.csv", summary["table_coverage"])

    llm = summary["rq5_llm"]
    llm["balanced_n"] = len(llm["balanced_contracts"])
    if not args.details:
        llm.pop("balanced_contracts")
    if args.details:
        write_jsonl(args.out / "input_pairs.jsonl", pairs)
        write_jsonl(args.out / "pair_scores.jsonl", rows)
        with (args.out / "pair_scores.csv").open("w", encoding="utf-8", newline="") as stream:
            fields = ("id", "dataset", "contract_id", "label", "mutation", "model", "status", *METHODS)
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                # print("CSV score fields:", row["id"], sorted(row.get("scores", {})))
                writer.writerow({field: row.get(field, row.get("scores", {}).get(field, "")) for field in fields})
    json_file(args.out / "summary.json", summary)
    print(f"{args.out}: evaluated {len(rows)} pairs")
    print(f"Dataset counts: {summary['counts']}")
    print(f"Table coverage: {args.out / 'table_status.csv'}")
    return summary
