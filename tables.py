import csv
from data import MUTATIONS, NOISE_TYPES
from analysis import METHODS


REPORT_FILES = (
    "summary.json", "table_status.csv", "opt_summary.csv", "opt_wilcoxon.csv", "ver_summary.csv",
    "ver_wilcoxon.csv", "ver_groups.csv", "ver_by_version.csv", "ver_order.csv", "mut_classification.csv",
    "mut_by_operator.csv", "mut_layers.csv", "noise_summary.csv", "ablation_summary.csv", "sensitivity.csv",
    "alpha_calibration.csv", "gate_calibration.csv", "llm_attempts.csv", "llm_summary.csv", "llm_tests.csv",
    "llm_ranks.csv", "llm_nemenyi.csv", "mutation_replay.csv", "mut_operating_points.csv",
    "mut_layered_detection.csv", "mut_traceability_groups.csv", "temporal_summary.csv", "cost_summary.csv",
    "raw_cost.csv", "field_traceability.csv", "trace_health.csv", "noise_spearman.csv", "input_pairs.jsonl",
    "pair_scores.jsonl", "pair_scores.csv", "mechanism_checks.jsonl", "runtime_order_diagnostic.csv",
)


def clear_previous_reports(directory, input_files):
    inputs = {path.resolve() for path in input_files}
    for name in REPORT_FILES:
        if (directory / name).resolve() in inputs:
            raise ValueError(f"Output would replace an input file: {directory / name}")
    for name in REPORT_FILES:
        (directory / name).unlink(missing_ok=True)


def save_table(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    # print("writing CSV:", path.name, "rows:", len(rows))
    # print("CSV fields:", fields)
    # print("first row:", rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def method_rows(scores):
    return [{"method": key, **values} for key, values in scores.items()]


def save_experiment_tables(summary, out, details=False):
    opt = summary["rq1_opt"]
    ver = summary["rq2_ver"]
    mut = summary["rq3_mut"]
    # 表 6 少了某个方法时，先核对参与分类统计的方法名。
    # print("分类方法", list(mut["classification"]))
    for prefix, results in (("opt", opt), ("ver", ver)):
        save_table(out / f"{prefix}_summary.csv", method_rows(results["scores"]))
        save_table(out / f"{prefix}_wilcoxon.csv", method_rows(results["statistics"]))

    for key in ("groups", "by_version"):
        records = []
        for group, scores in ver[key].items():
            for row in method_rows(scores):
                records.append({"group": group, **row})
        save_table(out / f"ver_{key}.csv", records)
    records = []
    for group, events in ver["order"].items():
        for event_pair, values in events.items():
            if values["n"]:
                records.append({"group": group, "event_pair": event_pair, **values})
    save_table(out / "ver_order.csv", records)

    # Old whole-runtime equality is diagnostic only, never a substitute for the
    # manuscript's matched event pairs within each function.
    diagnostic = []
    for group, events in ver.get("runtime_order_diagnostic", {}).items():
        for event_pair, values in events.items():
            diagnostic.append({"group": group, "event_pair": event_pair, **values})
    save_table(out / "runtime_order_diagnostic.csv", diagnostic)

    records = []
    for row in method_rows(mut["classification"]):
        intervals = row.pop("bootstrap_ci95", {})
        for metric, (lower, upper) in intervals.items():
            row[f"{metric}_ci_low"] = lower
            row[f"{metric}_ci_high"] = upper
        records.append(row)
    save_table(out / "mut_classification.csv", records)
    records = []
    layers = []
    for operator, values in mut["by_operator"].items():
        for row in method_rows(values["scores"]):
            detection_rate = values["sitm_detection_rate"] if row["method"] == "SITM" else None
            records.append({"operator": operator, "detection_rate": detection_rate, **row})
        layers.append({"operator": operator, **values["layers"]})
    save_table(out / "mut_by_operator.csv", records)
    if details:
        save_table(out / "mut_layers.csv", layers)

    records = []
    for kind, scores in summary["rq4_noise"].items():
        for row in method_rows(scores):
            records.append({"noise": kind, **row})
    save_table(out / "noise_summary.csv", records)
    variants = summary["rq6_ablation"]["variants"]
    if any(row["roc_auc"] is not None or row["opt_mean_dev"] is not None
           or row["ver_mean_dev"] is not None for row in variants.values()):
        save_table(out / "ablation_summary.csv", method_rows(variants))
    records = []
    for parameter, curve in summary["rq6_ablation"]["sensitivity"].items():
        for row in curve:
            records.append({"parameter": parameter, **row})
    save_table(out / "sensitivity.csv", records)

    calibration = summary["calibration"]
    if "alpha" in calibration:
        alpha = calibration["alpha"]
        records = []
        for event, value in alpha["alpha"].items():
            records.append({"event": event, "alpha": value, "n": alpha["n"],
                            "mean_trace": alpha["mean_trace"], "variance": alpha["variance"]})
        save_table(out / "alpha_calibration.csv", records)
    if "gate_at_published_alpha" in calibration:
        save_table(out / "gate_calibration.csv", [calibration["gate_at_published_alpha"]])

    llm = summary["rq5_llm"]
    attempts, scores = [], []
    for model, values in llm["models"].items():
        counts = {"model": model, "attempts": values["attempts"],
                  "compiled": values["compiled"], "failure_rate": values["failure_rate"]}
        attempts.append(counts)
        for row in method_rows(values["scores"]):
            scores.append({**counts, **row})
    save_table(out / "llm_attempts.csv", attempts)
    save_table(out / "llm_summary.csv", scores)
    if "friedman" in llm:
        tests = {"balanced_n": len(llm["balanced_contracts"]), **llm["friedman"],
                 "nemenyi_cd_05": llm["nemenyi_cd_05"]}
        save_table(out / "llm_tests.csv", [tests])
        ranks = []
        for model, rank in llm["mean_rank"].items():
            ranks.append({"model": model, "mean_rank": rank})
        save_table(out / "llm_ranks.csv", ranks)
        comparisons = []
        for left, values in llm["nemenyi_p"].items():
            for right, p_value in values.items():
                if left < right:
                    comparisons.append({"model_a": left, "model_b": right, "p_value": p_value})
        save_table(out / "llm_nemenyi.csv", comparisons)


def save_core_tables(summary, out, details=False):
    mut = summary["rq3_mut"]
    replay = mut.get("differential_replay", {})
    validation = []
    for operator, row in (*replay.get("by_operator", {}).items(), ("total", replay)):
        if not row:
            continue
        sampled, executable, confirmed = (row[key] for key in ("sampled", "executable", "confirmed"))
        validation.append({"operator": operator, "sampled": sampled, "confirmed": confirmed,
                           "no_divergence": executable - confirmed, "undeployable": sampled - executable,
                           "confirmation_rate_all": confirmed / sampled if sampled else None,
                           "confirmation_rate_executable": row["confirmation_rate"],
                           "evidence_scope": row.get("evidence_scope", "legacy_unspecified"),
                           "ordered_pairs": row.get("ordered_pairs", 0),
                           "legacy_unordered_pairs": row.get("legacy_unordered_pairs", 0),
                           "diagnostic_only_pairs": row.get("diagnostic_only_pairs", 0)})
    save_table(out / "mutation_replay.csv", validation)

    operating_points = []
    for threshold in (0.60, 0.80):
        score = mut["thresholds"][str(threshold)].get("SITM")
        if score:
            values = {key: score[key] for key in ("n", "precision", "recall", "f1", "fpr")}
            operating_points.append({"threshold": threshold, **values})
    save_table(out / "mut_operating_points.csv", operating_points)

    layered, traceability = [], []
    for operator, group in mut["by_operator"].items():
        score = group["scores"]["SITM"]
        row = {"operator": operator, "n": score["n"], "sitm_median": score["median"],
               "detect_rate": group["sitm_detection_rate"]}
        if details:
            row.update(group["layers"])
        layered.append(row)
    for label, groups in (*mut["by_traceability"].items(), ("overall", mut["by_operator"])):
        for operator in ("CH", "SH"):
            if operator in groups:
                values = groups[operator]
                score = values["scores"]["SITM"]
                traceability.append({"operator": operator, "group": label, "n": score["n"],
                                     "detect_rate": values["sitm_detection_rate"], "mean": score["mean"]})
    save_table(out / "mut_layered_detection.csv", layered)
    save_table(out / "mut_traceability_groups.csv", traceability)

    temporal = mut["temporal"]
    combined = []
    for method, score in temporal["classification"].items():
        opt = temporal["optimizer_equivalence"].get(method, {})
        combined.append({"method": method, "n": score["n"], "roc_auc": score["roc_auc"],
                         "optimizer_positive_n": opt.get("n"), "optimizer_mean": opt.get("mean")})
    save_table(out / "temporal_summary.csv", combined)


# 阈值表的临时展开版本，可多取几个阈值检查灰区样本。
# def preview_operating_points(summary):
#     results = []
#     for threshold in ("0.5", "0.6", "0.7", "0.8", "0.85"):
#         score = summary["rq3_mut"]["thresholds"].get(threshold, {}).get("SITM")
#         if score is None:
#             continue
#         results.append({
#             "threshold": float(threshold),
#             "n": score["n"],
#             "precision": score["precision"],
#             "recall": score["recall"],
#             "fpr": score["fpr"],
#         })
#     return results


def all_scores_present(scores, count, methods=METHODS):
    if not count:
        return False
    for method in methods:
        if scores.get(method, {}).get("n") != count:
            return False
    return True


def classification_evidence_present(scores, count, methods=METHODS, contracts=None):
    if not all_scores_present(scores, count, methods):
        return False
    for method in methods:
        score = scores[method]
        if (not score.get("bootstrap_ci95") or score.get("bootstrap_resamples", 0) < 1000
                or not score.get("bootstrap_identity_complete", False)):
            return False
        if contracts is not None and (score.get("n_contracts") != contracts
                                      or score.get("bootstrap_unit") != "contract"):
            return False
    return True


def core_coverage(summary, out):
    """Check table inputs and sample counts against the manuscript."""
    counts = summary["counts"]
    versions = summary["rq2_ver"]["groups"]
    mut = summary["rq3_mut"]
    operators = mut["by_operator"]
    traceability = mut["by_traceability"]
    replay = mut.get("differential_replay", {}).get("by_operator", {})
    noise = summary["rq4_noise"]

    versions_ready = True
    for group, expected in (("minor", 520), ("major", 158)):
        if versions.get(group, {}).get("SITM", {}).get("n") != expected:
            versions_ready = False
    operators_ready = True
    baselines_ready = True
    for kind in MUTATIONS:
        scores = operators.get(kind, {}).get("scores", {})
        # print("operator score counts:", kind, {method: values.get("n") for method, values in scores.items()})
        if scores.get("SITM", {}).get("n") != 200:
            operators_ready = False
        if not all_scores_present(scores, 200):
            baselines_ready = False
    mut_ready = operators_ready and summary["mut_label_counts"] == {"equivalent": 1000, "mutated": 1000}

    traceability_ready = True
    for kind in ("CH", "SH"):
        static = traceability.get("static", {}).get(kind, {}).get("scores", {}).get("SITM", {}).get("n", 0)
        dynamic = traceability.get("dynamic", {}).get(kind, {}).get("scores", {}).get("SITM", {}).get("n", 0)
        # print("traceability groups:", kind, "static:", static, "dynamic:", dynamic)
        if not static or not dynamic or static + dynamic != 200:
            traceability_ready = False

    temporal = mut["temporal"]
    temporal_ready = classification_evidence_present(temporal["classification"], 450,
                         ("OED", "CFG", "E2V", "SmartTrans", "SITM"), contracts=150)
    design = summary.get("temporal_design", {})
    temporal_ready = temporal_ready and design.get("contracts") == 150 and design.get("one_positive_two_negative_per_contract", False)
    for method in ("OED", "CFG", "E2V", "SmartTrans", "SITM"):
        if temporal["optimizer_equivalence"].get(method, {}).get("n") != 150:
            temporal_ready = False
    noise_ready = all(all_scores_present(noise.get(kind, {}), 500) for kind in NOISE_TYPES)
    replay_ready = all(replay.get(kind, {}).get("sampled") == 20
                       and replay[kind].get("executable", 0) > 0
                       and replay[kind].get("ordered_pairs", 0) == replay[kind].get("executable")
                       and replay[kind].get("evidence_scope") == "observed_behavior" for kind in MUTATIONS)
    classification_ready = mut_ready and classification_evidence_present(mut["classification"], 2000)
    order = summary["rq2_ver"]["order"]
    order_ready = counts.get("opt") == 24079 and versions_ready and all(
        order.get(group) and all(order[group].get(event_pair, {}).get("complete", False)
                                and order[group][event_pair].get("n", 0) > 0
                                for event_pair in ("CALL-SSTORE", "SSTORE-LOG", "CALL-LOG"))
        for group in ("opt", "minor", "major"))

    entries = [
        (1, "field_traceability.csv", counts.get("opt") == 24079, "24,079 Opt-Pair reference bytecodes"),
        (2, "alpha_calibration.csv", counts.get("calibration") == 500, "500 equivalent calibration pairs"),
        (3, "ver_groups.csv", versions_ready, "520 minor and 158 major Ver-Pair samples"),
        (4, "ver_order.csv", order_ready, "Matched within-function event pairs; complete annotations required"),
        (5, "mutation_replay.csv", replay_ready, "20 replay cases per operator; ordered behavior evidence; gas/root diagnostics excluded"),
        (6, "mut_classification.csv", classification_ready, "1,000 equivalent and 1,000 mutated pairs, all methods"),
        (7, "mut_operating_points.csv", mut_ready, "Balanced Mut-Set at thresholds 0.60 and 0.80"),
        (8, "mut_by_operator.csv", baselines_ready, "200 negatives per operator, all methods"),
        (9, "mut_layered_detection.csv", operators_ready, "200 negatives per operator"),
        (10, "mut_traceability_groups.csv", traceability_ready, "CH/SH labeled static or dynamic"),
        (11, "temporal_summary.csv", temporal_ready, "450 pairs from 150 independent contracts; contract-cluster confidence intervals"),
        (12, "noise_summary.csv", noise_ready, "500 pairs per noise type with clean/noisy scores"),
        (13, "ablation_summary.csv", counts.get("opt") == 24079 and versions_ready and mut_ready,
         "Opt-Pair, Ver-Pair and Mut-Set; local three-layer aggregation is described in README"),
    ]
    coverage = []
    for number, filename, ready, notes in entries:
        status = "missing"
        if (out / filename).is_file():
            status = "computed" if ready else "partial"
        coverage.append({"table": number, "csv": filename, "status": status, "inputs_and_scope": notes})
        # print("table:", number, filename, "status:", status, "inputs ready:", ready)
    return coverage
