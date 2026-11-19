# import numpy as np
import math
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import product

import numpy as np
from scipy import stats
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
# import argparse
from evm_sit import Bytecode, strip_metadata
from data import MUTATIONS
from util import (
    ALPHA_WEIGHTS, GAMMA, THETA_HI, THETA_LO, W_TAU, _edit_similarity, _trace_score,
    apply_soft_gating, calc_cfg_isomorphism_similarity_bc, calc_ngram_similarity_bc,
    calc_opcode_edit_similarity_bc, compare_traces, extract_positioned_trace,
    extract_static_interleaved_trace, softmax_weights,
    weighted_levenshtein,
)


STRUCTURAL = ("OED", "CFG", "NG")
METHODS = (*STRUCTURAL, "SF", "E2V", "SmartTrans", "SITM")
ABLATIONS = ("SITM", "w/o_interleave(3layer)", "w/o_soft_gating", "w/o_type_weight", "w/o_fine_fields")
ORDER_PAIRS = (("CALL", "SSTORE"), ("SSTORE", "LOG"), ("CALL", "LOG"))


@lru_cache(maxsize=128)
def trace(bytecode: bytes) -> tuple:
    return tuple(extract_static_interleaved_trace(bytecode, strip_meta=False))


def traces(row: dict) -> tuple[tuple, tuple]:
    return trace(strip_metadata(row["a"])), trace(strip_metadata(row["b"]))


@lru_cache(maxsize=128)
def structural_scores(a: Bytecode, b: Bytecode) -> dict[str, float]:
    return {"OED": calc_opcode_edit_similarity_bc(a, b),
            "CFG": calc_cfg_isomorphism_similarity_bc(a, b),
            "NG": calc_ngram_similarity_bc(a, b)}


def fuse(scores: dict, weights) -> float:
    return sum(weight * scores[method] for weight, method in zip(weights, STRUCTURAL))


def fit_sf(mixed: list[dict]) -> dict:
    labels = [row["label"] for row in mixed]
    if len(set(labels)) != 2:
        raise ValueError("SF calibration needs both labels in the mixed split")
    matrix = np.array([[row["scores"][key] for key in STRUCTURAL] for row in mixed])
    # print("SF methods:", STRUCTURAL, "matrix:", matrix.shape)
    # print("mixed labels:", labels.count(1), labels.count(0))
    # print("first feature rows:", matrix[:3])
    best_key = None
    best = None
    for oed in range(21):
        for cfg in range(21 - oed):
            candidate = (oed, cfg, 20 - oed - cfg)
            score = float(roc_auc_score(labels, matrix @ candidate))
            # if oed % 5 == 0 and cfg % 5 == 0:
            #     print("SF candidate:", candidate, "AUC:", score)
            # Equal AUCs prefer a more balanced weight vector.
            key = (score, -sum(weight * weight for weight in candidate))
            if best_key is None or key > best_key:
                best_key = key
                best = candidate
                # print("SF current best:", best, "AUC:", score)
    weights = tuple(weight / 20 for weight in best)
    # assert abs(sum(weights) - 1.0) < 1e-12
    return {"weights": weights, "roc_auc": float(roc_auc_score(labels, matrix @ weights)),
            "n": len(labels), "grid_step": 0.05}


# 几组固定 SF 权重的对照。传入独立 mixed 集的评分记录。
# def compare_fixed_sf_weights(rows):
#     labels = np.array([row["label"] for row in rows])
#     if len(set(labels)) != 2:
#         raise ValueError("Both labels are required")
#     results = []
#     for weights in ((1 / 3, 1 / 3, 1 / 3), (0.2, 0.6, 0.2), (0.05, 0.85, 0.1)):
#         scores = np.array([fuse(row["scores"], weights) for row in rows])
#         values = binary_metrics(labels, scores, 0.6)
#         results.append({"weights": weights, **values})
#     return results


def fit_alpha(calibration: list[dict], max_passes: int = 8) -> dict:
    if not calibration:
        raise ValueError("Alpha calibration needs equivalent pairs in the calibration split")
    trace_pairs = [traces(row) for row in calibration]
    # print("alpha calibration pairs:", len(trace_pairs))
    # print("trace lengths:", [(len(a), len(b)) for a, b in trace_pairs[:5]])
    alpha = dict.fromkeys(W_TAU, 1.0)

    def objective(values):
        weights = softmax_weights(values)
        highest = max(weights.values())
        similarities = [_trace_score(a, b, weighted_levenshtein(a, b, weights), highest) for a, b in trace_pairs]
        # print("alpha candidate:", values)
        # print("normalized weights:", weights, "mean:", np.mean(similarities), "variance:", np.var(similarities))
        return float(np.mean(similarities)), -float(np.var(similarities))

    for iteration in range(1, max_passes + 1):
        previous = alpha.copy()
        for kind in alpha:
            original_value = alpha[kind]
            best_key = None
            best_value = original_value
            for step in range(21):
                value = step / 10
                candidate = alpha.copy()
                candidate[kind] = value
                candidate_mean, negative_variance = objective(candidate)
                # print("coordinate:", iteration, kind, value, candidate_mean, -negative_variance)
                key = (candidate_mean, negative_variance, -abs(value - original_value))
                if best_key is None or key > best_key:
                    best_key = key
                    best_value = value
            alpha[kind] = best_value
            # print("权重搜索", iteration, kind, original_value, "->", best_value)
        if previous == alpha:
            # print("alpha unchanged at pass:", iteration)
            break
    mean, variance = objective(alpha)
    # print("校准均值", mean, "校准方差", -variance)
    return {"alpha": alpha, "mean_trace": mean, "variance": -variance, "passes": iteration,
            "converged": previous == alpha, "n": len(calibration)}


def fit_gate(mixed: list[dict]) -> dict:
    labels = np.array([row["label"] for row in mixed])
    if len(set(labels)) != 2:
        raise ValueError("Gate calibration needs both labels in the mixed split")
    base = [row["sitm_components"] for row in mixed]
    # print("gate calibration labels:", np.unique(labels, return_counts=True))
    # print("trace / call examples:", base[:3])
    best = None
    for hi, lo, gamma in product((.65, .70, .75), (.35, .40, .45), (.25, .35, .45)):
        scores = np.array([item["trace"] * (1 - apply_soft_gating(item["call"], theta_hi=hi, theta_lo=lo, gamma=gamma))
                           for item in base])
        # print("gate candidate:", hi, lo, gamma, "range:", scores.min(), scores.max())
        for threshold in (.45, .50, .55, .60, .65, .70, .75):
            result = binary_metrics(labels, scores, threshold)
            key = (result["f1"], -result["fpr"], -abs(threshold - .60),
                   -sum(abs(a-b) for a,b in zip((hi, lo, gamma), (THETA_HI, THETA_LO, GAMMA))))
            if best is None or key > best[0]:
                best = (key, dict(theta_hi=hi, theta_lo=lo, gamma=gamma, threshold=threshold, **result))
                # print("gate current best:", best[1])
    return {"n": len(mixed), **best[1]}


def ablation_scores(a: tuple, b: tuple, full) -> dict:
    call_store_log = ("call", "store", "log")
    layers = [([event for event in a if event[0] == key], [event for event in b if event[0] == key])
              for key in call_store_log]
    present = [(key, x, y) for key, (x, y) in zip(call_store_log, layers) if x or y]
    if present:
        independent = sum(W_TAU[key] * _edit_similarity(x, y) for key, x, y in present)
        independent /= sum(W_TAU[key] for key, _, _ in present)
    else:
        independent = 1.0
    return {
        "SITM": full.similarity,
        "w/o_interleave(3layer)": independent * (1 - full.penalty),
        "w/o_soft_gating": full.s_trace,
        "w/o_type_weight": compare_traces(a, b, weights=dict.fromkeys(W_TAU, 1.0)).similarity,
        "w/o_fine_fields": compare_traces(tuple(event[:2] for event in a),
                                          tuple(event[:2] for event in b)).similarity,
    }


# 只改事件字段粒度的对照，检查分数主要受哪一部分影响。
# def compare_event_fields(code_a, code_b):
#     left = tuple(extract_static_interleaved_trace(code_a))
#     right = tuple(extract_static_interleaved_trace(code_b))
#     opcode_left = tuple(event[:2] for event in left)
#     opcode_right = tuple(event[:2] for event in right)
#     type_left = tuple((event[0],) for event in left)
#     type_right = tuple((event[0],) for event in right)
#     return {
#         "full_fields": compare_traces(left, right).similarity,
#         "opcode_only": compare_traces(opcode_left, opcode_right).similarity,
#         "type_only": compare_traces(type_left, type_right).similarity,
#     }


def order_preservation(a: tuple, b: tuple) -> dict:
    """Whole-runtime projection equality, retained only as a layout diagnostic."""
    def project(events, ops):
        names = ["LOG" if event[0] == "log" else event[1] for event in events]
        return tuple(name for name in names if name in ops)

    out = {}
    for ops in ORDER_PAIRS:
        left, right = project(a, ops), project(b, ops)
        if set(left) == set(right) == set(ops):
            out["-".join(ops)] = left == right
    return out


def function_order_preservation(pair: dict) -> tuple[dict, dict]:
    """Count relative order of externally matched events within each function.

    A function_event_matches entry contains function_id, event_id, a_pc and
    b_pc. Identities must come from reviewed source maps/function recovery;
    opcode occurrence counts alone cannot establish cross-compiler identity.
    Partial mappings are reported with their coverage, never as a full table.
    """
    positioned = [extract_positioned_trace(strip_metadata(pair[side]), strip_meta=False)
                  for side in ("a", "b")]

    def event_name(event):
        if event[0] == "log":
            return "LOG"
        return "CALL" if event[0] == "call" else event[1]

    eligible_ops = {name for ops in ORDER_PAIRS for name in ops}
    events = [{pc: event_name(event) for pc, event in side
               if event_name(event) in eligible_ops} for side in positioned]
    matches = pair.get("function_event_matches")
    coverage = {"scope": "within_function_matched_event_pairs",
                "available": matches is not None, "matched_events": 0,
                "total_events_a": len(events[0]), "total_events_b": len(events[1]),
                "complete": False}
    if matches is None:
        coverage["reason"] = "function_event_matches not supplied"
        return {}, coverage
    if not isinstance(matches, list):
        raise ValueError(f"{pair['id']}: function_event_matches must be a list")

    functions = defaultdict(list)
    identities, used_a, used_b = set(), set(), set()
    for match in matches:
        if not isinstance(match, dict):
            raise ValueError(f"{pair['id']}: invalid function-event match")
        function, event_id = match.get("function_id"), match.get("event_id")
        pc_a, pc_b = match.get("a_pc"), match.get("b_pc")
        if (not isinstance(function, str) or not function.strip()
                or not isinstance(event_id, str) or not event_id.strip()
                or type(pc_a) is not int or type(pc_b) is not int):
            raise ValueError(f"{pair['id']}: match needs function_id, event_id and integer PCs")
        identity = function, event_id
        if identity in identities or pc_a in used_a or pc_b in used_b:
            raise ValueError(f"{pair['id']}: duplicate event identity or matched PC")
        op_a, op_b = events[0].get(pc_a), events[1].get(pc_b)
        if op_a is None or op_a != op_b:
            raise ValueError(f"{pair['id']}: matched PCs must refer to the same key event type")
        identities.add(identity)
        used_a.add(pc_a)
        used_b.add(pc_b)
        functions[function].append((op_a, pc_a, pc_b))

    counts = {}
    for left, right in ORDER_PAIRS:
        total = preserved = 0
        for matched in functions.values():
            lhs = [event for event in matched if event[0] == left]
            rhs = [event for event in matched if event[0] == right]
            for _, a_left, b_left in lhs:
                for _, a_right, b_right in rhs:
                    total += 1
                    preserved += (a_left < a_right) == (b_left < b_right)
        if total:
            counts[f"{left}-{right}"] = {"eligible": total, "preserved": preserved}
    coverage.update(matched_events=len(matches),
                    complete=used_a == set(events[0]) and used_b == set(events[1]))
    return counts, coverage


def external_scores(path_records: list[dict], pairs: list[dict]) -> dict:
    results = {}
    ids = {row["id"] for row in pairs}
    for row in path_records:
        identifier = row["id"]
        if identifier not in ids:
            raise ValueError(f"Unknown external score identifier: {identifier}")
        combined = results.setdefault(identifier, {"scores": {}, "clean_scores": {}, "model_details": {}})
        for field in ("scores", "clean_scores"):
            for name, value in row.get(field, {}).items():
                if name in combined[field]:
                    raise ValueError(f"Duplicate {field} for {identifier}/{name}")
                if name not in {"E2V", "SmartTrans"} or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"Invalid external score for {identifier}/{name}")
                combined[field][name] = value
        for name, details in row.get("model_details", {}).items():
            if name not in row.get("scores", {}) or not isinstance(details, dict):
                raise ValueError(f"Invalid model details for {identifier}/{name}")
            combined["model_details"][name] = details
    return results


def score_pairs(pairs: list[dict], external: dict) -> list[dict]:
    rows = []
    for number, pair in enumerate(pairs, 1):
        row = {key: value for key, value in pair.items() if key not in {"a", "b", "clean_b"}}
        # print("scoring:", number, row["id"], row["dataset"], row.get("label"))
        rows.append(row)
        if row["dataset"] == "llm" and row.get("status") == "failed":
            continue
        a, b = strip_metadata(pair["a"]), strip_metadata(pair["b"])
        # print("runtime lengths after metadata removal:", len(a), len(b))
        # print("same normalized bytecode:", a == b)
        t_a, t_b = trace(a), trace(b)
        # print("SIT lengths:", len(t_a), len(t_b))
        # print("left events:", t_a[:5])
        # print("right events:", t_b[:5])
        result = compare_traces(t_a, t_b)
        imported = external.get(row["id"], {})
        # print("external score fields:", row["id"], sorted(imported.get("scores", {})))
        row["scores"] = structural_scores(pair["a"], pair["b"]).copy()
        row["scores"]["SITM"] = result.similarity
        row["scores"].update(imported.get("scores", {}))
        # print("pair scores:", row["id"], row["scores"])
        # assert all(np.isfinite(value) for value in row["scores"].values())
        if imported.get("model_details"):
            row["model_details"] = imported["model_details"]
        row["sitm_components"] = {"trace": result.s_trace, "call": result.s_call, "penalty": result.penalty}
        # print("SITM components:", row["sitm_components"])
        if row["dataset"] in {"opt", "ver", "mut"}:
            row["ablations"] = ablation_scores(t_a, t_b, result)
        if row["dataset"] in {"opt", "ver"}:
            # Old: row["order"] = order_preservation(t_a, t_b)
            # Whole-runtime sequence equality is not the paper's function-pair rate.
            row["runtime_order"] = order_preservation(t_a, t_b)
            row["order"], row["order_coverage"] = function_order_preservation(pair)
        if row["dataset"] in {"mut", "mixed"} and row.get("label") == 0:
            row["layers"] = {}
            for kind in W_TAU:
                left = [event for event in t_a if event[0] == kind]
                right = [event for event in t_b if event[0] == kind]
                row["layers"][kind] = _edit_similarity(left, right)
                # print("layer:", row["id"], kind, len(left), len(right), row["layers"][kind])
        if row["dataset"] == "noise":
            clean = compare_traces(t_a, trace(strip_metadata(pair["clean_b"])))
            row["clean_scores"] = {**structural_scores(pair["a"], pair["clean_b"]), "SITM": clean.similarity,
                                   **imported.get("clean_scores", {})}
            # print("noise SITM clean / noisy:", clean.similarity, result.similarity)
        if number % 100 == 0:
            print(f"Scored {number}/{len(pairs)} pairs", flush=True)
    return rows


def distribution(values: list, threshold: float) -> dict | None:
    if not values:
        return None
    array = np.array(values)
    p5, p95 = np.percentile(array, [5, 95])
    return dict(n=len(values), mean=float(array.mean()), median=float(np.median(array)),
                std=float(array.std()), p5=float(p5), p95=float(p95),
                mean_dev=float((1 - array).mean()), max_dev=float((1 - array).max()),
                accepted=float((array >= threshold).mean()))


def metric_distributions(rows: list[dict], threshold: float) -> dict:
    results = {}
    for method in METHODS:
        values = [row["scores"][method] for row in rows if method in row.get("scores", {})]
        if values:
            results[method] = distribution(values, threshold)
    return results


def cliff_delta(a: np.ndarray, b: np.ndarray) -> float:
    ordered = np.sort(b)
    less = np.searchsorted(ordered, a, side="left").sum()
    greater = len(a) * len(b) - np.searchsorted(ordered, a, side="right").sum()
    return float((less - greater) / (len(a) * len(b)))


def paired_tests(rows: list[dict]) -> dict:
    out = {}
    for method in METHODS[:-1]:
        common = [row["scores"] for row in rows if method in row.get("scores", {})]
        if not common:
            continue
        a = np.array([row["SITM"] for row in common])
        b = np.array([row[method] for row in common])
        p_value = stats.wilcoxon(a - b).pvalue if np.any(a != b) else 1.0
        out[method] = {"n": len(common), "wilcoxon_p": float(p_value), "cliff_delta": cliff_delta(a, b)}
    return out


def binary_metrics(labels, scores, threshold: float) -> dict:
    predicted = scores >= threshold
    positive = labels == 1
    tp = np.count_nonzero(predicted & positive)
    fp = np.count_nonzero(predicted & ~positive)
    fn = np.count_nonzero(~predicted & positive)
    tn = np.count_nonzero(~predicted & ~positive)
    # print("threshold:", threshold, "TP FP FN TN:", tp, fp, fn, tn)
    # print("scores equal to threshold:", np.count_nonzero(scores == threshold))
    # assert tp + fp + fn + tn == len(labels)
    precision, recall, _ = precision_recall_curve(labels, scores)
    # print("PR points:", len(precision), len(recall))
    return {"roc_auc": float(roc_auc_score(labels, scores)), "pr_auc": float(auc(recall, precision)),
            "precision": float(tp / max(tp+fp, 1)), "recall": float(tp / max(tp+fn, 1)),
            "f1": float(2*tp / max(2*tp+fp+fn, 1)), "fpr": float(fp / max(fp+tn, 1))}


# 单独核对阈值边界。0.6 在当前规则下会被判成等价，0.5999 会被判成变异。
# labels = np.array([1, 1, 0, 0])
# scores = np.array([0.9, 0.6, 0.5999, 0.8])
# print(scores >= 0.6)
# print(binary_metrics(labels, scores, 0.6))
# print(binary_metrics(labels, scores, 0.8))


# 手动核对阈值统计的展开写法，先看 TP、FP、FN、TN 四个数。
# def threshold_counts(labels, scores, threshold=0.6):
#     if len(labels) != len(scores):
#         raise ValueError("Labels and scores must have the same length")
#     tp = fp = fn = tn = 0
#     for label, score in zip(labels, scores):
#         equivalent = score >= threshold
#         if label == 1:
#             if equivalent:
#                 tp += 1
#             else:
#                 fn += 1
#         else:
#             if equivalent:
#                 fp += 1
#             else:
#                 tn += 1
#     return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
#
# # case_labels = [1, 1, 0, 0]
# # case_scores = [0.9, 0.6, 0.5999, 0.8]
# # print(threshold_counts(case_labels, case_scores))


def bootstrap_strata(rows: list[dict], labels: np.ndarray) -> tuple[list, dict]:
    """Group whole contracts, stratified by each cluster's class counts.

    Resampling within (negative, positive) count strata preserves the observed
    class totals, including the one-positive/two-negative Temporal design.
    """
    contracts = defaultdict(list)
    missing_identity = False
    for index, row in enumerate(rows):
        identifier = row.get("contract_id")
        if isinstance(identifier, str) and identifier.strip():
            key = ("contract", identifier.strip().lower())
            # A content-derived legacy key can join exact duplicates, but it
            # cannot certify that compiler variants share a deployed address.
            missing_identity |= bool(row.get("contract_id_source"))
        else:
            key = ("row", str(index))
            missing_identity = True
        contracts[key].append(index)
    profiles = defaultdict(list)
    for key in sorted(contracts):
        indices = np.array(contracts[key], dtype=int)
        profile = tuple(int(np.count_nonzero(labels[indices] == label)) for label in (0, 1))
        profiles[profile].append(indices)
    repeated = any(len(indices) > 1 for group in profiles.values() for indices in group)
    return [profiles[key] for key in sorted(profiles)], {
        "bootstrap_unit": "contract" if repeated else "pair",
        "bootstrap_stratification": "contract_label_counts" if repeated else "label",
        "n_contracts": len(contracts),
        "bootstrap_singleton_strata": sum(len(group) == 1 for group in profiles.values()),
        "bootstrap_identity_complete": not missing_identity,
    }


def draw_bootstrap_indices(strata: list, rng: np.random.Generator) -> np.ndarray:
    return np.concatenate([cluster for group in strata
                           for cluster in (group[index] for index in rng.integers(len(group), size=len(group)))])


def classification(rows: list[dict], threshold: float, bootstrap: int, seed: int) -> dict:
    out = {}
    for method in METHODS:
        usable = [row for row in rows if method in row["scores"]]
        labels = np.array([row["label"] for row in usable], dtype=int)
        scores = np.array([row["scores"][method] for row in usable])
        # print("classification method:", method, "N:", len(usable))
        # print("label counts:", np.unique(labels, return_counts=True))
        if len(set(labels)) != 2:
            continue
        values = binary_metrics(labels, scores, threshold)
        # print("classification metrics:", method, values)
        out[method] = {"n": len(usable), **values}
        if bootstrap:
            rng = np.random.default_rng(seed)
            strata, metadata = bootstrap_strata(usable, labels)
            out[method].update(metadata, bootstrap_resamples=bootstrap)
            samples = []
            for _ in range(bootstrap):
                # Old: independently resample positive and negative pairs.
                # Keep correlated observations from the same contract together.
                ids = draw_bootstrap_indices(strata, rng)
                metric = binary_metrics(labels[ids], scores[ids], threshold)
                samples.append(list(metric.values()))
            intervals = np.percentile(samples, [2.5, 97.5], axis=0).T.tolist()
            # print("CI metric order:", list(values))
            # print("bootstrap CI95:", method, intervals)
            out[method]["bootstrap_ci95"] = dict(zip(values, intervals))
    return out


# 用同一小批数据看 bootstrap 次数对区间的影响，种子固定后再比较。
# def compare_bootstrap_sizes(rows):
#     reports = {}
#     for count in (100, 500, 1000):
#         report = classification(rows, threshold=0.6, bootstrap=count, seed=42)
#         reports[count] = {}
#         for method, values in report.items():
#             reports[count][method] = {
#                 "n": values["n"],
#                 "roc_auc": values["roc_auc"],
#                 "ci95": values.get("bootstrap_ci95"),
#             }
#     return reports


def mutation_breakdown(rows: list[dict], threshold: float) -> dict:
    result = {}
    for kind in MUTATIONS:
        group = [row for row in rows if row.get("label") == 0 and row.get("mutation") == kind]
        # print("mutation group:", kind, "negative pairs:", len(group))
        if group:
            layers = {}
            for event_type in W_TAU:
                values = [row["layers"][event_type] for row in group if "layers" in row]
                if values:
                    layers[event_type] = float(np.mean(values))
                    # print("layer mean:", kind, event_type, "N:", len(values), layers[event_type])
            result[kind] = {
                "scores": metric_distributions(group, threshold),
                "sitm_detection_rate": float(np.mean([row["scores"]["SITM"] < threshold for row in group])),
                "layers": layers,
            }
            # detected = sum(row["scores"]["SITM"] < threshold for row in group)
            # print("detected / total:", kind, detected, len(group), "threshold:", threshold)
            # for row in group:
            #     if row["scores"]["SITM"] >= threshold:
            #         print("missed negative:", row["id"], row["scores"]["SITM"])
    return result


def grouped(rows: list[dict], key: str, default="unclassified") -> dict:
    out = defaultdict(list)
    for row in rows:
        out[row.get(key, default)].append(row)
    return dict(out)


def order_summary(rows: list[dict]) -> dict:
    result = {}
    coverage = [row.get("order_coverage", {}) for row in rows]
    for left, right in ORDER_PAIRS:
        key = f"{left}-{right}"
        # Old: average one whole-runtime equality flag per contract. The correct
        # denominator is the number of matched within-function event pairs.
        counts = [row["order"][key] for row in rows if key in row.get("order", {})]
        total = sum(count["eligible"] for count in counts)
        preserved = sum(count["preserved"] for count in counts)
        result[key] = {"n": total, "preserved": preserved / total if total else None,
                       "scope": "within_function_matched_event_pairs", "total_pairs": len(rows),
                       "annotated_pairs": sum(bool(item.get("available")) for item in coverage),
                       "complete": bool(rows) and all(item.get("complete", False) for item in coverage),
                       "matched_events": sum(item.get("matched_events", 0) for item in coverage),
                       "total_events_a": sum(item.get("total_events_a", 0) for item in coverage),
                       "total_events_b": sum(item.get("total_events_b", 0) for item in coverage)}
    return result


def runtime_order_summary(rows: list[dict]) -> dict:
    """Retain the old diagnostic without presenting it as function-pair data."""
    result = {}
    for left, right in ORDER_PAIRS:
        key = f"{left}-{right}"
        flags = [row["runtime_order"][key] for row in rows if key in row.get("runtime_order", {})]
        result[key] = {"scope": "whole_runtime_sequence_equality", "n": len(flags),
                       "preserved": sum(flags) / len(flags) if flags else None}
    return result


def llm_report(rows: list[dict], threshold: float) -> dict:
    models = grouped(rows, "model")
    success = {}
    by_model = {}
    for model, attempts in models.items():
        compiled = {}
        for row in attempts:
            if "scores" in row:
                compiled[row["contract_id"]] = row
        success[model] = compiled
        by_model[model] = {
            "attempts": len(attempts),
            "compiled": len(compiled),
            "failure_rate": 1 - len(compiled) / len(attempts),
            "scores": metric_distributions(list(compiled.values()), threshold),
        }

    balanced = []
    if models:
        common = set.intersection(*(set(group) for group in success.values()))
        balanced = sorted(common)
    report = {"models": by_model, "balanced_contracts": balanced}
    if len(models) < 3 or len(balanced) < 2:
        return report

    scores = []
    for contract in balanced:
        scores.append([success[model][contract]["scores"]["SITM"] for model in models])
    matrix = np.array(scores)
    ranks = stats.rankdata(-matrix, axis=1).mean(axis=0)
    n, k = matrix.shape
    std = math.sqrt(k * (k + 1) / (6 * n))
    statistic, p_value = 0.0, 1.0
    if np.any(np.ptp(matrix, axis=1)):
        statistic, p_value = stats.friedmanchisquare(*matrix.T)

    probabilities = {}
    for i, left in enumerate(models):
        probabilities[left] = {}
        for j, right in enumerate(models):
            difference = abs(ranks[i] - ranks[j]) * math.sqrt(2) / std
            probabilities[left][right] = float(stats.studentized_range.sf(difference, k, np.inf))
    report["friedman"] = {"statistic": float(statistic), "p": float(p_value), "df": k - 1}
    report["mean_rank"] = dict(zip(models, ranks.tolist()))
    report["nemenyi_cd_05"] = float(stats.studentized_range.ppf(.95, k, np.inf) * std / math.sqrt(2))
    report["nemenyi_p"] = probabilities
    return report


def ablation_report(opt, ver, mutations, threshold: float) -> dict:
    out = {}
    for variant in ABLATIONS:
        opt_scores = [row["ablations"][variant] for row in opt]
        ver_scores = [row["ablations"][variant] for row in ver]
        labels = [row["label"] for row in mutations]
        record = {"opt_mean_dev": 1 - float(np.mean(opt_scores)) if opt_scores else None,
                  "ver_mean_dev": 1 - float(np.mean(ver_scores)) if ver_scores else None,
                  "roc_auc": float(roc_auc_score(labels, [row["ablations"][variant] for row in mutations]))
                  if len(set(labels)) == 2 else None}
        for kind in ("CH", "RS"):
            negative = [row["ablations"][variant] for row in mutations if row.get("mutation") == kind and row["label"] == 0]
            record[kind + "_detect"] = float(np.mean(np.array(negative) < threshold)) if negative else None
        out[variant] = record
    return out


def sensitivity_report(pairs: list[dict], rows: list[dict]) -> dict:
    mixed = [row for row in pairs if row["dataset"] == "mixed"]
    labels = np.array([row["label"] for row in mixed], dtype=int)
    if len(set(labels)) != 2:
        return {}
    baseline = {row["id"]: row["sitm_components"] for row in rows if row["dataset"] == "mixed"}
    original = [baseline[row["id"]] for row in mixed]
    scores = [item["trace"] * (1 - item["penalty"]) for item in original]
    output = {"threshold": [{"value": i / 100, **binary_metrics(labels, np.array(scores), i / 100)}
                            for i in range(45, 76, 5)]}
    pair_traces = [traces(pair) for pair in mixed]
    for variable, values in {"alpha_store": range(40, 81, 10), "theta_hi": range(65, 76, 5),
                             "gamma": range(25, 46, 5)}.items():
        curve = []
        for i in values:
            value = i / 100
            if variable == "alpha_store":
                weights = softmax_weights({**ALPHA_WEIGHTS, "store": value})
                scores = [compare_traces(a, b, weights=weights).similarity for a, b in pair_traces]
            else:
                gate = {variable: value}
                scores = [item["trace"] * (1 - apply_soft_gating(item["call"], **gate)) for item in original]
            ch = [score for score, row in zip(scores, mixed) if row.get("mutation") == "CH" and row["label"] == 0]
            curve.append({"value": value, "roc_auc": float(roc_auc_score(labels, scores)),
                          "ch_detect": float(np.mean(np.array(ch) < .60)) if ch else None})
        output[variable] = curve
    return output


def summarize(pairs: list[dict], rows: list[dict], threshold: float, bootstrap: int, seed: int) -> dict:
    datasets = grouped(rows, "dataset")
    opt = datasets.get("opt", [])
    ver = datasets.get("ver", [])
    mut = datasets.get("mut", [])
    temporal = datasets.get("temporal", [])

    version_groups = {}
    for group, records in grouped(ver, "group").items():
        version_groups[group] = metric_distributions(records, threshold)
    version_scores = {}
    for version, records in grouped(ver, "version").items():
        version_scores[version] = metric_distributions(records, threshold)
    order = {"opt": order_summary(opt)}
    runtime_order = {"opt": runtime_order_summary(opt)}
    for group, records in grouped(ver, "group").items():
        order[group] = order_summary(records)
        runtime_order[group] = runtime_order_summary(records)

    thresholds = {}
    for level in (0.50, 0.60, 0.70, 0.80, 0.85):
        thresholds[str(level)] = classification(mut, level, 0, seed)
    traceability = {}
    for group, records in grouped(mut, "traceability").items():
        traceability[group] = mutation_breakdown(records, threshold)
    difficulty = {}
    for group, records in grouped(mut, "difficulty").items():
        difficulty[group] = mutation_breakdown(records, threshold)

    noise = {}
    for kind, records in grouped(datasets.get("noise", []), "noise").items():
        noise[kind] = {}
        for method in METHODS:
            changes = []
            for row in records:
                if method in row["clean_scores"] and method in row["scores"]:
                    changes.append(row["clean_scores"][method] - row["scores"][method])
            if changes:
                noise[kind][method] = {"n": len(changes), "mean_delta": float(np.mean(changes))}

    return {
        "counts": dict(Counter(row["dataset"] for row in rows)),
        "threshold": threshold,
        "bootstrap": bootstrap,
        "seed": seed,
        "sitm_parameters": {"alpha": ALPHA_WEIGHTS, "theta_hi": THETA_HI, "theta_lo": THETA_LO, "gamma": GAMMA},
        "rq1_opt": {
            "scores": metric_distributions(opt, threshold),
            "statistics": paired_tests(opt),
        },
        "rq2_ver": {
            "scores": metric_distributions(ver, threshold),
            "statistics": paired_tests(ver),
            "groups": version_groups,
            "by_version": version_scores,
            "order": order,
            "runtime_order_diagnostic": runtime_order,
        },
        "rq3_mut": {
            "classification": classification(mut, threshold, bootstrap, seed),
            "thresholds": thresholds,
            "by_operator": mutation_breakdown(mut, threshold),
            "by_traceability": traceability,
            "by_difficulty": difficulty,
            "temporal": {
                "classification": classification(temporal, threshold, bootstrap, seed),
                "optimizer_equivalence": metric_distributions(
                    [row for row in temporal if row["label"] == 1], threshold),
            },
        },
        "rq4_noise": noise,
        "rq5_llm": llm_report(datasets.get("llm", []), threshold),
        "rq6_ablation": {
            "variants": ablation_report(opt, ver, mut, threshold),
            "sensitivity": sensitivity_report(pairs, rows),
        },
    }
