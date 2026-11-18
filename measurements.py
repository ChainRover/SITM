from random import Random
from statistics import mean, median, pstdev
from time import perf_counter_ns
from numpy import percentile
from scipy.stats import spearmanr

from evm_sit import bytecode_bytes
from analysis import METHODS
from tables import save_table
from util import (calc_cfg_isomorphism_similarity_bc, calc_ngram_similarity_bc,
                  calc_opcode_edit_similarity_bc, calculate_sitm_new_similarity_bc,
                  extract_static_interleaved_trace)


def measure_cost(pairs, count, warmup, seed, out, details=False):
    pairs = sorted(pairs, key=lambda row: row["id"])
    chosen = Random(seed).sample(pairs, min(count, len(pairs)))
    decoded = [(row["id"], bytecode_bytes(row["a"]), bytecode_bytes(row["b"])) for row in chosen]
    methods = {"OED": calc_opcode_edit_similarity_bc, "CFG": calc_cfg_isomorphism_similarity_bc,
               "NG": calc_ngram_similarity_bc, "SITM": calculate_sitm_new_similarity_bc}
    raw, summary = [], []
    for name, score in methods.items():
        for _, a, b in decoded[:warmup]:
            score(a, b)
        elapsed = []
        for identifier, a, b in decoded:
            start = perf_counter_ns()
            score(a, b)
            ms = (perf_counter_ns() - start) / 1_000_000
            elapsed.append(ms)
            raw.append({"method": name, "id": identifier, "time_ms": ms})
        if elapsed:
            # 小样本计时先看前几笔，排查首次调用的耗时是否特别高。
            # print(name, elapsed[:5])
            summary.append({"method": name, "n": len(elapsed), "mean_ms": mean(elapsed),
                            "median_ms": median(elapsed), "std_ms": pstdev(elapsed),
                            "p95_ms": float(percentile(elapsed, 95))})
    save_table(out / "cost_summary.csv", summary)
    if details:
        save_table(out / "raw_cost.csv", raw)
    return summary


def field_traceability(pairs, out, details=False):
    fields = {"call": {"target": 2, "selector": 3, "input_length": 4, "value": 5},
              "store": {"base_slot": 2, "pattern": 3}, "log": {"topic0": 3, "data_length": 4},
              "exc": {"opcode": 1, "condition": 2}, "ret": {"return_length": 2}}
    counters = {}
    health = []
    for pair_row in pairs:
        events = extract_static_interleaved_trace(pair_row["a"])
        health.append({"id": pair_row["id"], "events": len(events)})
        found = set()
        for event in events:
            kind = event[0]
            for name, index in fields[kind].items():
                value = event[index]
                if value == "N/A":
                    continue
                key = f"{kind}.{name}"
                known = not str(value).startswith("[DYN")
                counter = counters.setdefault(key, {"total_events": 0, "known_events": 0, "pairs_with_known": 0})
                counter["total_events"] += 1
                counter["known_events"] += int(known)
                if known:
                    found.add(key)
        for key in found:
            counters[key]["pairs_with_known"] += 1
    records = []
    for key, counter in counters.items():
        records.append({"field": key, **counter, "total_pairs": len(pairs),
                        "event_rate": counter["known_events"] / counter["total_events"],
                        "pair_rate": counter["pairs_with_known"] / len(pairs)})
    save_table(out / "field_traceability.csv", records)
    if details:
        save_table(out / "trace_health.csv", health)
    return records


def noise_correlations(rows):
    result = []
    for kind in sorted({row["noise"] for row in rows}):
        for method in METHODS:
            clean, noisy = [], []
            for row in rows:
                if row["noise"] == kind and method in row["clean_scores"] and method in row["scores"]:
                    clean.append(row["clean_scores"][method])
                    noisy.append(row["scores"][method])
            if clean:
                rho = None
                if len(set(clean)) > 1 and len(set(noisy)) > 1:
                    rho = float(spearmanr(clean, noisy).statistic)
                result.append({"noise": kind, "method": method, "n": len(clean), "spearman": rho})
    return result


