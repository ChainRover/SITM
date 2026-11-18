from data import (build, validate_pairs, pair, prepare_compiler_pairs, prepare_llm,
                  prepare_mutants, standard_input, with_noise, write_jsonl)
from mutations import discover_recipes
from analysis import ablation_scores
from util import compare_traces, extract_static_interleaved_trace


SMOKE_CONTRACT = '''pragma solidity >=0.5.0 <0.9.0;
contract Probe {
    function change() external {
        assembly {
            mstore(0, 0x12345678)
            sstore(0, 9)
            pop(call(gas(), 17, 0, 28, 4, 0, 0))
            pop(call(gas(), 18, 0, 28, 4, 0, 0))
            log1(0, 0, 98)
        }
    }
}
'''


def mechanism_checks(threshold: float) -> list[dict]:
    """Score hand-written runtime examples separately from benchmark pairs."""
    # PUSH1 9; PUSH1 slot; SSTORE.
    stores = [f"600960{slot:02x}55" for slot in range(8)]
    # CALL with no value or calldata, followed by POP to discard the result.
    calls = [f"5f5f5f5f5f60{address:02x}5af150" for address in range(64, 68)]
    redirected = [f"5f5f5f5f5f60{address:02x}5af150" for address in range(80, 84)]
    cases = [
        ("ch_one_target", stores + calls, stores + redirected[:1] + calls[1:]),
        ("ch_all_targets", stores + calls, stores + redirected),
        ("rs_one_swap", stores[:3] + calls[:1], stores[:2] + calls[:1] + stores[2:3]),
        ("rs_move_three_writes", stores[:3] + calls[:1], calls[:1] + stores[:3]),
    ]
    records = []
    for name, original, mutated in cases:
        a = "".join(original) + "00"
        b = "".join(mutated) + "00"
        trace_a = tuple(extract_static_interleaved_trace(a))
        trace_b = tuple(extract_static_interleaved_trace(b))
        result = compare_traces(trace_a, trace_b)
        # print("mechanism case:", name, "trace lengths:", len(trace_a), len(trace_b))
        # print("mechanism SITM:", result.similarity, "detected:", result.similarity < threshold)
        ablations = ablation_scores(trace_a, trace_b, result)
        records.append({"id": name, "sample_origin": "synthetic_mechanism_check", "a": a, "b": b,
                        "trace_a": trace_a, "trace_b": trace_b, "threshold": threshold,
                        "s_trace": result.s_trace, "s_call": result.s_call, "penalty": result.penalty,
                        "sitm": result.similarity, "detected": result.similarity < threshold,
                        "without_interleaving": ablations["w/o_interleave(3layer)"]})
    return records


def make_smoke_pairs(args):
    root = args.fixture_dir
    root.mkdir(parents=True, exist_ok=True)
    (root / "Probe.sol").write_text(SMOKE_CONTRACT, encoding="utf-8")
    spec = {"id": "probe", "target": "Probe.sol:Probe", "sources": {"Probe.sol": "Probe.sol"},
            "versions": [args.legacy_version, "0.8.20"], "version_groups": {"0.8.20": "major"}}
    mixed_spec = {**spec, "id": "mixed-probe", "split": "mixed", "versions": ["0.8.20"]}
    cal_spec = {**spec, "id": "cal-probe", "split": "calibration", "versions": ["0.8.20"]}
    compiled, errors = prepare_compiler_pairs([spec, mixed_spec, cal_spec], root, args.solc_dir)
    recipes, missing = discover_recipes([spec, mixed_spec], root, "0.8.20", args.solc_dir)
    mutation = getattr(args, "smoke_mutation", "CH")
    recipes = [row for row in recipes if row["dataset"] == "mixed" or row["mutation"] == mutation]
    negative, rejected = prepare_mutants(recipes, root, args.solc_dir)
    # print("smoke mutation selection:", mutation, "recipes:", len(recipes), "rejected:", len(rejected))
    # print("negative groups:", [(row["dataset"], row["mutation"]) for row in negative])
    if errors or missing or rejected:
        raise ValueError(f"Smoke build failures: {errors + missing + rejected}")
    if len(negative) != 6:
        raise ValueError(f"Expected one Mut-Set negative and five mixed negatives; got {len(negative)} pairs")
    mut_negative = next(row for row in negative if row["dataset"] == "mut")
    opt = next(row for row in compiled if row["dataset"] == "opt")
    latest_a = build(standard_input(spec, root, "0.8.20", False), spec["target"], "0.8.20", args.solc_dir).bytecode
    positives = [pair("mut", opt["contract_id"], "optimizer-positive", opt["a"], opt["b"], label=1,
                      version=opt["version"], positive_origin="smoke_optimizer_pair")]
    for index in range(1, 3):
        filename = f"ProbePositive{index}.sol"
        source = SMOKE_CONTRACT.replace("sstore(0, 9)", f"sstore(0, {9 + index})")
        (root / filename).write_text(source, encoding="utf-8")
        positive_spec = {"id": f"positive-probe-{index}", "target": f"{filename}:Probe",
                         "sources": {filename: filename}, "versions": [opt["version"]]}
        equivalent, failures = prepare_compiler_pairs([positive_spec], root, args.solc_dir)
        if failures or len(equivalent) != 1:
            raise ValueError(f"Smoke positive {index} build failed: {failures}")
        row = equivalent[0]
        positives.append(pair("mut", row["contract_id"], "optimizer-positive", row["a"], row["b"],
                              label=1, version=row["version"], positive_origin="smoke_optimizer_pair"))
    temporal = [pair("temporal", "temporal-fixture", "positive", opt["a"], opt["b"], label=1),
                pair("temporal", "temporal-fixture", "negative", latest_a, mut_negative["b"], label=0,
                     mutation=mut_negative["mutation"])]
    llm = []
    if args.mock_llm:
        llm = prepare_llm([{"id": identifier, "reference": latest_a, "attempts": {
            "fixture-a": {"runtime": latest_a}, "fixture-b": {"runtime": opt["a"]},
            "fixture-c": {"runtime": mut_negative["b"]}}} for identifier in ("llm-fixture-1", "llm-fixture-2")]
            + [{"id": "llm-fixture-3", "reference": latest_a, "attempts": {
                "fixture-a": {"runtime": latest_a}, "fixture-b": {"runtime": opt["a"]},
                "fixture-c": {"error": "fixture compilation failure"}}}], root, args.solc_dir)
    pairs = compiled + negative + positives + temporal + llm
    # 只核对 Mut-Set 的样本数，mixed 中的负样本另算。
    # mut_rows = [row for row in pairs if row["dataset"] == "mut"]
    # print("Mut 正样本", sum(row["label"] == 1 for row in mut_rows))
    # print("Mut 负样本", [row["mutation"] for row in mut_rows if row["label"] == 0])
    pairs = with_noise(pairs, 1, args.seed)
    for row in pairs:
        row["sample_origin"] = "smoke"
    validate_pairs(pairs)
    # assert sum(row["dataset"] == "mut" and row["label"] == 1 for row in pairs) == 3
    # assert sum(row["dataset"] == "mut" and row["label"] == 0 for row in pairs) == 1
    write_jsonl(root / "pairs.jsonl", pairs)
    write_jsonl(root / "mutations.jsonl", recipes)
    for filename in ("ProbePositive3.sol", "ProbePositive4.sol"):
        (root / filename).unlink(missing_ok=True)
    return pairs


