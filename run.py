import argparse
import math
from pathlib import Path
# from evm_cfg_builder.cfg import CFG
from random import Random
from data import (LEGACY_FILES, MUTATIONS, pair, read_pairs, validate_pairs, with_noise)
from experiments import evaluate
from smoke import make_smoke_pairs


CORE_DATASETS = ("opt", "ver", "mut", "noise", "calibration", "mixed", "temporal")
EXPERIMENTS = {
    "1": ("opt",), "2": ("ver",), "3": ("mut", "temporal"), "4": ("noise",),
    "5": ("opt", "ver", "mut"), "6": ("opt",), "7": ("mixed",),
    "8": ("calibration", "mixed"), "9": ("opt",), "10": ("opt",),
    "11": ("llm",), "12": ("llm",),
    "core": CORE_DATASETS,
}


def input_paths(args):
    sources = {path: None for path in args.pairs}
    wanted = set(EXPERIMENTS.get(args.exp, ()))
    if not args.pairs or args.dataset_dir is not None:
        root = args.dataset_dir or Path("dataset/smartbugs")
        # print("dataset directory:", root.resolve(), "experiment:", args.exp)
        wanted.update({"mixed"})  # Needed for SF fitting when available.
        if args.exp in {"3", "4", "5"}:
            wanted.add("opt")
        if args.calibrate:
            wanted.add("calibration")
        for name, dataset in LEGACY_FILES.items():
            path = root / name
            eligible = args.exp == "all" or dataset in wanted or (dataset == "calibration" and "mixed" in wanted)
            if path.is_file() and eligible:
                # print("selected input:", path.name, "dataset:", dataset)
                sources.setdefault(path, dataset)
        if args.exp in {"all", "11", "12"} and not args.llm_file:
            llm_dir = args.llm_dir or root.parent / "llm"
            files = sorted(llm_dir.glob("sitm_downstream_*_pairs.jsonl")) if args.exp != "12" else []
            flat = root.parent / "llm_common" / "common_addresses_flat.jsonl"
            if not files and flat.is_file():
                files = [flat]
            for path in files:
                sources.setdefault(path, "llm")
    elif args.llm_dir is not None and args.exp != "core":
        for path in sorted(args.llm_dir.glob("sitm_downstream_*_pairs.jsonl")):
            sources.setdefault(path, "llm")
    if args.exp != "core":
        for path in args.llm_file:
            sources.setdefault(path, "llm")
    for path in args.positive_file:
        sources[path] = "mut"
    return sources


def add_mutation_positives(pairs, count, seed):
    negative = [row for row in pairs if row["dataset"] == "mut" and row["label"] == 0]
    if not negative or any(row["dataset"] == "mut" and row["label"] == 1 for row in pairs):
        return []
    pool = sorted((row for row in pairs if row["dataset"] == "opt"), key=lambda row: row["id"])
    picked = Random(seed).sample(pool, min(count, len(pool)))
    # print("positive sampling:", "requested:", count, "pool:", len(pool), "picked:", len(picked), "seed:", seed)
    # print("positive source IDs:", [row["id"] for row in picked[:5]])
    return [pair("mut", row["contract_id"], "positive:" + row["id"], row["a"], row["b"], label=1,
                 positive_origin="sampled_opt_pair", source_pair_id=row["id"],
                 sample_origin=row.get("sample_origin", "provided_manifests")) for row in picked]


def run(args):
    if args.exp == "smoke":
        args.fixture_dir = args.out / "fixtures"
        pairs = make_smoke_pairs(args)
        sources = [args.fixture_dir / "pairs.jsonl"]
        args.calibrate = True
    else:
        inputs = input_paths(args)
        # 调整数据目录时，先打开下面两行核对实际读入的文件。
        # for path, dataset in inputs.items():
        #     print("读取文件", path, "分组", dataset)
        pairs = []
        for path, dataset in inputs.items():
            # before = len(pairs)
            pairs.extend(read_pairs(path, dataset))
            # print("added pairs:", path.name, len(pairs) - before, "running total:", len(pairs))
        sources = list(inputs)

    if args.skip_llm or args.exp == "core":
        pairs = [row for row in pairs if row["dataset"] != "llm"]
    args.generated_positives = 0
    if args.positive_count and args.exp in {"all", "core", "3", "5"}:
        positives = add_mutation_positives(pairs, args.positive_count, args.seed)
        pairs.extend(positives)
        args.generated_positives = len(positives)
    if args.noise_count and args.exp in {"all", "core", "4"}:
        pairs = with_noise(pairs, args.noise_count, args.seed)

    if args.exp in EXPERIMENTS and args.exp != "core":
        keep = set(EXPERIMENTS[args.exp])
        keep.add("mixed")
        if args.calibrate:
            keep.add("calibration")
        pairs = [row for row in pairs if row["dataset"] in keep]
        required = {"3": {"mut"}, "8": set()}.get(args.exp, set(EXPERIMENTS[args.exp]))
        available = {row["dataset"] for row in pairs}
        # print("required datasets:", sorted(required), "available:", sorted(available))
        if required - available:
            raise ValueError(f"Experiment {args.exp} needs: {', '.join(sorted(required - available))}")
    if not pairs:
        raise ValueError("No pair data found. Pass JSONL files or set --dataset-dir.")
    validate_pairs(pairs)
    # print("evaluation inputs:", len(pairs), "sources:", [path.name for path in sources])
    # print("mutation labels:", [row.get("label") for row in pairs if row["dataset"] == "mut"][:20])
    result = evaluate(pairs, args, sources)
    if args.exp == "smoke" and args.details:
        from smoke import mechanism_checks
        from data import write_jsonl
        write_jsonl(args.out / "mechanism_checks.jsonl", mechanism_checks(args.threshold))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", nargs="*", type=Path, help="Normalized manifests or named legacy datasets")
    parser.add_argument("--exp", choices=("all", "smoke", *EXPERIMENTS), default="all")
    parser.add_argument("--dataset-dir", "--dataset_dir", type=Path)
    parser.add_argument("--out", "--out_dir", type=Path)
    parser.add_argument("--positive-file", type=Path, action="append", default=[])
    parser.add_argument("--llm-dir", type=Path)
    parser.add_argument("--llm-file", type=Path, action="append", default=[])
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--scores", type=Path, action="append", default=[])
    parser.add_argument("--replays", type=Path, action="append", default=[])
    parser.add_argument("--sf-weights", nargs=3, type=float)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int)
    parser.add_argument("--calibrate", action="store_true", help="Search alpha and gate settings on the validation splits")
    parser.add_argument("--positive-count", type=int, default=0, help="Explicitly sample missing Mut positives from Opt-Pair")
    parser.add_argument("--noise-count", type=int, default=0, help="Explicitly generate suffix-noise examples from Opt-Pair")
    parser.add_argument("--cost-samples", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--details", action="store_true", help="Also save pair scores, selected inputs, and per-pair diagnostics")
    parser.add_argument("--solc-dir", type=Path)
    parser.add_argument("--legacy-version", default="0.5.17")
    parser.add_argument("--smoke-mutation", choices=MUTATIONS, default="CH",
                        help="The single negative mutation type in the smoke Mut-Set (default: CH)")
    parser.add_argument("--mock-llm", action="store_true", help="Include synthetic LLM records in the smoke test")
    args = parser.parse_args()
    if args.out is None:
        args.out = Path("smoke-results" if args.exp == "smoke" else "results")
    if args.bootstrap is None:
        args.bootstrap = 20 if args.exp == "smoke" else 1000
    if args.exp == "8":
        args.calibrate = True
    counts = (args.seed, args.bootstrap, args.noise_count, args.positive_count, args.cost_samples, args.warmup)
    if min(counts) < 0 or not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        parser.error("Use nonnegative counts and a finite threshold in [0, 1]")
    try:
        run(args)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()


