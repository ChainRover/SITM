import argparse
from pathlib import Path
from evm_sit import strip_metadata
from data import (assign_splits, load_pairs, prepare_compiler_pairs, prepare_llm,
                  prepare_mutants, read_jsonl, with_noise, write_jsonl)
from mutations import discover_recipes, verify_replays


def write_prepared(path: Path, records: list[dict], failures: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, records)
    write_jsonl(path.with_name(path.stem + ".failures.jsonl"), failures)
    print(f"{path}: {len(records)} rows, {len(failures)} failures")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    descriptions = {"prepare": "Compile optimizer and version pairs", "make-recipes": "Find supported literal Yul mutation sites",
                    "prepare-mutants": "Compile reviewed mutation recipes and verify ABI", "prepare-llm": "Import archived LLM attempts"}
    for name, description in descriptions.items():
        command = commands.add_parser(name, help=description)
        command.add_argument("manifest", type=Path)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--source-root", type=Path)
        command.add_argument("--solc-dir", type=Path)
        if name == "make-recipes":
            command.add_argument("--version", default="0.8.20")

    split = commands.add_parser("split", help="Create a seeded address-level split from an eligible inventory")
    split.add_argument("manifest", type=Path)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--sizes", nargs=4, type=int, default=(19200, 500, 500, 24079),
                       metavar=("TRAIN", "CALIBRATION", "MIXED", "TEST"))
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--min-opcodes", type=int)

    export = commands.add_parser("export-model-inputs", help="Export preprocessed bytecode for external baselines")
    export.add_argument("pairs", nargs="+", type=Path)
    export.add_argument("--noise-count", type=int, default=0)
    export.add_argument("--seed", type=int, default=42)
    export.add_argument("--output", type=Path, required=True)

    replay = commands.add_parser("replay", help="Replay sampled negative pairs in local Anvil")
    replay.add_argument("pairs", nargs="+", type=Path)
    replay.add_argument("--cases", type=Path, required=True)
    replay.add_argument("--anvil", default="anvil")
    replay.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "split":
            if min(args.sizes) < 0:
                raise ValueError("Split sizes cannot be negative")
            write_prepared(args.output, assign_splits(read_jsonl(args.manifest), args.sizes, args.seed, args.min_opcodes), [])
        elif args.command == "export-model-inputs":
            if min(args.seed, args.noise_count) < 0:
                raise ValueError("Noise count and seed cannot be negative")
            pairs = with_noise(load_pairs(args.pairs), args.noise_count, args.seed)
            records = []
            for row in pairs:
                if row.get("status") == "failed":
                    continue
                record = {key: row[key] for key in ("id", "dataset", "contract_id")}
                for key in ("a", "b", "clean_b"):
                    if key in row:
                        record[key] = strip_metadata(row[key]).hex()
                records.append(record)
            write_prepared(args.output, records, [])
        elif args.command == "replay":
            from replay import run_replays
            pairs = load_pairs(args.pairs)
            records = run_replays(read_jsonl(args.cases), pairs, args.anvil)
            verify_replays(records, {row["id"] for row in pairs if row.get("label") == 0})
            write_prepared(args.output, records, [])
        else:
            root = args.source_root or args.manifest.parent
            inputs = read_jsonl(args.manifest)
            if args.command == "prepare":
                records, failures = prepare_compiler_pairs(inputs, root, args.solc_dir)
            elif args.command == "make-recipes":
                records, failures = discover_recipes(inputs, root, args.version, args.solc_dir)
            elif args.command == "prepare-mutants":
                records, failures = prepare_mutants(inputs, root, args.solc_dir)
            else:
                records = prepare_llm(inputs, root, args.solc_dir)
                failures = [row for row in records if row["status"] == "failed"]
            write_prepared(args.output, records, failures)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
