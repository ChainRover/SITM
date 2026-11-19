

import json
from datetime import datetime, timezone
from hashlib import sha256
from copy import deepcopy

#  from pyevmasm import disassemble_all
from dataclasses import dataclass
from pathlib import Path
from semantic_version import Version
from solcx.exceptions import SolcError
from random import Random



from evm_sit import bytecode_bytes, strip_metadata

from util import compile_standard_with_version


VERSIONS = ("0.5.17", "0.6.12", "0.7.6", "0.8.19", "0.8.20")
MUTATIONS = ("CH", "SH", "RS", "PT", "ET")
DATASETS = {"opt", "ver", "mut", "noise", "llm", "calibration", "mixed", "temporal"}
NOISE_TYPES = ("metadata", "junk", "deadcode", "combined")
TEMPORAL_CUTOFF = datetime(2023, 10, 1, tzinfo=timezone.utc)


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                # if number <= 3:
                #     print("input record:", path.name, number, sorted(item) if isinstance(item, dict) else type(item))
                if not isinstance(item, dict):
                    raise ValueError("expected an object")
            except ValueError as error:
                raise ValueError(f"{path}:{number}: {error}") from error
            records.append(item)
    # print("loaded JSONL:", path, "records:", len(records))
    return records


def write_jsonl(path: Path, records) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def load_pairs(paths: list[Path]) -> list[dict]:
    pairs = []
    for path in paths:
        pairs.extend(read_jsonl(path))
    return validate_pairs(pairs)


def bytecode_fingerprint(code) -> str:
    """Identity evidence independent of filenames, hex case and solc metadata."""
    return sha256(strip_metadata(code)).hexdigest()


def temporal_provenance(row: dict) -> dict:
    """Validate supplied deployment claims; this does not query a chain node."""
    deployment = row.get("deployment", {})
    if not isinstance(deployment, dict):
        raise ValueError(f"{row['id']}: deployment must be an object")
    timestamp = deployment.get("timestamp", row.get("deployed_at"))
    evidence = deployment.get("evidence", row.get("deployment_evidence"))
    block = deployment.get("block_number", row.get("deployment_block"))
    if evidence is not None and (not isinstance(evidence, str) or not evidence.strip()):
        raise ValueError(f"{row['id']}: deployment evidence must be a nonempty reference string")
    if timestamp is not None:
        try:
            parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"{row['id']}: deployment timestamp must use ISO 8601") from error
        if parsed.tzinfo is None or parsed < TEMPORAL_CUTOFF:
            raise ValueError(f"{row['id']}: temporal deployment must be timezone-aware and on/after 2023-10-01")
    if block is not None and (type(block) is not int or block < 0):
        raise ValueError(f"{row['id']}: deployment block must be a nonnegative integer")
    corpora = row.get("source_corpora", [])
    if not isinstance(corpora, list) or any(not isinstance(name, str) for name in corpora):
        raise ValueError(f"{row['id']}: source_corpora must be a list of corpus names")
    original = {"smartbugs", "smartbugs2.0", "scrawid", "scrawld"}
    if any(name.lower().replace(" ", "") in original for name in corpora):
        raise ValueError(f"{row['id']}: temporal sample is declared part of an original corpus")
    return {"deployment_claim_complete": timestamp is not None and block is not None and bool(evidence),
            "independent_chain_verification": False,
            "cutoff": TEMPORAL_CUTOFF.isoformat()}


def validate_pairs(pairs: list[dict]) -> list[dict]:
    seen = set()
    attempts = set()
    partitions = {}
    fingerprints = {}
    for row in pairs:
        key, dataset = row["id"], row["dataset"]
        # print("validate:", key, dataset, row.get("label"), row.get("mutation"))
        if key in seen or dataset not in DATASETS:
            raise ValueError(f"Duplicate pair identifier or unknown dataset: {key}, {dataset}")
        seen.add(key)
        if not isinstance(row["contract_id"], str) or not row["contract_id"].strip():
            raise ValueError(f"{key}: contract_id must be a nonempty string")
        contract = row["contract_id"].strip().lower()
        # Old: temporal was grouped with the original evaluation data, allowing
        # the same contract into both the original corpus and temporal holdout.
        partition = dataset if dataset in {"calibration", "mixed", "temporal"} else "test"
        # print("contract partition:", contract, partition, "previous:", partitions.get(contract))
        if partitions.setdefault(contract, partition) != partition:
            raise ValueError(f"{contract} occurs in incompatible subsets: {partitions[contract]} and {partition}")
        if dataset == "temporal":
            temporal_provenance(row)
        if dataset in {"mut", "temporal", "mixed"} and row.get("label") not in (0, 1):
            raise ValueError(f"{key} needs label=1 (equivalent) or label=0 (mutated)")
        if dataset == "calibration" and row.get("label", 1) != 1:
            raise ValueError(f"{key}: alpha calibration needs equivalent pairs")
        if row.get("label") == 0 and row.get("mutation") not in MUTATIONS:
            raise ValueError(f"{key} has an unknown mutation operator")
        if dataset == "llm":
            model, status = row.get("model"), row.get("status", "success")
            if not model or status not in {"success", "failed"} or (contract, model) in attempts:
                raise ValueError(f"{key} needs a unique contract/model attempt with status success or failed")
            attempts.add((contract, model))
            if status == "failed":
                # Failed attempts may retain a usable reference; never require
                # a generated runtime that the model failed to produce.
                fields = ("a",) if row.get("a") else ()
            else:
                fields = ("a", "b")
        else:
            fields = ("a", "b")
        if dataset == "noise" and (row.get("noise") not in NOISE_TYPES or not bytecode_bytes(row["clean_b"])):
            raise ValueError(f"{key} needs a noise type and nonempty clean_b")
        for field in fields:
            # print("bytecode field:", key, field, "bytes:", len(bytecode_bytes(row[field])))
            if not bytecode_bytes(row[field]):
                raise ValueError(f"{key} has empty runtime bytecode in {field}")
        # Old isolation used only caller-supplied contract IDs. Filename-derived
        # legacy IDs could hide an identical bytecode in another partition.
        # Smoke fixtures deliberately reuse tiny bytecodes and are explicitly
        # synthetic; publication checks must exclude this origin entirely.
        if row.get("sample_origin") != "smoke":
            identity_fields = (*fields, *(("clean_b",) if dataset == "noise" else ()))
            for field in identity_fields:
                fingerprint = bytecode_fingerprint(row[field])
                previous = fingerprints.setdefault(fingerprint, (partition, key))
                if previous[0] != partition:
                    raise ValueError(f"{key}: normalized bytecode overlaps {previous[1]} across "
                                     f"{previous[0]}/{partition} subsets")
    # print("validated IDs:", len(seen), "contracts:", len(partitions), "LLM attempts:", len(attempts))
    return pairs


def pair(dataset: str, contract: str, suffix: str, a: str, b: str, **metadata) -> dict:
    return {"id": f"{dataset}:{contract}:{suffix}", "dataset": dataset, "contract_id": contract,
            "a": a, "b": b, **metadata}


def assign_splits(records: list[dict], sizes: list[int], seed: int, min_opcodes: int | None) -> list[dict]:
    eligible = {}
    seen = set()
    for spec in records:
        identifier = spec["id"].lower()
        if identifier in seen:
            raise ValueError(f"Resolve duplicate contract inventory entry before splitting: {identifier}")
        seen.add(identifier)
        if min_opcodes is None or spec["opcode_count"] > min_opcodes:
            eligible[identifier] = spec
    picked = [eligible[key] for key in sorted(eligible)]
    # print("inventory:", len(records), "eligible:", len(picked), "minimum opcodes:", min_opcodes)
    if len(picked) != sum(sizes):
        raise ValueError(f"Split sizes require {sum(sizes)} eligible contracts, got {len(picked)}")
    Random(seed).shuffle(picked)
    # print("split seed:", seed, "first IDs:", [spec["id"] for spec in picked[:5]])
    result = []
    start = 0
    for split, size in zip(("train", "calibration", "mixed", "test"), sizes):
        # print("split slice:", split, start, start + size)
        result.extend({**spec, "split": split} for spec in picked[start:start + size])
        start += size
    return result


def standard_input(spec: dict, root: Path, version: str, optimize: bool, *, ast: bool = False) -> dict:
    sources = {**spec["sources"], **spec.get("source_overrides", {}).get(version, {})}
    settings = deepcopy(spec.get("settings", {}))
    settings.update(deepcopy(spec.get("settings_overrides", {}).get(version, {})))
    optimizer = {**settings.get("optimizer", {}), "enabled": optimize}
    optimizer.setdefault("runs", 200)
    settings["optimizer"] = optimizer
    settings["outputSelection"] = {"*": {"*": ["abi", "evm.deployedBytecode.object"]}}
    if ast:
        settings["outputSelection"]["*"][""] = ["ast"]
    compiler_sources = {}
    for name, filename in sources.items():
        content = (root / filename).read_bytes().decode("utf-8-sig")
        compiler_sources[name] = {"content": content}
    return {"language": "Solidity", "settings": settings, "sources": compiler_sources}


@dataclass(frozen=True)
class Build:
    bytecode: str
    abi: list[dict]


def build(config: dict, target: str, version: str, solc_dir: Path | None) -> Build:
    file, name = target.rsplit(":", 1)
    output = compile_standard_with_version(config, version, solc_dir=solc_dir)
    # print("build target:", target, "version:", version)
    # print("compiled files:", list(output.get("contracts", {})))
    contract = output.get("contracts", {}).get(file, {}).get(name)
    if not contract:
        raise ValueError(f"Compiled source does not contain {target}")
    code = bytecode_bytes(contract["evm"]["deployedBytecode"]["object"])
    # print("runtime bytes:", len(code), "ABI entries:", len(contract["abi"]))
    if not code:
        raise ValueError(f"{target} has no runtime bytecode")
    return Build(code.hex(), contract["abi"])


def prepare_compiler_pairs(specs: list[dict], root: Path, solc_dir: Path | None) -> tuple[list, list]:
    pairs, failures = [], []
    for spec in specs:
        identifier = spec["id"]
        split = spec.get("split", "test")
        if split not in {"train", "calibration", "mixed", "test"}:
            raise ValueError(f"Unknown split for {identifier}: {split}")
        if split == "train":
            continue
        versions = sorted(set(spec.get("versions", VERSIONS)), key=Version)
        # print("compiler pair contract:", identifier, "split:", split, "versions:", versions)
        compiled = {}
        for version in versions:
            for optimize in (False, True):
                try:
                    config = standard_input(spec, root, version, optimize)
                    compiled[version, optimize] = build(config, spec["target"], version, solc_dir)
                    # print("compile OK:", identifier, version, optimize)
                    # print("runtime bytes:", len(compiled[version, optimize].bytecode) // 2)
                except (OSError, SolcError, ValueError, KeyError) as error:
                    failures.append({"id": identifier, "version": version, "optimize": optimize, "error": str(error)})
                    # print("compile skipped:", identifier, version, optimize, str(error)[:200])

        complete = [version for version in versions if (version, False) in compiled and (version, True) in compiled]
        # print("versions with both optimizer settings:", complete)
        if complete:
            version = spec.get("opt_version", complete[0])
            if version not in complete:
                failures.append({"id": identifier, "version": version, "error": "opt_version did not compile both ways"})
            else:
                dataset = {"test": "opt", "calibration": "calibration", "mixed": "mixed"}[split]
                pairs.append(pair(dataset, identifier, version, compiled[version, False].bytecode,
                                  compiled[version, True].bytecode, version=version, label=1))

        if split != "test":
            continue
        optimize = spec.get("version_optimize", False)
        available = [version for version in versions if (version, optimize) in compiled]
        if not available:
            continue
        earliest = available[0]
        # print("version baseline:", earliest, "targets:", available[1:])
        for version in available[1:]:
            groups = spec.get("version_groups", {})
            pairs.append(pair("ver", identifier, f"{earliest}-{version}", compiled[earliest, optimize].bytecode,
                              compiled[version, optimize].bytecode, base_version=earliest, version=version,
                              group=groups.get(version, "unclassified"), optimize=optimize, label=1))
    return pairs, failures


def apply_edits(config: dict, edits: list[dict]) -> dict:
    mutated = deepcopy(config)
    by_file = {}
    for edit in edits:
        by_file.setdefault(edit["source"], []).append(edit)
    for source, group in by_file.items():
        content = config["sources"][source]["content"].encode("utf-8")
        # print("edited source:", source, "edits:", len(group), "UTF-8 bytes:", len(content))
        changed = content
        boundary = len(content)
        for edit in sorted(group, key=lambda item: item["start"], reverse=True):
            start = edit["start"]
            expected = edit["before"].encode("utf-8")
            end = start + len(expected)
            # print("edit byte range:", start, end, "previous boundary:", boundary)
            # print("before:", content[start:end].decode("utf-8"), "after:", edit["after"])
            if not 0 <= start < end <= boundary or content[start:end] != expected:
                raise ValueError(f"Stale or overlapping mutation span in {source} at byte {start}")
            changed = changed[:start] + edit["after"].encode("utf-8") + changed[end:]
            boundary = start
        mutated["sources"][source]["content"] = changed.decode("utf-8")
        # print("source bytes before / after:", len(content), len(changed))
    return mutated


def abi_signature(abi: list[dict]) -> list[str]:
    return sorted(json.dumps(entry, sort_keys=True) for entry in abi)


def prepare_mutants(recipes: list[dict], root: Path, solc_dir: Path | None) -> tuple[list, list]:
    pairs, failures = [], []
    for recipe in recipes:
        spec = recipe["contract"]
        version = recipe["version"]
        kind = recipe["mutation"]
        dataset = recipe.get("dataset", "mut")
        if kind not in MUTATIONS or dataset not in {"mut", "mixed", "temporal"}:
            raise ValueError(f"Unknown mutation category or split in {recipe['id']}")
        try:
            config = standard_input(spec, root, version, recipe.get("optimize", False))
            original = build(config, spec["target"], version, solc_dir)
            changed = build(apply_edits(config, recipe["edits"]), spec["target"], version, solc_dir)
            # print("mutation:", recipe["id"], kind, "solc:", version)
            # print("runtime bytes before / after:", len(original.bytecode) // 2, len(changed.bytecode) // 2)
            # print("ABI unchanged:", abi_signature(original.abi) == abi_signature(changed.abi))
            if abi_signature(original.abi) != abi_signature(changed.abi):
                raise ValueError("Mutation changed the ABI")
            if strip_metadata(original.bytecode) == strip_metadata(changed.bytecode):
                raise ValueError("Mutation only changed metadata or produced identical bytecode")
            # print("runtime differs after metadata removal:", recipe["id"])
            pairs.append(pair(dataset, spec["id"], recipe["id"], original.bytecode, changed.bytecode,
                              label=0, mutation=kind, version=version, evidence="operator_and_abi",
                              **{key: recipe[key] for key in ("traceability", "difficulty", "generator", "variant",
                                                            "requires_differential_confirmation") if key in recipe}))
        except (OSError, SolcError, ValueError, KeyError) as error:
            failures.append({"id": recipe["id"], "mutation": kind, "error": str(error)})
            # print("mutation rejected:", recipe["id"], kind, str(error)[:200])
    return pairs, failures


def prepare_llm(specs: list[dict], root: Path, solc_dir: Path | None) -> list[dict]:
    records = []
    for spec in specs:
        for model, attempt in spec["attempts"].items():
            output = dict(id=f"llm:{spec['id']}:{model}", dataset="llm", contract_id=spec["id"],
                          model=model, a=spec["reference"], category=spec.get("category", "unclassified"))
            try:
                if "error" in attempt:
                    raise ValueError(attempt["error"])
                if "runtime" in attempt:
                    runtime = bytecode_bytes(attempt["runtime"])
                else:
                    version = attempt["version"]
                    config = standard_input(attempt, root, version, attempt.get("optimize", False))
                    runtime = bytecode_bytes(build(config, attempt["target"], version, solc_dir).bytecode)
                if not runtime:
                    raise ValueError("Empty runtime")
                output.update(status="success", b=runtime.hex())
            except (OSError, SolcError, ValueError, KeyError) as error:
                output.update(status="failed", error=str(error))
            records.append(output)
    return records


def generate_noise(b: bytes, rng: Random) -> dict[str, bytes]:
    # Non-standard metadata uses fixed-width PUSH payloads, keeping random data out
    # of opcode positions. Appending code preserves existing jump offsets.
    metadata = b"".join(b"\x7f" + rng.randbytes(32) for _ in range(rng.randint(1, 2)))
    junk = bytes.fromhex("600050") * 3
    dead = bytes.fromhex("600060ff5600") * rng.randint(2, 5)
    # print("噪声字节数", len(metadata), len(junk), len(dead))
    return {"metadata": b + b"\x00" + metadata, "junk": b + b"\x00" + junk,
            "deadcode": b + b"\x00" + dead, "combined": b + b"\x00" + metadata + junk + dead}


def noise_pairs(opt_pairs: list[dict], count: int, seed: int) -> list[dict]:
    rng = Random(seed)
    ordered = sorted(opt_pairs, key=lambda row: row["id"])
    sampled = rng.sample(ordered, min(count, len(ordered)))
    rows = []
    for row in sampled:
        variants = generate_noise(bytecode_bytes(row["b"]), rng)
        for kind, runtime in variants.items():
            rows.append(pair("noise", row["contract_id"], f"{row['id']}:{kind}", row["a"], runtime.hex(),
                             clean_b=row["b"], noise=kind, noise_origin="local_suffix",
                             sample_origin=row.get("sample_origin", "provided_manifests")))
    return rows


def with_noise(pairs: list[dict], count: int, seed: int) -> list[dict]:
    if not count or any(row["dataset"] == "noise" for row in pairs):
        return pairs
    return pairs + noise_pairs([row for row in pairs if row["dataset"] == "opt"], count, seed)


LEGACY_FILES = {
    "dataset_opt_pair.jsonl": "opt",
    "dataset_ver_pair.jsonl": "ver",
    "dataset_mut_pair.jsonl": "mut",
    "dataset_noise_pair.jsonl": "noise",
    "dataset_calib.jsonl": "calibration",
    "dataset_mixed.jsonl": "mixed",
    "dataset_temporal.jsonl": "temporal",
}


# 五字段负样本的直接转换写法，检查旧文件字段时可单独调用。
# def load_mutation_negatives_direct(path):
#     records = []
#     for number, row in enumerate(read_jsonl(Path(path)), 1):
#         if row["label"] != 0:
#             continue
#         contract = row.get("address") or row.get("contract_id")
#         if not contract:
#             contract = f"sample-{number}"
#         converted = pair(
#             "mut", contract, row["pair_id"],
#             row["bytecode_original"], row["bytecode_mutated"],
#             label=0, mutation=row["mutation_type"],
#         )
#         converted["legacy_pair_id"] = row["pair_id"]
#         records.append(converted)
#     return records


def legacy_pair(row, dataset, path, number):
    contract = str(row.get("contract_id") or row.get("address") or "").strip().lower()
    fallback = contract in {"", "unknown", "?"}
    if fallback:
        # Old: filename/line identity could hide the same contract across splits.
        # A failed attempt without reference bytecode remains explicitly unknown.
        contract = f"legacy:unresolved:{path.stem}:{row.get('pair_id', number)}"

    fields = {}
    suffix = row.get("pair_id", number)
    if "bytecode_original" in row:
        a, b = row["bytecode_original"], row["bytecode_mutated"]
        fields["label"] = int(row.get("label", 0))
        if fields["label"] == 0:
            fields["mutation"] = row["mutation_type"].strip().upper()
    elif "bytecode_unopt" in row:
        a, b = row["bytecode_unopt"], row["bytecode_opt"]
        fields.update(label=int(row.get("label", 1)), version=row.get("sol_v", "unknown"))
        suffix = fields["version"] or suffix
    elif "bytecode_base" in row:
        a, b = row["bytecode_base"], row["bytecode_target"]
        fields.update(base_version=row["baseline_version"], version=row["target_version"])
        suffix = f"{fields['base_version']}-{fields['version']}"
        group = row.get("subset", "unclassified")
        fields["legacy_subset"] = group
        fields["group"] = {"equivalence": "minor", "semantic_change": "major"}.get(group, group)
        if dataset in {"mixed", "calibration", "mut", "temporal"}:
            fields["label"] = int(row.get("label", 1))
    elif "bytecode_clean" in row:
        a, b = row["bytecode_clean"], row["bytecode_noisy"]
        fields.update(clean_b=a, noise=row["noise_type"], noise_origin="legacy_archive")
    elif dataset == "llm":
        a, b = row.get("origin_bytecode"), row.get("generated_bytecode")
        model = row.get("model") or path.stem.removeprefix("sitm_downstream_").removesuffix("_pairs")
        status = row.get("status") or ("success" if a and b else "failed")
        fields.update(model=model, status=status)
        if "error" in row:
            fields["error"] = row["error"]
        suffix = model
    else:
        raise ValueError("Unrecognized bytecode fields")

    if fallback and a:
        contract = f"legacy:runtime:{bytecode_fingerprint(a)}"
    for key in ("traceability", "difficulty", "sample_origin", "category",
                "function_event_matches", "deployment", "deployed_at", "deployment_evidence",
                "deployment_block", "source_corpora"):
        if key in row:
            fields[key] = row[key]
    result = pair(dataset, contract, str(suffix), a, b, **fields)
    # print("legacy mapping:", path.name, number, row.get("pair_id"), "->", result["id"])
    # print("normalized labels:", result.get("label"), result.get("mutation"))
    if "pair_id" in row:
        result["legacy_pair_id"] = row["pair_id"]
    if fallback:
        result["contract_id_source"] = "legacy_record_without_address"
    return result


# 分组前先找重复合约标识，用来检查 mixed、calibration 和测试部分的交集。
# def inspect_split_overlap(records):
#     groups = {"calibration": set(), "mixed": set(), "test": set()}
#     for row in records:
#         dataset = row["dataset"]
#         group = dataset if dataset in ("calibration", "mixed") else "test"
#         groups[group].add(row["contract_id"].lower())
#     overlap = {}
#     names = ("calibration", "mixed", "test")
#     for i, left in enumerate(names):
#         for right in names[i + 1:]:
#             shared = groups[left] & groups[right]
#             overlap[f"{left}/{right}"] = sorted(shared)
#     return overlap


def read_pairs(path, dataset=None):
    rows = read_jsonl(path)
    dataset = dataset or LEGACY_FILES.get(path.name)
    # print("reading pairs:", path, "dataset hint:", dataset, "rows:", len(rows))
    if dataset == "calibration" and any(str(row.get("label", 1)) == "0" for row in rows):
        dataset = "mixed"
        print(f"{path.name}: negative labels found; reading legacy records as mixed validation", flush=True)
    if not dataset and rows and ("origin_bytecode" in rows[0] or path.name.startswith("sitm_downstream_")):
        dataset = "llm"
    results = []
    for number, row in enumerate(rows, 1):
        if "dataset" in row:
            results.append(row)
            continue
        if not dataset:
            raise ValueError(f"{path}: use current pair fields or a recognized legacy filename")
        try:
            results.append(legacy_pair(row, dataset, path, number))
            # if number <= 3:
            #     print("normalized pair:", results[-1]["id"], results[-1]["contract_id"])
        except (KeyError, ValueError) as error:
            raise ValueError(f"{path}, record {number}: {error}") from error
    # ids = [row["id"] for row in results]
    # print("normalized rows / unique IDs:", len(ids), len(set(ids)))
    return results


# 读正式数据前，可以先单独看类型分布和旧 ID 转换后的样子。
# from collections import Counter
# records = read_pairs(Path("dataset/smartbugs/dataset_mut_pair.jsonl"))
# print("样本数量", len(records))
# print("标签数量", Counter(row.get("label") for row in records))
# print("变异类型", Counter(row.get("mutation") for row in records))
# print([row["id"] for row in records[:3]])


