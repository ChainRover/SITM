"""Report missing reproducibility evidence without inventing paper artifacts."""

import hashlib
import json
import platform
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


REQUIRED_ARTIFACTS = {
    "source_inventory", "split_manifest", "mutation_recipes", "replay_cases",
    "e2v_training", "smarttrans_training", "e2v_checkpoint", "smarttrans_checkpoint",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_input_paths(manifest):
    """Protect the manifest and every referenced file from report replacement."""
    if manifest is None:
        return []
    manifest = Path(manifest)
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    entries = spec.get("artifacts")
    if not isinstance(entries, list):
        raise ValueError("Artifact manifest needs an artifacts list")
    paths = [manifest]
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("Each artifact needs a string path")
        path = Path(entry["path"])
        paths.append(path if path.is_absolute() else manifest.parent / path)
    return paths


def artifact_evidence(manifest):
    """Hash local files. A matching hash establishes identity, not authenticity."""
    result = {"artifacts": [], "missing_roles": sorted(REQUIRED_ARTIFACTS), "issues": []}
    if manifest is None:
        result["issues"].append("No frozen artifact manifest supplied")
        return result
    manifest = Path(manifest)
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    entries = spec.get("artifacts")
    if not isinstance(entries, list):
        raise ValueError("Artifact manifest needs an artifacts list")
    result["manifest_sha256"] = sha256_file(manifest)
    roles = set()
    for entry in entries:
        if not isinstance(entry, dict) or not all(isinstance(entry.get(k), str) for k in ("role", "path", "sha256")):
            raise ValueError("Each artifact needs string role, path and sha256")
        path = Path(entry["path"])
        if not path.is_absolute():
            path = manifest.parent / path
        row = {"role": entry["role"], "path": entry["path"], "expected_sha256": entry["sha256"]}
        if not re.fullmatch(r"[0-9a-fA-F]{64}", entry["sha256"]):
            row["status"] = "invalid_expected_hash"
        elif not path.is_file():
            row["status"] = "missing_file"
        else:
            row["actual_sha256"] = sha256_file(path)
            row["status"] = "matched" if row["actual_sha256"] == entry["sha256"].lower() else "hash_mismatch"
        if row["status"] == "matched":
            roles.add(row["role"])
        else:
            result["issues"].append(f"{row['role']}: {row['status']} ({row['path']})")
        result["artifacts"].append(row)
    result["missing_roles"] = sorted(REQUIRED_ARTIFACTS - roles)
    return result


def execution_provenance():
    root = Path(__file__).resolve().parent
    source_files = sorted(root.glob("*.py")) + sorted(root.glob("requirements*.txt"))
    result = {"python": sys.version, "platform": platform.platform(),
              "source_sha256": {path.name: sha256_file(path) for path in source_files}}
    try:
        result["git_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
        result["git_worktree_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL, timeout=5).strip())
    except (OSError, subprocess.SubprocessError):
        result["git_revision"] = None
    return result


def paper_readiness(summary, pairs, rows, manifest=None):
    from data import temporal_provenance
    from e2v import opcode_text

    issues = []
    if summary.get("bootstrap", 0) < 1000:
        issues.append("Paper confidence intervals require at least 1000 contract-aware bootstrap resamples")
    for dataset, expected in (("opt", 24079), ("calibration", 500), ("mixed", 500), ("temporal", 150)):
        contracts = {row["contract_id"].strip().lower() for row in pairs if row["dataset"] == dataset}
        if len(contracts) != expected:
            issues.append(f"{dataset} unique contract count: {len(contracts)} (expected {expected})")
    temporal_profiles = {}
    temporal_mutations = Counter()
    for row in pairs:
        if row["dataset"] == "temporal":
            profile = temporal_profiles.setdefault(row["contract_id"].strip().lower(), Counter())
            profile[row["label"]] += 1
            if row["label"] == 0:
                temporal_mutations[row.get("mutation")] += 1
    if any(profile != Counter({1: 1, 0: 2}) for profile in temporal_profiles.values()):
        issues.append("Temporal needs one equivalent and two mutation pairs per contract")
    if temporal_mutations != Counter({"CH": 150, "RS": 150}):
        issues.append("Temporal needs 150 CH and 150 RS negative pairs")
    incomplete = [item["table"] for item in summary["table_coverage"] if item["status"] != "computed"]
    if incomplete:
        issues.append(f"Incomplete paper table inputs: {incomplete}")
    artifacts = artifact_evidence(manifest)
    issues.extend(artifacts["issues"])
    if artifacts["missing_roles"]:
        issues.append("Missing frozen artifact roles: " + ", ".join(artifacts["missing_roles"]))
    synthetic = sum(row.get("sample_origin") in {"smoke", "synthetic_mechanism_check"} for row in pairs)
    if synthetic:
        issues.append(f"Synthetic smoke/mechanism pairs cannot establish paper reproduction: {synthetic}")
    missing_address = sum(not re.fullmatch(r"0x[0-9a-fA-F]{40}",
                          str(row.get("address") or row["contract_id"])) for row in pairs)
    if missing_address:
        issues.append(f"Pairs without stable on-chain addresses: {missing_address}")
    temporal_missing = sum(not temporal_provenance(row)["deployment_claim_complete"]
                           for row in pairs if row["dataset"] == "temporal")
    if temporal_missing:
        issues.append(f"Temporal pairs without complete deployment evidence declarations: {temporal_missing}")
    unarchived = Counter()
    inconsistent_e2v = []
    by_id = {row["id"]: row for row in pairs}
    for row in rows:
        for method in ("E2V", "SmartTrans"):
            if method not in row.get("scores", {}):
                continue
            details = row.get("model_details", {}).get(method, {})
            training = details.get("training_provenance", {})
            if (not (details.get("checkpoint_sha256") or details.get("checkpoint_files_sha256"))
                    or not details.get("input_file_sha256")
                    or training.get("status") != "archived_metadata"):
                unarchived[method] += 1
            if method == "E2V" and details.get("representation", "opcode-only") == "opcode-only":
                source = by_id[row["id"]]
                if (opcode_text(source["a"]) == opcode_text(source["b"])
                        and abs(row["scores"][method] - 1.0) > 1e-5):
                    inconsistent_e2v.append(row["id"])
    for method, count in sorted(unarchived.items()):
        issues.append(f"{method} scores without complete checkpoint/input/training provenance: {count}")
    if inconsistent_e2v:
        issues.append(f"Opcode-only E2V scores differ from 1 for identical model inputs: {len(inconsistent_e2v)}")
    return {"ready": not issues, "scope": "artifact_identity_and_computational_coverage",
            "independent_training_or_chain_verification": False,
            "numerical_match_to_manuscript": "not_checked_requires_original_results",
            "issues": issues, "artifacts": artifacts,
            "e2v_identical_input_mismatches": inconsistent_e2v}
