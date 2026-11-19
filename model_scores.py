from hashlib import sha256
import json
import math
from pathlib import Path

from evm_sit import bytecode_bytes, strip_metadata
from data import read_jsonl, write_jsonl


def bytecode_id(code):
    return sha256(bytecode_bytes(code)).hexdigest()


def file_sha256(path):
    """Hash large checkpoints without loading the entire file into memory."""
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_hashes(path):
    """Return content hashes for the actual files used by an external model."""
    path = Path(path)
    if path.is_file():
        return {path.name: file_sha256(path)}
    if not path.is_dir():
        raise ValueError(f"Missing model artifact: {path}")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Empty model artifact directory: {path}")
    return {item.relative_to(path).as_posix(): file_sha256(item) for item in files}


def artifact_fingerprint(path):
    hashes = artifact_hashes(path)
    encoded = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"sha256": sha256(encoded).hexdigest(), "file_count": len(hashes)}


def export_bytecodes(inputs, directory):
    directory.mkdir(parents=True, exist_ok=True)
    seen = set()
    for row in read_jsonl(inputs):
        for field in ("a", "b", "clean_b"):
            if field not in row:
                continue
            # Old code exported raw metadata, unlike the comparison pipeline.
            # code = bytecode_bytes(row[field])
            code = strip_metadata(row[field])
            name = bytecode_id(code)
            if name not in seen:
                (directory / f"{name}.hex").write_text(code.hex() + "\n", encoding="ascii")
                seen.add(name)
    return len(seen)


def score_file(inputs, output, name, similarity, details, diagnostics=None):
    records = []
    provenance = {**details, "input_file_sha256": file_sha256(inputs),
                  "preprocessing": "solidity_cbor_metadata_stripped"}
    for row in read_jsonl(inputs):
        if row.get("status") == "failed":
            continue
        try:
            # Old direct scoring relied on a separate export command to strip
            # metadata. Apply the same preprocessing for every scoring entry.
            # score = similarity(row["a"], row["b"])
            codes = {key: strip_metadata(row[key]) for key in ("a", "b", "clean_b") if key in row}
            score = float(similarity(codes["a"], codes["b"]))
            if not math.isfinite(score):
                raise ValueError("Non-finite model similarity")
            pair_details = {**provenance, "bytecode_sha256": {key: bytecode_id(code) for key, code in codes.items()}}
            if diagnostics is not None:
                pair_details["diagnostics"] = diagnostics(codes["a"], codes["b"])
            result = {"id": row["id"], "scores": {name: score}, "model_details": {name: pair_details}}
            if "clean_b" in row:
                clean_score = float(similarity(codes["a"], codes["clean_b"]))
                if not math.isfinite(clean_score):
                    raise ValueError("Non-finite clean model similarity")
                result["clean_scores"] = {name: clean_score}
            records.append(result)
        except (KeyError, ValueError) as error:
            raise ValueError(f"{row.get('id', inputs)}: {error}") from error
    if not records:
        raise ValueError("The model input file contains no pairs")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, records)
    print(f"{name}: wrote {len(records)} scores to {output}")
