from hashlib import sha256

from evm_sit import bytecode_bytes
from data import read_jsonl, write_jsonl


def bytecode_id(code):
    return sha256(bytecode_bytes(code)).hexdigest()


def export_bytecodes(inputs, directory):
    directory.mkdir(parents=True, exist_ok=True)
    seen = set()
    for row in read_jsonl(inputs):
        for field in ("a", "b", "clean_b"):
            if field not in row:
                continue
            code = bytecode_bytes(row[field])
            name = bytecode_id(code)
            if name not in seen:
                (directory / f"{name}.hex").write_text(code.hex() + "\n", encoding="ascii")
                seen.add(name)
    return len(seen)


def score_file(inputs, output, name, similarity, details):
    records = []
    for row in read_jsonl(inputs):
        try:
            score = similarity(row["a"], row["b"])
            result = {"id": row["id"], "scores": {name: score}, "model_details": {name: details}}
            if "clean_b" in row:
                result["clean_scores"] = {name: similarity(row["a"], row["clean_b"])}
            records.append(result)
        except (KeyError, ValueError) as error:
            raise ValueError(f"{row.get('id', inputs)}: {error}") from error
    if not records:
        raise ValueError("The model input file contains no pairs")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, records)
    print(f"{name}: wrote {len(records)} scores to {output}")
