from pathlib import Path

from solcx.exceptions import SolcError

from data import standard_input
from util import compile_standard_with_version


CALLS = {"call", "callcode", "delegatecall", "staticcall"}
ATTACKER = "0x4213421342134213421342134213421342134213"
OBSERVATIONS = {"calls", "writes", "logs", "exit", "storage_root", "receipts_root"}


def walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk(value)


def yul_name(node: dict) -> str | None:
    if node and node.get("nodeType") == "YulFunctionCall":
        return (node.get("functionName") or {}).get("name")
    return None


def number(node: dict) -> int | None:
    if node.get("nodeType") != "YulLiteral" or node.get("kind") != "number":
        return None
    value = node["value"].replace("_", "")
    return int(value, 16 if value.startswith("0x") else 10)


def span(node: dict) -> tuple[int, int]:
    start, length, _ = node["src"].split(":")
    return int(start), int(start) + int(length)


def edit(source: str, data: bytes, start: int, end: int, replacement: bytes) -> dict:
    return {"source": source, "start": start, "before": data[start:end].decode("utf-8"),
            "after": replacement.decode("utf-8")}


def propose_yul(ast: dict, source: str, content: str) -> dict[str, list[dict]]:
    data = content.encode("utf-8")
    nodes = list(walk(ast))
    calls, proposals = [], {}
    for node in nodes:
        name = yul_name(node)
        if not name:
            continue
        args = node["arguments"]
        if name in CALLS:
            calls.append(node)
            positions = [("PT", 4 if name in {"call", "callcode"} else 3, 32)]
        elif name == "sstore":
            positions = [("SH", 0, 1)]
        elif name in {"log0", "log1", "log2", "log3", "log4"}:
            positions = [("ET", 2 if name != "log0" else 1, 1)]
        else:
            continue
        for kind, index, increment in positions:
            value = number(args[index])
            if kind in proposals or value is None or (kind == "PT" and value < 4):
                continue
            if value + increment >= 1 << 256:
                continue
            first, last = span(args[index])
            proposals[kind] = [edit(source, data, first, last, str(value + increment).encode())]

    targets = []
    attacker = int(ATTACKER, 16)
    for node in calls:
        target = node["arguments"][1]
        value = number(target)
        if value is not None and value != attacker:
            targets.append(target)
    if len(targets) >= 2 and len(targets) * 2 > len(calls):
        proposals["CH"] = [edit(source, data, *span(node), ATTACKER.encode()) for node in targets]

    for node in nodes:
        if node.get("nodeType") != "YulBlock":
            continue
        statements = node.get("statements") or []
        for a, b in zip(statements, statements[1:]):
            left, right = (a or {}).get("expression"), (b or {}).get("expression")
            if yul_name(left) != "sstore" or yul_name(right) != "pop":
                continue
            nested = right["arguments"][0]
            if yul_name(nested) not in CALLS:
                continue
            # Only literal statements, with an optional gas() operand. General
            # Solidity statement motion needs reviewed data-dependence analysis.
            if any(number(arg) is None for arg in left["arguments"]):
                continue
            args = nested["arguments"]
            if any(number(arg) is None for arg in args[1:]) or (number(args[0]) is None and yul_name(args[0]) != "gas"):
                continue
            lo, mid = span(a)
            next_lo, hi = span(b)
            replacement = data[next_lo:hi] + data[mid:next_lo] + data[lo:mid]
            proposals["RS"] = [edit(source, data, lo, hi, replacement)]
            return proposals
    return proposals


def discover_recipes(specs: list[dict], root: Path, version: str, solc_dir: Path | None) -> tuple[list, list]:
    recipes, failures = [], []
    for spec in specs:
        if spec.get("split", "test") not in {"test", "mixed"}:
            continue
        try:
            config = standard_input(spec, root, version, False, ast=True)
            file, name = spec["target"].rsplit(":", 1)
            output = compile_standard_with_version(config, version, solc_dir=solc_dir)
            source_ast = output["sources"][file]["ast"]
            target = next((node for node in source_ast["nodes"]
                           if node.get("nodeType") == "ContractDefinition" and node["name"] == name), None)
            if target is None:
                raise ValueError(f"No AST for {spec['target']}")
            proposals = propose_yul(target, file, config["sources"][file]["content"])
            if not proposals:
                failures.append({"id": spec["id"], "reason": "no supported literal Yul site"})
            dataset = "mixed" if spec.get("split") == "mixed" else "mut"
            for kind, edits in proposals.items():
                recipes.append({"id": f"yul-{kind}", "mutation": kind, "version": version,
                                "dataset": dataset, "contract": spec, "edits": edits,
                                **({"traceability": "static"} if kind in {"CH", "SH"} else {})})
        except (OSError, SolcError, KeyError, ValueError) as error:
            failures.append({"id": spec["id"], "error": str(error)})
    return recipes, failures


def verify_replays(records: list[dict], pair_ids: set[str]) -> dict:
    outcomes = {}
    for record in records:
        identifier = record["id"]
        if identifier not in pair_ids or identifier in outcomes:
            raise ValueError(f"Unknown or duplicate replay pair: {identifier}")
        if not record.get("deployable", True):
            outcomes[identifier] = {"status": "undeployable"}
            continue
        runs = record["transactions"]
        if len(runs) != 5:
            raise ValueError(f"Expected five transactions for replay {identifier}")
        differing = []
        for tx in runs:
            if not tx.get("context") or not OBSERVATIONS <= tx["a"].keys() or not OBSERVATIONS <= tx["b"].keys():
                raise ValueError(f"Missing context or observable fields in replay {identifier}")
            differing.append(sorted(key for key in OBSERVATIONS if tx["a"][key] != tx["b"][key]))
        outcomes[identifier] = {"status": "diverged" if any(differing) else "no_divergence",
                                "differences": differing}
    executable = [row for row in outcomes.values() if row["status"] != "undeployable"]
    confirmed = sum(row["status"] == "diverged" for row in executable)
    return {"sampled": len(outcomes), "executable": len(executable), "confirmed": confirmed,
            "confirmation_rate": confirmed / len(executable) if executable else None, "pairs": outcomes}


# 测试抽检样本全部无法部署时的统计，检查可执行样本数为 0 时的确认率。
# cases = [{"id": "case-1", "deployable": False}]
# result = verify_replays(cases, {"case-1"})
# print(result["sampled"], result["executable"], result["confirmation_rate"])
