from pathlib import Path

from solcx.exceptions import SolcError

from data import standard_input
from util import compile_standard_with_version


CALLS = {"call", "callcode", "delegatecall", "staticcall"}
ATTACKER = "0x4213421342134213421342134213421342134213"
# Old: OBSERVATIONS also included receipts_root. Receipts encode gas usage, so
# a different receipt root alone is not evidence of changed contract behavior.
OBSERVATIONS = {"calls", "writes", "logs", "exit", "storage_root"}
DIAGNOSTICS = {"receipts_root", "gas_used"}
EVENT_FIELDS = {
    "call": ("depth", "op", "target", "value", "input_size", "input_prefix"),
    "write": ("depth", "op", "slot", "value"),
    "log": ("depth", "op", "topics", "data_size", "data_prefix"),
    "exception": ("depth", "op"),
}
GENERATOR_CAPABILITIES = {
    "scope": "Yul AST sites; general Solidity statement rewriting is unsupported",
    "CH": ["literal target redirection", "computed target call-type replacement"],
    "SH": ["literal slot increment", "computed slot XOR with 1"],
    "RS": ["adjacent literal SSTORE/CALL exchange"],
    "PT": ["literal calldata length increment"],
    "ET": ["literal topic0 or LOG0 length increment"],
    "validation": "ABI checks and changed bytecode do not prove semantic divergence; replay/audit required",
}


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
    calls, computed_stores, proposals = [], [], {}
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
            if number(args[0]) is None:
                computed_stores.append(args[0])
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

    # Old generation supported literal targets/slots only, while labeling all
    # recipes static. Add explicit computed-operand variants, with one evaluation
    # of each original operand and no claim of a reachable behavioral difference.
    if "CH" not in proposals:
        replacements = []
        for node in calls:
            name = yul_name(node)
            args = node["arguments"]
            if number(args[1]) is not None:
                continue
            if name in {"call", "callcode"}:
                # Dropping a nonconstant value operand can remove side effects.
                if number(args[2]) != 0:
                    continue
                new_name, kept = "delegatecall", args[:2] + args[3:]
            else:
                new_name = "staticcall" if name == "delegatecall" else "delegatecall"
                kept = args
            arguments = [data[slice(*span(arg))] for arg in kept]
            replacement = new_name.encode() + b"(" + b", ".join(arguments) + b")"
            replacements.append(edit(source, data, *span(node), replacement))
        if len(replacements) >= 2 and len(replacements) * 2 > len(calls):
            proposals["CH"] = replacements

    if "SH" not in proposals and computed_stores:
        target = computed_stores[0]
        lo, hi = span(target)
        proposals["SH"] = [edit(source, data, lo, hi, b"xor(" + data[lo:hi] + b", 1)")]

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
                failures.append({"id": spec["id"], "reason": "no supported Yul mutation site",
                                 "generator_capabilities": GENERATOR_CAPABILITIES})
            dataset = "mixed" if spec.get("split") == "mixed" else "mut"
            for kind, edits in proposals.items():
                # Traceability must be established from compiled original sites,
                # not inferred from whether source text happened to be a literal.
                # Until an instruction/source-map audit is supplied, retain only
                # the source-level variant, rather than fabricating Table 10 groups.
                variant = {"CH": "literal_target_redirection" if all(change["after"] == ATTACKER for change in edits)
                           else "computed_target_call_type_replacement",
                           "SH": "computed_slot_xor" if edits[0]["after"].startswith("xor(") else "literal_slot_increment",
                           "PT": "literal_calldata_length_increment", "ET": "literal_log_field_increment",
                           "RS": "adjacent_literal_cei_exchange"}[kind]
                recipes.append({"id": f"yul-{kind}", "mutation": kind, "version": version,
                                "dataset": dataset, "contract": spec, "edits": edits,
                                "generator": "reviewed_yul_v2", "variant": variant,
                                "requires_differential_confirmation": True})
        except (OSError, SolcError, KeyError, ValueError) as error:
            failures.append({"id": spec["id"], "error": str(error)})
    return recipes, failures


def comparable_events(events):
    """Drop diagnostic metadata while validating the ordered event schema."""
    if not isinstance(events, list):
        raise ValueError("Replay events must be a list")
    normalized = []
    for event in events:
        if not isinstance(event, dict) or event.get("kind") not in EVENT_FIELDS:
            raise ValueError("Unknown or malformed ordered replay event")
        fields = EVENT_FIELDS[event["kind"]]
        if not set(fields) <= event.keys():
            raise ValueError(f"Missing fields in ordered {event['kind']} event")
        normalized.append({"kind": event["kind"], **{key: event[key] for key in fields}})
    return normalized


def verified_events(observation, identifier):
    """Check that an ordered archive did not omit projected trace evidence."""
    events = comparable_events(observation["events"])
    projections = {"calls": [], "writes": [], "exceptions": []}
    for event in events:
        if event["kind"] == "call":
            projections["calls"].append([event[key] for key in
                                         ("depth", "op", "target", "value", "input_size", "input_prefix")])
        elif event["kind"] == "write":
            projections["writes"].append([event[key] for key in ("depth", "slot", "value")])
        elif event["kind"] == "exception":
            projections["exceptions"].append([event["depth"], event["op"]])
    for field in ("calls", "writes"):
        if projections[field] != observation[field]:
            raise ValueError(f"Ordered events do not match {field} projection in replay {identifier}")
    exit_observation = observation["exit"]
    if not isinstance(exit_observation, dict) or "exceptions" not in exit_observation:
        raise ValueError(f"Ordered replay {identifier} requires exit.exceptions projection")
    if projections["exceptions"] != exit_observation["exceptions"]:
        raise ValueError(f"Ordered events do not match exit.exceptions projection in replay {identifier}")
    # Trace LOGs include reverted attempts; receipt logs contain only committed
    # emissions, so comparing these two lists would reject valid EVM traces.
    return events


def verify_replays(records: list[dict], pair_ids: set[str]) -> dict:
    outcomes = {}
    for record in records:
        identifier = record["id"]
        if identifier not in pair_ids or identifier in outcomes:
            raise ValueError(f"Unknown or duplicate replay pair: {identifier}")
        schema_version = record.get("schema_version", 1)
        if type(schema_version) is not int or schema_version not in {1, 2}:
            raise ValueError(f"Unsupported replay schema version for {identifier}: {schema_version}")
        if not record.get("deployable", True):
            outcomes[identifier] = {"status": "undeployable"}
            continue
        runs = record["transactions"]
        if len(runs) != 5:
            raise ValueError(f"Expected five transactions for replay {identifier}")
        differing, diagnostic_differences = [], []
        order_verified = True
        for tx in runs:
            if not tx.get("context") or not OBSERVATIONS <= tx["a"].keys() or not OBSERVATIONS <= tx["b"].keys():
                raise ValueError(f"Missing context or observable fields in replay {identifier}")
            differences = sorted(key for key in OBSERVATIONS if tx["a"][key] != tx["b"][key])
            left, right = tx["a"], tx["b"]
            if ("events" in left) != ("events" in right):
                raise ValueError(f"One-sided ordered event evidence in replay {identifier}")
            if "events" in left:
                # Old: event-list presence alone marked ordering as verified,
                # even when events=[] contradicted nonempty calls or writes.
                if verified_events(left, identifier) != verified_events(right, identifier):
                    differences.append("events")
            else:
                if schema_version >= 2:
                    raise ValueError(f"Version 2 replay {identifier} requires ordered events")
                order_verified = False
            differing.append(differences)
            diagnostic_differences.append(sorted(key for key in DIAGNOSTICS
                                                 if key in left and key in right and left[key] != right[key]))
        outcomes[identifier] = {"status": "diverged" if any(differing) else "no_divergence",
                                "differences": differing, "diagnostic_differences": diagnostic_differences,
                                "order_verified": order_verified, "evidence_scope": "observed_behavior"}
    executable = [row for row in outcomes.values() if row["status"] != "undeployable"]
    confirmed = sum(row["status"] == "diverged" for row in executable)
    return {"sampled": len(outcomes), "executable": len(executable), "confirmed": confirmed,
            "confirmation_rate": confirmed / len(executable) if executable else None,
            "confirmation_rate_all": confirmed / len(outcomes) if outcomes else None,
            "ordered_pairs": sum(row["order_verified"] for row in executable),
            "legacy_unordered_pairs": sum(not row["order_verified"] for row in executable),
            "diagnostic_only_pairs": sum(not any(row["differences"]) and any(row["diagnostic_differences"])
                                         for row in executable),
            "evidence_scope": "observed_behavior", "pairs": outcomes}


# 测试抽检样本全部无法部署时的统计，检查可执行样本数为 0 时的确认率。
# cases = [{"id": "case-1", "deployable": False}]
# result = verify_replays(cases, {"case-1"})
# print(result["sampled"], result["executable"], result["confirmation_rate"])
