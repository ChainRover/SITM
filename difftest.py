"""Compare archived runtime bytecode with a Solidity reconstruction on local Anvil."""

import argparse
import json
import re
from pathlib import Path


CONTRACT_ADDR = "0x6Bb545B8A2e869009487eb7e5e045178b7221e36"
RUNTIME_FILE = Path(f"data/hex/{CONTRACT_ADDR}_runtime.hex")
SOL_FILE = Path(f"output/gemini3.1-pro/d_{CONTRACT_ADDR}.sol")
SOLC_VERSION = "0.4.24"
CONSTRUCTOR_ARGS = ()
LOCAL_RPC = "http://127.0.0.1:8545"
TEST_SENDER = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"

# 有构造参数时可以修改这里，也可以传 --constructor-args。
# CONSTRUCTOR_ARGS = (1000000 * 10**18, "SMBQ Token", "SMBQ")


def hex_data(value):
    if isinstance(value, str):
        value = bytes.fromhex(value.strip().removeprefix("0x").removeprefix("0X"))
    return "0x" + bytes(value).hex()


def extract_selectors(runtime_hex):
    code = bytes.fromhex(hex_data(runtime_hex)[2:])
    selectors = set()
    pc = 0
    while pc < len(code):
        opcode = code[pc]
        if opcode == 0x63 and pc + 4 < len(code):
            selectors.add("0x" + code[pc + 1:pc + 5].hex())
            # print("PUSH4:", hex(pc), code[pc + 1:pc + 5].hex())
        width = opcode - 0x5f if 0x60 <= opcode <= 0x7f else 0
        pc += 1 + width
    return selectors


def canonical_type(item):
    kind = item["type"]
    if kind.startswith("tuple"):
        fields = ",".join(canonical_type(field) for field in item["components"])
        return "(" + fields + ")" + kind[5:]
    return re.sub(r"^(u?int)(?=\[|$)", r"\g<1>256", kind)


def function_signature(item):
    fields = ",".join(canonical_type(field) for field in item.get("inputs", []))
    return f"{item['name']}({fields})"


def get_abi_selectors(abi):
    from web3 import Web3

    return {hex_data(Web3.keccak(text=function_signature(item))[:4])
            for item in abi if item["type"] == "function"}


def compile_and_match_main(runtime_hex, args):
    from util import compile_standard_with_version

    source = args.source.read_text(encoding="utf-8-sig")
    settings = {
        "optimizer": {"enabled": not args.no_optimize, "runs": args.runs},
        "outputSelection": {"*": {"*": ["abi", "evm.bytecode.object", "evm.deployedBytecode.object"]}},
    }
    if args.evm_version:
        settings["evmVersion"] = args.evm_version
    config = {"language": "Solidity", "sources": {args.source.name: {"content": source}},
              "settings": settings}
    compiled = compile_standard_with_version(config, args.solc_version, solc_dir=args.solc_dir)
    contracts = compiled.get("contracts", {}).get(args.source.name, {})
    selectors = extract_selectors(runtime_hex)
    candidates = []
    print(f"Runtime PUSH4 candidates: {len(selectors)}")
    for name, contract in contracts.items():
        creation = contract["evm"]["bytecode"]["object"]
        runtime = contract["evm"]["deployedBytecode"]["object"]
        if not creation or not runtime:
            continue
        functions = get_abi_selectors(contract["abi"])
        score = len(selectors & functions) / max(len(selectors), len(functions), 1)
        candidates.append((score, name, contract))
        # print("matching candidate:", name, len(functions), score)
        # print("runtime bytes:", len(runtime) // 2)
    if args.contract_name:
        candidates = [item for item in candidates if item[1] == args.contract_name]
    if not candidates:
        raise ValueError("No deployable contract found; check --source and --contract-name")
    candidates.sort(key=lambda item: (-item[0], item[1]))
    score, name, contract = candidates[0]
    if not args.contract_name:
        if score < 0.2:
            raise ValueError("Selector match is below 0.20; check the source or pass --contract-name")
        if len(candidates) > 1 and candidates[1][0] == score:
            raise ValueError("Several contracts share the best selector score; pass --contract-name")
    creation = hex_data(contract["evm"]["bytecode"]["object"])
    runtime = hex_data(contract["evm"]["deployedBytecode"]["object"])
    print(f"Selected {name}: selector overlap {score:.1%}")
    return name, contract["abi"], creation, runtime


def rpc(w3, method, *params):
    result = w3.provider.make_request(method, list(params))
    if "error" in result:
        raise RuntimeError(f"{method}: {result['error']}")
    if "result" not in result:
        raise RuntimeError(f"{method}: missing RPC result")
    return result["result"]


def evm_snapshot(w3):
    return rpc(w3, "evm_snapshot")


def evm_revert(w3, snapshot):
    if rpc(w3, "evm_revert", snapshot) is not True:
        raise RuntimeError(f"Failed to restore snapshot {snapshot}")


def set_contract_code(w3, address, runtime_hex):
    runtime = hex_data(runtime_hex)
    rpc(w3, "anvil_setCode", address, runtime)
    if hex_data(w3.eth.get_code(address)) != runtime:
        raise RuntimeError("Runtime replacement did not take effect")


def deploy_contract(w3, abi, creation_code, constructor_args, sender, gas, timeout):
    contract = w3.eth.contract(abi=abi, bytecode=creation_code)
    transaction = contract.constructor(*constructor_args).transact({"from": sender, "gas": gas})
    receipt = w3.eth.wait_for_transaction_receipt(transaction, timeout=timeout)
    # print("deployment status:", receipt["status"], receipt["contractAddress"])
    if receipt["status"] != 1 or not receipt["contractAddress"]:
        raise RuntimeError("Constructor transaction reverted; check arguments and deployment gas")
    return receipt["contractAddress"]


def abi_to_strategy(item, addresses=()):
    from hypothesis import strategies as st
    from web3 import Web3

    if isinstance(item, str):
        item = {"type": item}
    kind = item["type"]
    # 数组要先处理，uint256[] 不能进入整数分支。
    if kind.endswith("]"):
        inner, length = kind.rsplit("[", 1)
        size = length[:-1]
        strategy = abi_to_strategy({**item, "type": inner}, addresses)
        return st.lists(strategy, min_size=int(size) if size else 0, max_size=int(size) if size else 16)
    if kind == "tuple":
        return st.tuples(*(abi_to_strategy(field, addresses) for field in item["components"]))
    integer = re.fullmatch(r"(u?int)(\d*)", kind)
    if integer:
        bits = int(integer[2] or "256")
        if bits < 8 or bits > 256 or bits % 8:
            raise ValueError(f"Invalid ABI integer type: {kind}")
        low, high = (0, 2**bits - 1) if integer[1] == "uint" else (-2**(bits - 1), 2**(bits - 1) - 1)
        return st.one_of(st.integers(min_value=max(low, -16), max_value=min(high, 16)),
                         st.integers(min_value=low, max_value=high))
    if kind == "address":
        random_address = st.binary(min_size=20, max_size=20).map(Web3.to_checksum_address)
        if addresses:
            known = st.sampled_from([Web3.to_checksum_address(address) for address in addresses])
            return st.one_of(known, random_address)
        return random_address
    if kind == "bool":
        return st.booleans()
    if kind == "string":
        return st.text(max_size=64)
    if kind == "bytes":
        return st.binary(max_size=64)
    if re.fullmatch(r"bytes([1-9]|[12]\d|3[0-2])", kind):
        size = int(kind[5:])
        return st.binary(min_size=size, max_size=size)
    raise ValueError(f"Unsupported ABI type: {kind}")


def observe_call(w3, address, abi, signature, values, sender, args, gas_price):
    contract = w3.eth.contract(address=address, abi=abi)
    function = contract.get_function_by_signature(signature)(*values)
    transaction = function.transact({"from": sender, "value": args.value,
                                     "gas": args.gas, "gasPrice": gas_price})
    receipt = w3.eth.wait_for_transaction_receipt(transaction, timeout=args.timeout)
    trace = rpc(w3, "debug_traceTransaction", hex_data(transaction),
                {"disableStorage": True, "disableStack": True, "enableMemory": False})
    if "returnValue" not in trace:
        raise RuntimeError("Transaction trace has no returnValue")
    proof = w3.eth.get_proof(address, [], receipt["blockNumber"])
    logs = [{"address": log["address"].lower(), "topics": [hex_data(topic) for topic in log["topics"]],
             "data": hex_data(log["data"])} for log in receipt["logs"]]
    result = {"ok": receipt["status"] == 1, "return_data": hex_data(trace["returnValue"]),
              "storage_root": hex_data(proof["storageHash"]), "logs": logs,
              "balance": w3.eth.get_balance(address, receipt["blockNumber"])}
    # print("transaction status:", signature, receipt["status"])
    # print("observed fields:", result)
    return result


def diff_test_once(w3, address, abi, signature, values, sender, runtime_a, runtime_b, args):
    block = w3.eth.get_block("latest")
    timestamp = block["timestamp"] + 1
    gas_price = w3.eth.gas_price
    observations = []
    # 同一地址、同一初始状态分别执行；每一侧完成后都恢复快照。
    for runtime in (runtime_a, runtime_b):
        snapshot = evm_snapshot(w3)
        try:
            set_contract_code(w3, address, runtime)
            rpc(w3, "evm_setNextBlockTimestamp", timestamp)
            if "baseFeePerGas" in block:
                rpc(w3, "anvil_setNextBlockBaseFeePerGas", hex(block["baseFeePerGas"]))
            observations.append(observe_call(w3, address, abi, signature, values, sender, args, gas_price))
        finally:
            evm_revert(w3, snapshot)
    a, b = observations
    differences = [field for field in a if a[field] != b[field]]
    # print("differing fields:", signature, differences)
    return {"same": not differences, "differences": differences, "a": a, "b": b}


def is_stateful(item):
    if item.get("type") != "function":
        return False
    if "stateMutability" in item:
        return item["stateMutability"] not in {"view", "pure"}
    return not item.get("constant", False)


def test_function(w3, address, abi, item, sender, runtime_a, runtime_b, args):
    from hypothesis import given, seed, settings

    signature = function_signature(item)
    addresses = [sender, address, "0x" + "00" * 20, *w3.eth.accounts[:3]]
    strategies = [abi_to_strategy(field, addresses) for field in item.get("inputs", [])]
    from hypothesis import strategies as st

    result = {"function": signature, "status": "no_difference", "executions": 0,
              "successful_pairs": 0, "both_reverted": 0}
    print(f"Testing {signature}", flush=True)

    @seed(args.seed)
    @settings(max_examples=args.cases, deadline=None, database=None)
    @given(values=st.tuples(*strategies))
    def check(values):
        observation = diff_test_once(w3, address, abi, signature, values, sender, runtime_a, runtime_b, args)
        result["executions"] += 1
        if observation["a"]["ok"] and observation["b"]["ok"]:
            result["successful_pairs"] += 1
        elif not observation["a"]["ok"] and not observation["b"]["ok"]:
            result["both_reverted"] += 1
        # print("case:", signature, values, observation["differences"])
        if not observation["same"]:
            result["counterexample"] = {"arguments": values, **observation}
            raise AssertionError(f"{signature}: different {observation['differences']}; args={values!r}")

    try:
        check()
    except AssertionError as error:
        if "counterexample" not in result:
            raise
        result["status"] = "different"
        result["message"] = str(error)
    if result["status"] == "no_difference" and not result["successful_pairs"]:
        result["status"] = "inconclusive"
    print(f"  {result['status']}: {result['executions']} executions, "
          f"{result['successful_pairs']} successful pairs, {result['both_reverted']} both reverted", flush=True)
    if "message" in result:
        print(" ", result["message"], flush=True)
    return result


def json_value(value):
    if isinstance(value, bytes):
        return hex_data(value)
    raise TypeError(f"Cannot save {type(value).__name__} as JSON")


def run(args):
    from web3 import Web3

    w3 = Web3(Web3.HTTPProvider(args.rpc, request_kwargs={"timeout": args.timeout}))
    if not w3.is_connected():
        raise ConnectionError(f"Cannot connect to local Anvil at {args.rpc}")
    if "anvil" not in w3.client_version.lower():
        raise ValueError("This script requires Anvil snapshot and code-replacement RPC methods")
    sender = Web3.to_checksum_address(args.sender)
    runtime_a = hex_data(args.runtime.read_text(encoding="utf-8-sig"))
    if runtime_a == "0x":
        raise ValueError("The reference runtime file is empty")
    name, abi, creation, _ = compile_and_match_main(runtime_a, args)
    constructor = next((item for item in abi if item["type"] == "constructor"), {})
    if len(constructor.get("inputs", [])) != len(args.constructor_args):
        raise ValueError(f"Constructor inputs: {constructor.get('inputs', [])}; check --constructor-args")
    functions = [item for item in abi if item["type"] == "function" and (args.include_view or is_stateful(item))]
    if args.function:
        functions = [item for item in functions if function_signature(item) in args.function]
        missing = set(args.function) - {function_signature(item) for item in functions}
        if missing:
            raise ValueError(f"Functions not selected: {sorted(missing)}; use full signatures and --include-view if needed")
    if not functions:
        raise ValueError("No eligible functions to test; try --include-view")
    print(f"Contract: {name}; selected functions: {len(functions)}")
    print("Initial state: reconstructed constructor; independent calls from that state")
    results = []
    snapshot = evm_snapshot(w3)
    try:
        address = deploy_contract(w3, abi, creation, args.constructor_args, sender, args.deploy_gas, args.timeout)
        runtime_b = hex_data(w3.eth.get_code(address))
        if runtime_b == "0x":
            raise RuntimeError("Deployment produced no runtime code")
        # 使用实际部署的 runtime，保留构造函数填入的 immutable 值。
        # print("test address:", address, "runtime bytes:", (len(runtime_b) - 2) // 2)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
        for item in functions:
            result = test_function(w3, address, abi, item, sender, runtime_a, runtime_b, args)
            results.append(result)
            if args.out:
                with args.out.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"contract": name, "seed": args.seed, **result},
                                            ensure_ascii=False, default=json_value) + "\n")
    finally:
        evm_revert(w3, snapshot)
    if any(row["status"] == "different" for row in results):
        return 1
    return 2 if any(row["status"] == "inconclusive" for row in results) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=RUNTIME_FILE)
    parser.add_argument("--source", type=Path, default=SOL_FILE)
    parser.add_argument("--contract-name")
    parser.add_argument("--solc-version", default=SOLC_VERSION)
    parser.add_argument("--solc-dir", type=Path)
    parser.add_argument("--evm-version")
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--constructor-args", type=json.loads, default=list(CONSTRUCTOR_ARGS), help="JSON array")
    parser.add_argument("--rpc", default=LOCAL_RPC)
    parser.add_argument("--sender", default=TEST_SENDER)
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gas", type=int, default=2_000_000)
    parser.add_argument("--deploy-gas", type=int, default=5_000_000)
    parser.add_argument("--value", type=int, default=0, help="Call value in wei")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--function", action="append", default=[], help="Full signature, e.g. transfer(address,uint256)")
    parser.add_argument("--include-view", action="store_true")
    parser.add_argument("--out", type=Path, help="Append one JSONL result per function")
    args = parser.parse_args()
    if (min(args.cases, args.gas, args.deploy_gas, args.timeout) <= 0 or min(args.runs, args.value) < 0
            or not isinstance(args.constructor_args, list)):
        parser.error("Counts, gas and timeout must be positive; runs/value nonnegative; constructor args a JSON array")
    if args.out and args.out.resolve() in {args.runtime.resolve(), args.source.resolve()}:
        parser.error("The result path must differ from input files")
    try:
        return run(args)
    except ImportError as error:
        parser.exit(2, f"Missing dependency: {error}. Install project requirements, web3 and hypothesis.\n")
    except Exception as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
