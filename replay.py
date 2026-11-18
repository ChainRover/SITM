import json
import os
import socket
import subprocess
import time
from contextlib import contextmanager
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

from evm_sit import CALL_OPS, bytecode_bytes


def rpc(url: str, method: str, *params):
    body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    request = Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with build_opener(ProxyHandler({})).open(request, timeout=60) as response:
        payload = json.load(response)
    if "error" in payload:
        raise ValueError(f"{method}: {payload['error']}")
    return payload["result"]


@contextmanager
def local_chain(executable: str, hardfork: str, chain_id: int, timestamp: int, block_number: int):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    endpoint = f"http://127.0.0.1:{port}"
    command = [executable, "--host", "127.0.0.1", "--port", str(port), "--quiet", "--steps-tracing",
               "--accounts", "0", "--hardfork", hardfork, "--chain-id", str(chain_id),
               "--timestamp", str(timestamp), "--number", str(block_number)]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, creationflags=flags)
    try:
        for _ in range(100):
            if process.poll() is not None:
                raise ValueError(f"Anvil exited during startup with status {process.returncode}")
            try:
                rpc(endpoint, "eth_chainId")
                break
            except URLError:
                time.sleep(.05)
        else:
            raise ValueError("Anvil did not become ready")
        yield endpoint
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def integer(value: int | str) -> int:
    return int(value, 0) if isinstance(value, str) else value


def seed_account(url: str, address: str, state: dict) -> None:
    if "balance" in state:
        rpc(url, "anvil_setBalance", address, hex(integer(state["balance"])))
    if "nonce" in state:
        rpc(url, "anvil_setNonce", address, hex(integer(state["nonce"])))
    if "code" in state:
        rpc(url, "anvil_setCode", address, "0x" + bytecode_bytes(state["code"]).hex())
    for slot, value in state.get("storage", {}).items():
        rpc(url, "anvil_setStorageAt", address, f"0x{integer(slot):064x}", f"0x{integer(value):064x}")


def observable_steps(steps: list[dict]) -> tuple[list, list, list]:
    calls, writes, exceptions = [], [], []
    for step in steps:
        name, depth = step["op"], step["depth"]
        if name in CALL_OPS:
            args = [int(word, 16) for word in reversed(step.get("stack", []))]
            has_value = name in {"CALL", "CALLCODE"}
            arity = 7 if has_value else 6
            if len(args) < arity:
                exceptions.append([depth, "CALL_STACK_UNDERFLOW"])
                continue
            offset_index = 3 if has_value else 2
            offset, size = args[offset_index:offset_index + 2]
            memory = step.get("memory") or ""
            if isinstance(memory, list):
                memory = "".join(word.removeprefix("0x") for word in memory)
            data = bytecode_bytes(memory)
            # A suffix of zeroes is implicit. Retain the total requested size.
            prefix = data[offset:offset + size].rstrip(b"\0")
            calls.append([depth, name, f"0x{args[1] & ((1 << 160)-1):040x}",
                          args[2] if has_value else 0, size, prefix.hex()])
        elif name == "SSTORE":
            args = step.get("stack", [])
            if len(args) >= 2:
                writes.append([depth, int(args[-1], 16), int(args[-2], 16)])
        elif name in {"REVERT", "INVALID"} or step.get("error"):
            exceptions.append([depth, name])
    return calls, writes, exceptions


def replay_side(case: dict, code: str, executable: str) -> list[dict]:
    target = case["target"]
    genesis = case.get("genesis", {})
    with local_chain(executable, genesis.get("hardfork", "shanghai"), genesis.get("chain_id", 31337),
                     genesis.get("timestamp", 1700000000), genesis.get("number", 0)) as url:
        for address, state in case["accounts"].items():
            seed_account(url, address, state)
        rpc(url, "anvil_setCode", target, "0x" + bytecode_bytes(code).hex())
        senders = {tx["from"] for tx in case["transactions"]}
        for sender in senders:
            rpc(url, "anvil_impersonateAccount", sender)

        genesis_block = rpc(url, "eth_getBlockByNumber", "latest", False)
        observations = []
        for tx in case["transactions"]:
            rpc(url, "evm_setNextBlockTimestamp", integer(tx["timestamp"]))
            base_fee = tx.get("base_fee", genesis_block.get("baseFeePerGas"))
            if base_fee is not None:
                rpc(url, "anvil_setNextBlockBaseFeePerGas", hex(integer(base_fee)))
            request = {"from": tx["from"], "to": target, "data": "0x" + bytecode_bytes(tx.get("data", "")).hex(),
                       "gas": hex(integer(tx.get("gas", 3_000_000))), "value": hex(integer(tx.get("value", 0)))}
            if "gas_price" in tx:
                request["gasPrice"] = hex(integer(tx["gas_price"]))
            transaction = rpc(url, "eth_sendTransaction", request)
            receipt = rpc(url, "eth_getTransactionReceipt", transaction)
            result = rpc(url, "debug_traceTransaction", transaction, {"enableMemory": True, "disableStorage": True})
            calls, writes, exceptions = observable_steps(result["structLogs"])
            logs = [{key: entry[key] for key in ("address", "topics", "data")} for entry in receipt["logs"]]
            proof = rpc(url, "eth_getProof", target, [], receipt["blockNumber"])
            block = rpc(url, "eth_getBlockByHash", receipt["blockHash"], False)
            observations.append({"calls": calls, "writes": writes, "logs": logs,
                                 "exit": {"status": receipt["status"], "exceptions": exceptions,
                                          "return_value": result["returnValue"]},
                                 "storage_root": proof["storageHash"], "receipts_root": block["receiptsRoot"]})
        return observations


def run_replays(cases: list[dict], pairs: list[dict], executable: str) -> list[dict]:
    by_id = {row["id"]: row for row in pairs if row.get("label") == 0}
    results = []
    for case in cases:
        identifier = case["id"]
        if identifier not in by_id:
            raise ValueError(f"No negative pair found for replay {identifier}")
        if case.get("deployable") is False:
            results.append({"id": identifier, "deployable": False})
            continue
        if len(case["transactions"]) != 5:
            raise ValueError(f"Expected five transactions for replay {identifier}")
        pair = by_id[identifier]
        original = replay_side(case, pair["a"], executable)
        mutated = replay_side(case, pair["b"], executable)
        transactions = []
        for context, a, b in zip(case["transactions"], original, mutated):
            transactions.append({"context": context, "a": a, "b": b})
        results.append({"id": identifier, "transactions": transactions})
    return results
