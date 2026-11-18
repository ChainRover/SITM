import math
import os
import re
#  from tqdm import tqdm 
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable

# from evm_cfg_builder.cfg import CFG
import networkx as nx
import solcx
from rapidfuzz.distance.Levenshtein import distance as _levenshtein
from semantic_version import NpmSpec, Version

from evm_sit import Analysis, Bytecode, Event, analyze, bytecode_bytes, disassemble, strip_metadata


ALPHA_WEIGHTS = {"call": 1.8, "store": 0.6, "log": 0.3, "exc": 0.5, "ret": 0.1}
THETA_HI = 0.70
THETA_LO = 0.40
GAMMA = 0.35


def softmax_weights(alpha: Mapping[str, float]) -> dict[str, float]:
    highest = max(alpha.values())
    exponentials = {kind: math.exp(value - highest) for kind, value in alpha.items()}
    total = sum(exponentials.values())
    return {kind: value / total for kind, value in exponentials.items()}


W_TAU = softmax_weights(ALPHA_WEIGHTS)
NON_CODE = re.compile(r"/\*.*?\*/|//[^\r\n]*|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'", re.DOTALL)


def _compiler_in(directory: Path, version: str) -> Path | None:
    stem = f"solc-v{version}"
    for filename in (stem, stem + ".exe", f"{stem}/solc.exe"):
        path = directory / filename
        if path.is_file():
            return path.resolve()
    return None


def _compiler_version(source: str, directory: Path | None) -> str:
    source_code = NON_CODE.sub(" ", source)
    pragmas = re.findall(r"\bpragma\s+solidity\s+([^;]+);", source_code)
    # print("pragma constraints:", pragmas)
    if not pragmas:
        return "0.8.20"
    literals = set(re.findall(r"\b\d+\.\d+\.\d+\b", " ".join(pragmas)))
    if len(literals) == 1 and all(re.fullmatch(r"=?\s*\d+\.\d+\.\d+", p.strip()) for p in pragmas):
        return literals.pop()
    constraints = [NpmSpec(re.sub(r"([<>=~^]+)\s+", r"\1", pragma.strip())) for pragma in pragmas]
    if directory is None:
        installed = {str(version).removeprefix("v") for version in solcx.get_installed_solc_versions()}
    else:
        installed = set()
        for path in directory.glob("solc-v*"):
            match = re.fullmatch(r"solc-v(\d+\.\d+\.\d+)(?:\.exe)?", path.name)
            if match:
                installed.add(match[1])
    for candidates in (installed, literals | {"0.8.20"}):
        versions = map(Version, candidates)
        compatible = [version for version in versions if all(spec.match(version) for spec in constraints)]
        # print("installed candidates:", sorted(candidates))
        # print("compatible:", [str(version) for version in compatible])
        if compatible:
            return str(max(compatible))
    raise ValueError("No compatible compiler found for the pragma; pass solc_version explicitly")


def compile_with_optimizer(
    solidity_code: str,
    optimize: bool,
    optimize_runs: int = 200,
    *,
    contract_name: str | None = None,
    solc_version: str | None = None,
    evm_version: str | None = None,
    solc_dir: str | Path | None = None,
) -> bytes:
    """Compile a self-contained source file and return deployed bytecode.

    Use contract_name when the source contains multiple deployable contracts.
    solc_dir (or SITM_SOLC_DIR) allows offline use of preinstalled binaries.
    Otherwise py-solc-x downloads uncached compiler versions.
    """
    location = solc_dir or os.environ.get("SITM_SOLC_DIR")
    directory = Path(location).expanduser() if location is not None else None
    version = (
        str(Version(solc_version.removeprefix("v")))
        if solc_version else _compiler_version(solidity_code, directory)
    )
    filename = "Contract.sol"
    settings = {
        "optimizer": {"enabled": optimize, "runs": optimize_runs},
        "outputSelection": {"*": {"*": ["evm.deployedBytecode.object"]}},
    }
    if evm_version is not None:
        settings["evmVersion"] = evm_version
    # print("solc:", version, "optimizer:", settings["optimizer"])
    # print("source chars:", len(solidity_code), "evm:", evm_version)
    compiled = compile_standard_with_version({
        "language": "Solidity",
        "sources": {filename: {"content": solidity_code}},
        "settings": settings,
    }, version, solc_dir=directory)
    contracts = compiled.get("contracts", {}).get(filename, {})
    # print("编译器", version, "合约", list(contracts))
    runtimes = {name: contract["evm"]["deployedBytecode"]["object"]
                for name, contract in contracts.items()}
    runtimes = {name: runtime for name, runtime in runtimes.items() if runtime}
    # for name, runtime in runtimes.items():
    #     print(name, "runtime bytes:", len(runtime) // 2)
    #     print("runtime head:", runtime[:80])
    if contract_name is None and len(runtimes) == 1:
        contract_name = next(iter(runtimes))
    if contract_name not in runtimes:
        available = ", ".join(sorted(runtimes)) or "none"
        raise ValueError(f"Choose a deployable contract with contract_name (available: {available})")
    try:
        # print("selected contract:", contract_name)
        # assert len(runtimes[contract_name]) % 2 == 0
        return bytes.fromhex(runtimes[contract_name])
    except ValueError as error:
        raise ValueError(f"{contract_name} contains unlinked library references") from error


# 优化器 runs 的备选对照。需要单独看编译参数影响时再调用。
# def compare_optimizer_runs(source, version, contract_name=None):
#     original = compile_with_optimizer(
#         source, False, solc_version=version, contract_name=contract_name
#     )
#     records = []
#     for runs in (1, 50, 200, 1000):
#         optimized = compile_with_optimizer(
#             source, True, runs, solc_version=version, contract_name=contract_name
#         )
#         record = {
#             "runs": runs,
#             "original_bytes": len(original),
#             "optimized_bytes": len(optimized),
#             "OED": calc_opcode_edit_similarity_bc(original, optimized),
#             "SITM": calculate_sitm_new_similarity_bc(original, optimized),
#         }
#         records.append(record)
#     return records


def compile_standard_with_version(
    standard_input: dict, version: str, *, solc_dir: str | Path | None = None
) -> dict:
    """Compile a complete Standard JSON input with the requested solc version."""
    version = str(Version(version.removeprefix("v")))
    location = solc_dir or os.environ.get("SITM_SOLC_DIR")
    if location is not None:
        directory = Path(location).expanduser()
        binary = _compiler_in(directory, version)
    else:
        directory = solcx.get_solcx_install_folder()
        binary = _compiler_in(directory, version)
        if binary is None:
            solcx.install_solc(version)
            binary = _compiler_in(directory, version)
    if binary is None:
        raise FileNotFoundError(f"solc {version} was not found in {directory}")
    return solcx.compile_standard(standard_input, solc_binary=str(binary))


def _edit_similarity(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    scale = max(len(a), len(b))
    return 1.0 - _levenshtein(a, b) / scale if scale else 1.0


def _opcode_names(bytecode: Bytecode) -> list[str]:
    return [ins.name for ins in disassemble(strip_metadata(bytecode))]


def calc_opcode_edit_similarity_bc(bc_a: Bytecode, bc_b: Bytecode) -> float:
    def normalize(ops: list[str]) -> list[str]:
        # print("opcode count:", len(ops), "push count:", sum(op.startswith("PUSH") for op in ops))
        # print("raw opcodes:", ops[:30])
        return ["PUSH" if op.startswith("PUSH") else op for op in ops]

    # a_ops = normalize(_opcode_names(bc_a))
    # b_ops = normalize(_opcode_names(bc_b))
    # print("OED lengths:", len(a_ops), len(b_ops))
    # print("OED distance:", Levenshtein.distance(a_ops, b_ops))
    return _edit_similarity(normalize(_opcode_names(bc_a)), normalize(_opcode_names(bc_b)))


# 保留 PUSH 常量的操作码版本，便于看常量变化被归一化掉了多少。
# def opcode_edit_with_constants(code_a, code_b):
#     left = []
#     right = []
#     for ins in disassemble(strip_metadata(code_a)):
#         left.append((ins.name, ins.immediate))
#     for ins in disassemble(strip_metadata(code_b)):
#         right.append((ins.name, ins.immediate))
#     return _edit_similarity(left, right)
#
# # 例如换一个 slot，带常量和不带常量的结果可以分别看一下。
# # print(opcode_edit_with_constants("600160005500", "600160015500"))


def calc_ngram_similarity_bc(bc_a: Bytecode, bc_b: Bytecode, n: int = 3) -> float:
    if n < 1:
        raise ValueError("n must be positive")

    def features(bytecode: Bytecode) -> Counter:
        ops = _opcode_names(bytecode)
        # print("n:", n, "opcode count:", len(ops), "windows:", max(0, len(ops) - n + 1))
        return Counter(tuple(ops[index:index + n]) for index in range(len(ops) - n + 1))

    a, b = features(bc_a), features(bc_b)
    # print("distinct ngrams:", len(a), len(b))
    # print("left frequent:", a.most_common(5))
    # print("right frequent:", b.most_common(5))
    if not a or not b:
        return float(not a and not b)
    dot = sum(count * b[token] for token, count in a.items())
    norm = math.sqrt(sum(count ** 2 for count in a.values()) * sum(count ** 2 for count in b.values()))
    # print("ngram dot / norm:", dot, norm)
    # assert 0.0 <= dot / norm <= 1.0 + 1e-12
    return dot / norm


# n-gram 的集合写法。集合会丢掉出现次数，可用来对照词频的影响。
# def ngram_set_similarity(code_a, code_b, n=3):
#     if n < 1:
#         raise ValueError("n must be positive")
#     left_ops = _opcode_names(code_a)
#     right_ops = _opcode_names(code_b)
#     left = set()
#     right = set()
#     for index in range(len(left_ops) - n + 1):
#         left.add(tuple(left_ops[index:index + n]))
#     for index in range(len(right_ops) - n + 1):
#         right.add(tuple(right_ops[index:index + n]))
#     union = left | right
#     if not union:
#         return 1.0
#     return len(left & right) / len(union)


def _cfg(analysis: Analysis) -> nx.DiGraph:
    graph = nx.DiGraph()
    for block in analysis.blocks:
        counts = Counter(ins.name for ins in block)
        graph.add_node(block[0].pc, label=tuple(sorted(counts.items())))
    graph.add_edges_from(analysis.edges)
    return graph


def _rooted_neighborhoods(graph: nx.DiGraph) -> dict[tuple, list[nx.DiGraph]]:
    groups = defaultdict(list)
    for root in graph:
        nodes = {root, *graph.predecessors(root), *graph.successors(root)}
        neighborhood = graph.subgraph(nodes).copy()
        neighborhood.nodes[root]["root"] = True
        # print("root:", root, "neighbors:", sorted(nodes))
        # print("local edges:", list(neighborhood.edges()))
        labels = Counter(data["label"] for _, data in neighborhood.nodes(data=True))
        key = (graph.nodes[root]["label"], frozenset(labels.items()), neighborhood.number_of_edges())
        groups[key].append(neighborhood)
    return groups


def calc_cfg_isomorphism_similarity_bc(bc_a: Bytecode, bc_b: Bytecode) -> float:
    """VF2 overlap of rooted one-hop CFG neighborhoods with opcode-multiset labels.

    Section 4.1 specifies approximate VF2 matching but gives no partial-score
    formula. Here each matching neighborhood contributes one matched block.
    """
    code_a, code_b = strip_metadata(bc_a), strip_metadata(bc_b)
    if code_a == code_b:
        return 1.0
    a, b = _cfg(analyze(code_a)), _cfg(analyze(code_b))
    groups_a, groups_b = _rooted_neighborhoods(a), _rooted_neighborhoods(b)
    # print("CFG nodes:", len(a), len(b), "edges:", a.number_of_edges(), b.number_of_edges())
    # print("neighborhood groups:", len(groups_a), len(groups_b))
    node_match = nx.algorithms.isomorphism.categorical_node_match(["label", "root"], [(), False])
    matched = 0
    for key in groups_a.keys() & groups_b.keys():
        candidates = groups_b[key]
        # print("candidate group sizes:", len(groups_a[key]), len(candidates))
        for neighborhood in groups_a[key]:
            for index, candidate in enumerate(candidates):
                if nx.is_isomorphic(neighborhood, candidate, node_match=node_match):
                    matched += 1
                    # print("matched blocks:", matched, "candidate index:", index)
                    candidates.pop(index)
                    break
    # assert matched <= min(len(a), len(b))
    # print("CFG overlap:", matched, "denominator:", max(len(a), len(b)))
    return matched / max(len(a), len(b))


def calc_sf_similarity_bc(
    bc_a: Bytecode, bc_b: Bytecode, *, weights: tuple[float, float, float]
) -> float:
    """Apply SF using coefficients calibrated on the user's validation set."""
    scores = (calc_opcode_edit_similarity_bc(bc_a, bc_b),
              calc_cfg_isomorphism_similarity_bc(bc_a, bc_b),
              calc_ngram_similarity_bc(bc_a, bc_b))
    return _fuse_structural_scores(scores, weights)


def _fuse_structural_scores(scores: tuple[float, float, float], weights: tuple[float, float, float]) -> float:
    if len(weights) != 3 or not all(weight >= 0 for weight in weights) or not math.isclose(sum(weights), 1.0):
        raise ValueError("SF weights must be three nonnegative values that sum to 1 (OED, CFG, NG)")
    return sum(weight * score for weight, score in zip(weights, scores))


def extract_positioned_trace(bytecode: Bytecode, strip_meta: bool = True) -> list[tuple[int, Event]]:
    code = strip_metadata(bytecode) if strip_meta else bytecode_bytes(bytecode)
    return analyze(code).events


def extract_static_interleaved_trace(bytecode: Bytecode, strip_meta: bool = True) -> list[Event]:
    return [event for _, event in extract_positioned_trace(bytecode, strip_meta)]


def weighted_levenshtein(
    trace_a: Sequence[Event], trace_b: Sequence[Event], w_tau: Mapping[str, float] | None = None
) -> float:
    weights = W_TAU if w_tau is None else w_tau
    # print("trace lengths:", len(trace_a), len(trace_b))
    # print("event weights:", dict(weights))
    if trace_a == trace_b:
        return 0.0
    if len(trace_a) < len(trace_b):
        trace_a, trace_b = trace_b, trace_a
    costs_b = [weights[event[0]] for event in trace_b]
    previous = [0.0]
    for cost in costs_b:
        previous.append(previous[-1] + cost)
    # print("DP initial row:", previous[:20])
    # assert len(previous) == len(trace_b) + 1
    for event_a in trace_a:
        cost_a = weights[event_a[0]]
        current = [previous[0] + cost_a]
        for j, (event_b, cost_b) in enumerate(zip(trace_b, costs_b), start=1):
            # if len(trace_a) <= 10 and len(trace_b) <= 10:
            #     print("compare:", event_a, event_b)
            #     print("delete / insert:", previous[j] + cost_a, current[j - 1] + cost_b)
            #     print("substitute:", previous[j - 1] + (0.0 if event_a == event_b else max(cost_a, cost_b)))
            current.append(min(
                previous[j] + cost_a,
                current[j - 1] + cost_b,
                previous[j - 1] + (0.0 if event_a == event_b else max(cost_a, cost_b)),
            ))
        # assert len(current) == len(trace_b) + 1
        previous = current
        # 代价对不上时，可只取十几个事件，逐行看这一行的插入、删除和替换代价。
        # print(event_a, current)
    return previous[-1]


# 核对滚动数组 DP 时可用完整矩阵。只取短轨迹，避免矩阵太大。
# def weighted_distance_matrix(trace_a, trace_b, weights=None):
#     if weights is None:
#         weights = W_TAU
#     height = len(trace_a) + 1
#     width = len(trace_b) + 1
#     matrix = [[0.0] * width for _ in range(height)]
#     for i, event in enumerate(trace_a, 1):
#         matrix[i][0] = matrix[i - 1][0] + weights[event[0]]
#     for j, event in enumerate(trace_b, 1):
#         matrix[0][j] = matrix[0][j - 1] + weights[event[0]]
#     for i, event_a in enumerate(trace_a, 1):
#         for j, event_b in enumerate(trace_b, 1):
#             delete_cost = weights[event_a[0]]
#             insert_cost = weights[event_b[0]]
#             replace_cost = 0.0
#             if event_a != event_b:
#                 replace_cost = max(delete_cost, insert_cost)
#             matrix[i][j] = min(
#                 matrix[i - 1][j] + delete_cost,
#                 matrix[i][j - 1] + insert_cost,
#                 matrix[i - 1][j - 1] + replace_cost,
#             )
#     return matrix[-1][-1], matrix
#
# # distance, matrix = weighted_distance_matrix(trace_a, trace_b)
# # for row in matrix:
# #     print(row)
# # assert abs(distance - weighted_levenshtein(trace_a, trace_b)) < 1e-12


def _trace_score(a: Sequence[Event], b: Sequence[Event], distance: float, max_weight: float) -> float:
    if not a or not b:
        return float(not a and not b)
    return max(0.0, 1.0 - distance / (max(len(a), len(b)) * max_weight))


def apply_soft_gating(
    s_call: float, *, theta_hi: float = THETA_HI, theta_lo: float = THETA_LO, gamma: float = GAMMA
) -> float:
    if s_call >= theta_hi:
        return 0.0
    if s_call <= theta_lo:
        return gamma
    return gamma * (theta_hi - s_call) / (theta_hi - theta_lo)


# 硬门控的对照写法，用来观察阈值两边是否出现明显跳变。
# def hard_call_gate(trace_score, call_score, cutoff=0.7):
#     if call_score < cutoff:
#         return 0.0
#     return trace_score
#
# # for call_score in (0.39, 0.4, 0.55, 0.69, 0.7, 0.71):
# #     soft = 0.9 * (1 - apply_soft_gating(call_score))
# #     hard = hard_call_gate(0.9, call_score)
# #     print(call_score, soft, hard)


def _subsequence(trace: Sequence[Event], kind: str) -> list[Event]:
    return [event for event in trace if event[0] == kind]


@dataclass(frozen=True)
class SITMResult:
    weighted_distance: float
    s_trace: float
    s_call: float
    penalty: float
    similarity: float


def compare_traces(
    trace_a: Sequence[Event], trace_b: Sequence[Event], *, weights: Mapping[str, float] = W_TAU,
    theta_hi: float = THETA_HI, theta_lo: float = THETA_LO, gamma: float = GAMMA
) -> SITMResult:
    distance = weighted_levenshtein(trace_a, trace_b, weights)
    # print("event type counts A:", Counter(event[0] for event in trace_a))
    # print("event type counts B:", Counter(event[0] for event in trace_b))
    s_trace = _trace_score(trace_a, trace_b, distance, max(weights.values()))
    s_call = _edit_similarity(_subsequence(trace_a, "call"), _subsequence(trace_b, "call"))
    penalty = apply_soft_gating(s_call, theta_hi=theta_hi, theta_lo=theta_lo, gamma=gamma)
    # print("weighted distance:", distance, "trace score:", s_trace)
    # print("call score:", s_call, "gate:", penalty, "SITM:", s_trace * (1.0 - penalty))
    # assert 0.0 <= s_trace * (1.0 - penalty) <= 1.0
    return SITMResult(distance, s_trace, s_call, penalty, s_trace * (1.0 - penalty))


def compare_sitm(bc_a: Bytecode, bc_b: Bytecode) -> SITMResult:
    return compare_traces(extract_static_interleaved_trace(bc_a), extract_static_interleaved_trace(bc_b))


def calculate_sitm_trace_similarity_bc(
    bc_a: Bytecode, bc_b: Bytecode, w_tau: Mapping[str, float] | None = None
) -> float:
    weights = W_TAU if w_tau is None else w_tau
    a, b = extract_static_interleaved_trace(bc_a), extract_static_interleaved_trace(bc_b)
    return _trace_score(a, b, weighted_levenshtein(a, b, weights), max(weights.values()))


def calculate_sitm_new_similarity_bc(bc_a: Bytecode, bc_b: Bytecode) -> float:
    return compare_sitm(bc_a, bc_b).similarity


def calculate_sitm_new_similarity(code_a: str, code_b: str, *, optimize: bool = False) -> float:
    bytecode_a = compile_with_optimizer(code_a, optimize)
    bytecode_b = compile_with_optimizer(code_b, optimize)
    return calculate_sitm_new_similarity_bc(bytecode_a, bytecode_b)


def sitm_layerwise_similarity(bc_a: Bytecode, bc_b: Bytecode) -> dict[str, float]:
    a, b = extract_static_interleaved_trace(bc_a), extract_static_interleaved_trace(bc_b)
    return {f"sim_{kind}": _edit_similarity(_subsequence(a, kind), _subsequence(b, kind))
            for kind in W_TAU}


# 单独测试槽位变化时，可以先用这两段短字节码看分数和分层结果。
# code_a = "600160005500"  # 向 slot 0 写入 1
# code_b = "600160015500"  # 向 slot 1 写入 1
# print(extract_static_interleaved_trace(code_a))
# print(extract_static_interleaved_trace(code_b))
# print(compare_sitm(code_a, code_b))
# print(sitm_layerwise_similarity(code_a, code_b))
# 正反各算一次，检查交换输入后的分数。
# forward = calculate_sitm_new_similarity_bc(code_a, code_b)
# backward = calculate_sitm_new_similarity_bc(code_b, code_a)
# assert abs(forward - backward) < 1e-12


def evaluate_pair_all_metrics(
    bc_a: Bytecode, bc_b: Bytecode, *, sf_weights: tuple[float, float, float] | None = None
) -> dict[str, float]:
    scores = {
        "OED": calc_opcode_edit_similarity_bc(bc_a, bc_b),
        "CFG": calc_cfg_isomorphism_similarity_bc(bc_a, bc_b),
        "NG": calc_ngram_similarity_bc(bc_a, bc_b),
        "SITM": calculate_sitm_new_similarity_bc(bc_a, bc_b),
    }
    if sf_weights is not None:
        scores["SF"] = _fuse_structural_scores((scores["OED"], scores["CFG"], scores["NG"]), sf_weights)
    return scores
