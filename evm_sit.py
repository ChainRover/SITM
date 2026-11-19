from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO

import cbor2
from operator import add, and_, eq, gt, lt, mul, or_, sub, xor

from Crypto.Hash import keccak
from pyevmasm import instruction_tables

##  ----------------------------------------------------------------
Bytecode = bytes | str
Event = tuple[str | int, ...]
CALL_OPS = {"CALL", "CALLCODE", "DELEGATECALL", "STATICCALL"}
VALUE_CALL_OPS = {"CALL", "CALLCODE"}
TERMINATORS = {"JUMP", "JUMPI", "STOP", "RETURN", "REVERT", "INVALID", "SELFDESTRUCT"}
PREDICATES = {"EQ", "LT", "GT", "SLT", "SGT", "ISZERO"}
MASK = (1 << 256) - 1
ADDRESS_MASK = (1 << 160) - 1

OPCODES = {
    inst.opcode: (inst.name, inst.pops, inst.pushes)
    for inst in instruction_tables["istanbul"]
}
OPCODES.update({
    0x44: ("PREVRANDAO", 0, 1),
    0x48: ("BASEFEE", 0, 1),
    0x49: ("BLOBHASH", 1, 1),
    0x4A: ("BLOBBASEFEE", 0, 1),
    0x5C: ("TLOAD", 1, 1),
    0x5D: ("TSTORE", 2, 0),
    0x5E: ("MCOPY", 3, 0),
    0x5F: ("PUSH0", 0, 1),
})


def bytecode_bytes(bytecode: Bytecode) -> bytes:
    if isinstance(bytecode, bytes):
        return bytecode
    if isinstance(bytecode, str):
        value = bytecode.strip()
        if value[:2].lower() == "0x":
            value = value[2:]
        return bytes.fromhex(value)
    raise TypeError("Expected runtime bytecode as bytes or a hex string")


def strip_metadata(bytecode: Bytecode) -> bytes:
    code = bytecode_bytes(bytecode)
    if len(code) < 3:
        return code
    size = int.from_bytes(code[-2:], "big")
    # print("bytecode length:", len(code), "metadata length candidate:", size)
    if not 0 < size <= len(code) - 2:
        return code
    stream = BytesIO(code[-2 - size:-2])
    try:
        metadata = cbor2.CBORDecoder(stream).decode()
        # print("decoded metadata:", metadata)
        # print("CBOR consumed:", stream.tell(), "expected:", size)
    except (cbor2.CBORDecodeError, ValueError, EOFError, OverflowError, RecursionError):
        return code
    if not isinstance(metadata, dict) or stream.tell() != size:
        return code
    hash_sizes = {"ipfs": 34, "bzzr0": 32, "bzzr1": 32}
    for name, length in hash_sizes.items():
        digest = metadata.get(name)
        if isinstance(digest, bytes) and len(digest) == length:
            # print("metadata hash:", name, "removed bytes:", size + 2)
            return code[:-2 - size]
    version = metadata.get("solc")
    if (isinstance(version, bytes) and len(version) == 3) or (
        isinstance(version, str) and version[:1].isdigit() and "." in version
    ):
        return code[:-2 - size]
    return code


@dataclass(frozen=True)
class Instruction:
    pc: int
    name: str
    pops: int
    pushes: int
    immediate: int | None = None

    @property
    def ends_block(self) -> bool:
        return self.name in TERMINATORS or self.name.startswith("UNDEFINED_")


def disassemble(code: bytes) -> list[Instruction]:
    instructions = []
    pc = 0
    while pc < len(code):
        opcode = code[pc]
        if opcode in OPCODES:
            name, pops, pushes = OPCODES[opcode]
        else:
            name, pops, pushes = f"UNDEFINED_{opcode:02X}", 0, 0
        width = opcode - 0x5F if 0x60 <= opcode <= 0x7F else 0
        # if name.startswith("UNDEFINED"):
        #     print("undefined opcode:", hex(pc), hex(opcode))
        immediate = None
        if 0x5F <= opcode <= 0x7F:
            data = code[pc + 1:pc + 1 + width]
            # EVM pads a terminal PUSH immediate with zeros on the right.
            immediate = int.from_bytes(data.ljust(width, b"\0"), "big")
            # if len(data) != width:
            #     print("truncated PUSH:", hex(pc), name, data.hex(), "width:", width)
        instructions.append(Instruction(pc, name, pops, pushes, immediate))
        # print(hex(pc), name, "immediate:", immediate, "pop/push:", pops, pushes)
        pc += 1 + width
    return instructions


# 测试 PUSH 数据里出现 f1 的情况，反汇编结果里应当只有一次 SSTORE。
# code = bytes.fromhex("60f160005500")
# instructions = disassemble(code)
# print([(ins.pc, ins.name, ins.immediate) for ins in instructions])
# assert [ins.name for ins in instructions] == ["PUSH1", "PUSH1", "SSTORE", "STOP"]


# 保留一份 PC 到指令的映射，核对跳转位置和 PUSH 宽度时比较方便。
# def instruction_position_map(code):
#     mapping = {}
#     for ins in disassemble(bytecode_bytes(code)):
#         mapping[ins.pc] = {
#             "name": ins.name,
#             "immediate": ins.immediate,
#             "pops": ins.pops,
#             "pushes": ins.pushes,
#         }
#     return mapping
#
# # positions = instruction_position_map("61010260005500")
# # for pc, instruction in positions.items():
# #     print(hex(pc), instruction)


def split_blocks(instructions: list[Instruction]) -> list[list[Instruction]]:
    blocks = []
    start = 0
    for index in range(1, len(instructions)):
        if instructions[index].name == "JUMPDEST" or instructions[index - 1].ends_block:
            blocks.append(instructions[start:index])
            start = index
    if instructions:
        blocks.append(instructions[start:])
    return blocks


@dataclass(frozen=True)
class Slot:
    # The keccak access pattern can be known even when its base value is not.
    base: int | None
    pattern: str


@dataclass(frozen=True)
class Word:
    constant: int | None = None
    slot: Slot | None = None
    predicate: str | None = None


UNKNOWN = Word()


class Memory:
    """Track byte writes and exact word identities within the current block."""

    def __init__(self) -> None:
        self.cells: dict[int, int | None] = {}
        self.words: dict[int, Word] = {}

    def invalidate(self, offset: int | None, size: int | None) -> None:
        if size == 0:
            return
        if offset is None:
            self.cells.clear()
            self.words.clear()
            return
        stop = offset + size if size is not None else None
        for mapping, width in ((self.cells, 1), (self.words, 32)):
            start = max(0, offset - width + 1)
            if stop is not None and stop - start <= len(mapping):
                for pos in range(start, stop):
                    mapping.pop(pos, None)
            else:
                for pos in list(mapping):
                    if pos >= start and (stop is None or pos < stop):
                        del mapping[pos]

    def store(self, offset: int | None, value: Word, size: int) -> None:
        self.invalidate(offset, size)
        if offset is None:
            return
        if value.constant is None:
            data = (None,) * size
        else:
            data = (value.constant & ((1 << (8 * size)) - 1)).to_bytes(size, "big")
        self.cells.update(enumerate(data, offset))
        if size == 32:
            self.words[offset] = value if value is not UNKNOWN else Word()

    def read(self, offset: int, size: int) -> bytes | None:
        data = [self.cells.get(pos) for pos in range(offset, offset + size)]
        return bytes(data) if None not in data else None

    def load_word(self, offset: int | None) -> Word:
        if offset is None:
            return Word()
        if offset not in self.words:
            data = self.read(offset, 32)
            self.words[offset] = Word(int.from_bytes(data, "big") if data is not None else None)
        return self.words[offset]

    def hash(self, offset: int | None, size: int | None) -> Word:
        if size == 0:
            return Word(int.from_bytes(keccak.new(digest_bits=256, data=b"").digest(), "big"))
        # Solidity layouts: keccak256(key || base), keccak256(base) + index.
        if offset is None or size not in (32, 64):
            return Word()
        if not all(pos in self.cells for pos in range(offset, offset + size)):
            return Word()
        base_word = self.load_word(offset + size - 32)
        slot = base_word.slot
        # Old: require a constant base before retaining the access pattern.
        # Keep these two independently observable fields separate for Table 1.
        if slot is None:
            slot = Slot(base_word.constant, "mapping" if size == 64 else "array")
        data = self.read(offset, size)
        digest = keccak.new(digest_bits=256, data=data).digest() if data is not None else None
        return Word(int.from_bytes(digest, "big") if digest else None, slot)


def signed(value: int) -> int:
    return value if value < 1 << 255 else value - (1 << 256)


def signed_div(a: int, b: int) -> int:
    a, b = signed(a), signed(b)
    return (abs(a) // abs(b)) * (-1 if (a < 0) != (b < 0) else 1) if b else 0


def signed_mod(a: int, b: int) -> int:
    a, b = signed(a), signed(b)
    return (abs(a) % abs(b)) * (-1 if a < 0 else 1) if b else 0


def signextend(byte: int, value: int) -> int:
    if byte >= 32:
        return value
    sign = 1 << (byte * 8 + 7)
    return (value & (sign - 1)) - (value & sign)


FOLD = {
    "ADD": add, "MUL": mul, "SUB": sub,
    "DIV": lambda a, b: a // b if b else 0,
    "SDIV": signed_div,
    "MOD": lambda a, b: a % b if b else 0,
    "SMOD": signed_mod,
    "ADDMOD": lambda a, b, m: (a + b) % m if m else 0,
    "MULMOD": lambda a, b, m: (a * b) % m if m else 0,
    "EXP": lambda a, b: pow(a, b, 1 << 256),
    "SIGNEXTEND": signextend,
    "LT": lt, "GT": gt, "EQ": eq,
    "SLT": lambda a, b: signed(a) < signed(b),
    "SGT": lambda a, b: signed(a) > signed(b),
    "ISZERO": lambda a: a == 0,
    "AND": and_, "OR": or_, "XOR": xor, "NOT": lambda a: ~a,
    "BYTE": lambda i, x: (x >> (8 * (31 - i))) & 255 if i < 32 else 0,
    "SHL": lambda shift, x: x << shift if shift < 256 else 0,
    "SHR": lambda shift, x: x >> min(shift, 256),
    "SAR": lambda shift, x: signed(x) >> min(shift, 256),
}


def fold(name: str, operands: tuple[Word, ...]) -> Word:
    constants = tuple(word.constant for word in operands)
    # print("fold operands:", name, constants)
    value = None
    if all(constant is not None for constant in constants):
        value = FOLD[name](*constants) & MASK
    elif name in {"SUB", "XOR", "EQ"}:
        left, right = operands
        if left is right and left is not UNKNOWN:
            value = int(name == "EQ")
    slot = None
    if name == "ADD":
        slots = [word.slot for word in operands if word.slot is not None]
        if len(slots) == 1 and slots[0].pattern == "array":
            slot = slots[0]
    # if slot is not None:
    #     print("propagated slot:", name, slot.base, slot.pattern)
    # print("fold result:", value)
    return Word(value, slot, name if name in PREDICATES else None)


class BlockState:
    def __init__(self) -> None:
        self.stack: list[Word] = []
        self.memory = Memory()

    def top(self, depth: int) -> Word:
        return self.stack[-depth] if len(self.stack) >= depth else UNKNOWN

    def step(self, ins: Instruction) -> None:
        name = ins.name
        # print("before step:", hex(ins.pc), name, "stack size:", len(self.stack))
        # print("stack top:", [word.constant for word in self.stack[-8:]])
        if ins.immediate is not None:
            self.stack.append(Word(ins.immediate))
            return
        if name.startswith("DUP"):
            # print("DUP source:", name, self.top(int(name[3:])).constant)
            self.stack.append(self.top(int(name[3:])))
            return
        if name.startswith("SWAP"):
            depth = int(name[4:]) + 1
            # print("SWAP depth:", depth, "available:", len(self.stack))
            if len(self.stack) < depth:
                self.stack[:0] = [UNKNOWN] * (depth - len(self.stack))
            self.stack[-1], self.stack[-depth] = self.stack[-depth], self.stack[-1]
            return

        operands = tuple(self.top(i) for i in range(1, ins.pops + 1))
        # print("popped values:", [word.constant for word in operands])
        # if len(self.stack) < ins.pops:
        #     print("unknown incoming stack words:", ins.pops - len(self.stack))
        if ins.pops:
            del self.stack[-ins.pops:]
        if name in FOLD:
            self.stack.append(fold(name, operands))
        elif name == "PC":
            self.stack.append(Word(ins.pc))
        elif name == "MLOAD":
            self.stack.append(self.memory.load_word(operands[0].constant))
        elif name in {"MSTORE", "MSTORE8"}:
            # print("memory write:", name, "offset:", operands[0].constant, "value:", operands[1].constant)
            self.memory.store(operands[0].constant, operands[1], 32 if name == "MSTORE" else 1)
        elif name == "SHA3":
            self.stack.append(self.memory.hash(*(word.constant for word in operands)))
        else:
            if name in CALL_OPS:
                # print("invalidate call output:", operands[-2].constant, operands[-1].constant)
                self.memory.invalidate(operands[-2].constant, operands[-1].constant)
            elif name == "EXTCODECOPY":
                self.memory.invalidate(operands[1].constant, operands[3].constant)
            elif name in {"CALLDATACOPY", "CODECOPY", "RETURNDATACOPY", "MCOPY"}:
                self.memory.invalidate(operands[0].constant, operands[2].constant)
            self.stack.extend(Word() for _ in range(ins.pushes))
        # print("after step:", name, "stack size:", len(self.stack))

    def event(self, name: str) -> Event | None:
        if name in CALL_OPS:
            target = self.top(2).constant
            has_value = name in VALUE_CALL_OPS
            offset = self.top(4 if has_value else 3).constant
            length = self.top(5 if has_value else 4).constant
            value = self.top(3).constant if has_value else 0
            selector = None
            if offset is not None and length is not None and length >= 4:
                selector = self.memory.read(offset, 4)
            # print("CALL kind:", name, "target:", target, "value:", value)
            # print("CALL input:", offset, length, "selector:", selector.hex() if selector is not None else None)
            # print("CALL stack:", [self.top(i).constant for i in range(1, 8 if has_value else 7)])
            return (
                "call", name,
                f"0x{target & ADDRESS_MASK:040x}" if target is not None else "[DYN_ADDR]",
                "0x" + selector.hex() if selector is not None else "[DYN_SEL]",
                length if length is not None else "[DYN_LEN]",
                int(value != 0) if value is not None else "[DYN_VAL]",
            )
        if name in {"SLOAD", "SSTORE"}:
            word = self.top(1)
            # print("storage key:", name, word.constant, "derived slot:", word.slot)
            # if name == "SSTORE":
            #     print("stored value:", self.top(2).constant)
            if word.slot is not None:
                base = hex(word.slot.base) if word.slot.base is not None else "[DYN_SLOT]"
                return ("store", name, base, word.slot.pattern)
            if word.constant is not None:
                return ("store", name, hex(word.constant), "direct")
            return ("store", name, "[DYN_SLOT]", "[DYN]")
        if name.startswith("LOG"):
            count = int(name[3:])
            topic = self.top(3).constant
            length = self.top(2).constant
            topic_field = "N/A"
            if count:
                topic_field = f"0x{topic:064x}" if topic is not None else "[DYN_TOPIC]"
            # print("LOG fields:", name, "topics:", count, "topic0:", topic_field, "data bytes:", length)
            return ("log", name, count, topic_field, length if length is not None else "[DYN_DLEN]")
        if name in {"REVERT", "INVALID"}:
            # Old: return ("exc", name, "unconditional") for unknown guards too.
            # analyze() upgrades this only when control-flow evidence is available.
            return ("exc", name, "[DYN_COND]")
        if name == "RETURN":
            length = self.top(2).constant
            # print("RETURN offset / length:", self.top(1).constant, length)
            return ("ret", name, length if length is not None else "[DYN_LEN]")
        return None


# 逐条查看块内栈的临时写法。这里打印的是静态回溯状态。
# def inspect_block_stack(code):
#     instructions = disassemble(bytecode_bytes(code))
#     found = []
#     for block in split_blocks(instructions):
#         state = BlockState()
#         print("block", hex(block[0].pc))
#         for ins in block:
#             before = [word.constant for word in state.stack[-4:]]
#             event = state.event(ins.name)
#             state.step(ins)
#             after = [word.constant for word in state.stack[-4:]]
#             print(hex(ins.pc), ins.name, before, "->", after)
#             if event is not None:
#                 found.append((ins.pc, event))
#                 print("event", event)
#     return found


@dataclass
class Analysis:
    blocks: list[list[Instruction]]
    edges: set[tuple[int, int]]
    events: list[tuple[int, Event]]


def analyze(code: bytes) -> Analysis:
    blocks = split_blocks(disassemble(code))
    destinations = {block[0].pc for block in blocks if block[0].name == "JUMPDEST"}
    # print("basic blocks:", len(blocks), "jump destinations:", len(destinations))
    incoming: dict[int, set[str | None]] = defaultdict(set)
    edges = set()
    events_by_block = []
    for index, block in enumerate(blocks):
        state = BlockState()
        events = []
        # print("block:", index, hex(block[0].pc), "instructions:", len(block))
        # print("block tail:", hex(block[-1].pc), block[-1].name)
        for ins in block[:-1]:
            event = state.event(ins.name)
            # 核对字段来源时，连同事件发生前的栈顶常量一起打印。
            # print(hex(ins.pc), ins.name, state.top(1).constant, event)
            if event is not None:
                events.append((ins.pc, event))
            state.step(ins)

        tail = block[-1]
        event = state.event(tail.name)
        # print("tail event:", hex(tail.pc), event)
        if event is not None:
            events.append((tail.pc, event))
        events_by_block.append(events)

        successors = []
        condition = state.top(2).predicate if tail.name == "JUMPI" else None
        if tail.name in {"JUMP", "JUMPI"}:
            target = state.top(1).constant
            # print("jump target:", target, "resolved:", target in destinations, "condition:", condition)
            if target in destinations:
                successors.append(target)
        if index + 1 < len(blocks) and (tail.name == "JUMPI" or not tail.ends_block):
            successors.append(blocks[index + 1][0].pc)
        # print("successors:", hex(block[0].pc), [hex(pc) for pc in successors])
        for successor in successors:
            edges.add((block[0].pc, successor))
            incoming[successor].add(condition)

    # A path from entry containing no conditional or unresolved jump proves the
    # exception is unconditional. An absent/unrecognized guard alone does not.
    unconditional = set()
    successors_by_pc = defaultdict(list)
    for source, target in edges:
        successors_by_pc[source].append(target)
    blocks_by_pc = {block[0].pc: block for block in blocks}
    pending = [blocks[0][0].pc] if blocks else []
    while pending:
        start = pending.pop()
        if start in unconditional:
            continue
        unconditional.add(start)
        if blocks_by_pc[start][-1].name != "JUMPI":
            pending.extend(successors_by_pc[start])

    trace = []
    for block, events in zip(blocks, events_by_block):
        guards = incoming[block[0].pc]
        condition = next(iter(guards)) if len(guards) == 1 else None
        # print("incoming guards:", hex(block[0].pc), guards, "selected:", condition)
        for pc, event in events:
            if event[0] == "exc":
                guard = "unconditional" if block[0].pc in unconditional else condition
                event = ("exc", event[1], guard if guard is not None else "[DYN_COND]")
            trace.append((pc, event))
            # print("trace event:", hex(pc), event)
    # print("analysis totals:", len(blocks), "blocks,", len(edges), "edges,", len(trace), "events")
    return Analysis(blocks, edges, trace)

# 按静态 CFG 边筛可达事件的备选写法。动态跳转需要另外核对。
# def reachable_trace_by_static_edges(code):
#     result = analyze(bytecode_bytes(code))
#     if not result.blocks:
#         return []
#     successors = defaultdict(list)
#     for source, target in result.edges:
#         successors[source].append(target)
#     pending = [result.blocks[0][0].pc]
#     visited = set()
#     while pending:
#         current = pending.pop()
#         if current in visited:
#             continue
#         visited.add(current)
#         pending.extend(successors[current])
#     block_for_pc = {}
#     for block in result.blocks:
#         for ins in block:
#             block_for_pc[ins.pc] = block[0].pc
#     selected = []
#     for pc, event in result.events:
#         if block_for_pc[pc] in visited:
#             selected.append((pc, event))
#     return selected
