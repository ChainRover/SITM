import argparse

from util import (
    calc_cfg_isomorphism_similarity_bc,
    calc_ngram_similarity_bc,
    calc_opcode_edit_similarity_bc,
    calculate_sitm_new_similarity_bc,
    compare_traces,
    compile_with_optimizer,
    extract_positioned_trace,
)


code_a = """\
pragma solidity 0.5.0;

contract Demo {
    mapping(address => uint256) public balances;
    event Withdraw(address indexed user, uint256 amount);

    function withdraw(uint256 amount) external {
        require(balances[msg.sender] >= amount);
        balances[msg.sender] -= amount;                  // SSTORE
        (bool ok, ) = msg.sender.call.value(amount)("");  // CALL
        require(ok);
        emit Withdraw(msg.sender, amount);               // LOG2
    }
}
"""

code_b = """\
pragma solidity 0.8.20;

contract Demo {
    mapping(address => uint256) public balances;
    event Withdraw(address indexed user, uint256 amount);

    function withdraw(uint256 amount) external {
        require(balances[msg.sender] >= amount);
        balances[msg.sender] -= amount;
        (bool ok, ) = payable(msg.sender).call{value: amount}("");
        require(ok);
        emit Withdraw(msg.sender, amount);
    }
}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="store_true", help="Print both SITs and the call penalty")
    parser.add_argument("--solc-dir", help="Directory with preinstalled versioned solc binaries")
    args = parser.parse_args()

    code_a_bc = compile_with_optimizer(code_a, True, 200, solc_dir=args.solc_dir)
    code_b_bc = compile_with_optimizer(code_b, True, 200, solc_dir=args.solc_dir)

    print(f"solc 0.5.0  ({len(code_a_bc)} bytes): 0x{code_a_bc.hex()}")
    print(f"solc 0.8.20 ({len(code_b_bc)} bytes): 0x{code_b_bc.hex()}")
    print("OED:", f"{calc_opcode_edit_similarity_bc(code_a_bc, code_b_bc):.6f}")
    print("CFG:", f"{calc_cfg_isomorphism_similarity_bc(code_a_bc, code_b_bc):.6f}")
    print("NG:", f"{calc_ngram_similarity_bc(code_a_bc, code_b_bc):.6f}")
    print("SITM:", f"{calculate_sitm_new_similarity_bc(code_a_bc, code_b_bc):.6f}")

    if args.trace:
        trace_a = extract_positioned_trace(code_a_bc)
        trace_b = extract_positioned_trace(code_b_bc)
        for version, trace in (("0.5.0", trace_a), ("0.8.20", trace_b)):
            print(f"\nSIT (solc {version}):")
            for pc, event in trace:
                print(f"  0x{pc:04x}  {event}")
        result = compare_traces([event for _, event in trace_a], [event for _, event in trace_b])
        print(f"\nS_trace={result.s_trace:.6f} S_call={result.s_call:.6f} penalty={result.penalty:.6f}")


if __name__ == "__main__":
    main()
