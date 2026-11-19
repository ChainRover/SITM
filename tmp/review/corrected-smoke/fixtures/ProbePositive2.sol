pragma solidity >=0.5.0 <0.9.0;
contract Probe {
    function change() external {
        assembly {
            mstore(0, 0x12345678)
            sstore(0, 11)
            pop(call(gas(), 17, 0, 28, 4, 0, 0))
            pop(call(gas(), 18, 0, 28, 4, 0, 0))
            log1(0, 0, 98)
        }
    }
}
