from typing import List, Protocol

class ParityOracle(Protocol):
    async def get_parity(self, indices: List[int]) -> int:
        ...
    
    async def get_parities(self, blocks: List[List[int]]) -> List[int]:
        ...
