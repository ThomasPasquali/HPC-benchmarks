from dataclasses import dataclass

# cn-eth       up 6-01:00:00      1  drain cn19
# cn-eth       up 6-01:00:00      2  alloc cn[20-21]
# cn-eth       up 6-01:00:00      7   idle cn[22-28]
# ETH: 19 - 28
# cn-ib*       up 6-01:00:00      2  drain cn[12,15]
# cn-ib*       up 6-01:00:00      5  alloc cn[11,14,16-18]
# cn-ib*       up 6-01:00:00      3   idle cn[09-10,13]
# IB: 09 - 18

@dataclass
class HAICGUNodesMap:
    def get_id(self, node: str) -> int:
        return int(node.split('.')[0][2:])
        
    def get_partition(self, id: int) -> str | None:
        if id >= 9 and id <= 18:
            return 'ib'
        if id >= 19 and id <= 28:
            return 'eth' 
        
    def get_node_distance(self, node1: str, node2: str) -> int:
        id1 = self.get_id(node1)
        id2 = self.get_id(node2)
        part1 = self.get_partition(id1)
        part2 = self.get_partition(id2)
        if part1 != part2:
            return 99999
        if id1 == id2:
            return 0
        return 1