"""Arc-level road graph used by the HMM.

Hidden states are ARCS (directed road segments). Arc a can be followed directly by arc b
when to_node[a] == from_node[b]. `hops(a, b)` is the number of arc-steps needed to go from
a to b along the directed network (0 = same arc, 1 = b directly follows a, ...).
"""
from collections import deque

import numpy as np


class RoadGraph:
    def __init__(self, arcs: np.ndarray):
        """arcs: (n_arcs, 2) int array of [from_node, to_node]; row index = arc_id."""
        arcs = np.asarray(arcs, dtype=np.int64)
        self.from_node = arcs[:, 0]
        self.to_node = arcs[:, 1]
        self.n_arcs = len(arcs)

        # CSR-style lookup: node -> arcs that START at that node.
        order = np.argsort(self.from_node, kind="stable")
        self._sorted_arcs = order
        n_nodes = int(max(self.from_node.max(), self.to_node.max())) + 1
        counts = np.bincount(self.from_node, minlength=n_nodes)
        self._start = np.r_[0, np.cumsum(counts)]
        self._cache: dict[tuple[int, bool], dict[int, int]] = {}

        # reverse[a] = the arc running the opposite way along the same road (-1 if none)
        by_ends = {}
        for a in range(self.n_arcs):
            by_ends.setdefault((int(self.from_node[a]), int(self.to_node[a])), a)
        self.reverse = np.array([by_ends.get((int(self.to_node[a]), int(self.from_node[a])), -1)
                                 for a in range(self.n_arcs)], dtype=np.int64)

    def successors(self, arc: int) -> np.ndarray:
        """Arcs that can directly follow `arc`."""
        n = self.to_node[arc]
        if n + 1 >= len(self._start):
            return np.empty(0, dtype=np.int64)
        return self._sorted_arcs[self._start[n]:self._start[n + 1]]

    def reachable(self, arc: int, max_hops: int, no_uturn: bool = False) -> dict[int, int]:
        """{arc_id: hops} for every arc reachable from `arc` in <= max_hops steps (BFS, cached).

        no_uturn=True never steps from an arc onto its own reverse arc, so reaching the opposite-direction
        twin needs a real loop instead of a free 1-hop turnaround.
        """
        key = (arc, no_uturn)
        hit = self._cache.get(key)
        if hit is not None and hit["_max"] >= max_hops:
            return hit
        dist = {arc: 0}
        q = deque([arc])
        while q:
            a = q.popleft()
            d = dist[a]
            if d == max_hops:
                continue
            for s in self.successors(a):
                s = int(s)
                if no_uturn and s == self.reverse[a]:
                    continue
                if s not in dist:
                    dist[s] = d + 1
                    q.append(s)
        dist["_max"] = max_hops
        self._cache[key] = dist
        return dist

    def hops(self, a: int, b: int, max_hops: int) -> int | None:
        """Directed arc-steps from a to b, or None if not reachable within max_hops."""
        h = self.reachable(a, max_hops).get(b)
        return h if (h is not None and h <= max_hops) else None   # cache may be deeper than asked
