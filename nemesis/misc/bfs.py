from typing import Dict, List
from collections import deque

import logging
from uuid import UUID

logger = logging.getLogger(__name__)


def bfs(inputs_map: Dict[UUID, List[UUID]]) -> List[UUID]:
    result: List[UUID] = []

    initial = UUID(int=0)

    q: deque = deque()

    visited: Dict[UUID, bool] = {x: False for x in inputs_map.keys()}

    q.append(initial)

    while q:
        curr = q.popleft()

        if visited[curr]:
            continue

        result.append(curr)

        visited[curr] = True

        for x in inputs_map[curr]:
            q.append(x)

    return result
