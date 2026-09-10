"""The scripted teacher: a potential field over the board, one BFS a turn.

**It is not a baseline and must never be shipped as one.** It is a *label source*. The whole point
of the class ladder is to take one fixed teacher and ask how much of it fits in 8 KiB, in 64 KiB, in
512 KiB — and that question is only meaningful if the teacher is the same for every class and if it
is a player rather than a network. Distilling a heuristic is also the fastest honest way to get a
ladder something to play against; `axon`'s untrained fixtures hold still and teach nobody anything.

It is a *player*, not a second engine. It reads the observation the protocol defines and answers a
move — the same thing a competitor's model does. It implements no rule: it never decides who dies,
what a hill is worth, or when a match ends. `ants/CLAUDE.md`'s prohibition is on a second
implementation of the *rules*, and this has none.

## How it plays

One multi-source BFS a turn, over the cells this seat knows are not water, from every target at
once: food, enemy hills, and the frontier of what has never been seen. Each ant then steps down the
resulting distance field. That is the classic Ants opening bot and it is deliberately ordinary —
the interesting question here is what a network retains of it, not how strong it is.

Three refinements, each of which changes the games it produces rather than its style:

* **Targets are weighted by a head start**, not by a multiplier. Food seeds the queue at distance 0,
  an enemy hill at `-hill_lead`, so hills win ties out to that many steps. A multiplier on a
  distance field would break the BFS's monotonicity and is the usual bug here.
* **No two ants take one square.** They would both die (`ants/src/turn.rs`: every ant finishing on a
  shared square dies, your own included), which is the single cheapest way to lose a colony.
* **A share of ants scout.** With `explore` at 0, a colony that has eaten everything it can see sits
  still and the match ends `idle_food` — which is exactly what the untrained fixtures do.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np

from .planes import MOVES

# Row-major deltas in the order `MOVES` names them, so an index into one indexes the other.
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOLD = MOVES.index("-")

FAR = np.int32(1 << 29)


class Teacher:
    """One instance per process; it holds no state between turns and is safe to reuse."""

    def __init__(self, hill_lead: int = 6, explore: float = 0.25, max_depth: int = 32,
                 seed: int = 0):
        self.hill_lead = hill_lead
        self.explore = explore
        # An ant decides one step, so a target sixty cells away and a target thirty cells away lead
        # to the same move. Capping the flood is therefore free in play quality and is most of the
        # run time: uncapped, the frontier seeding makes almost every cell of the board get visited
        # every seat-turn, in pure Python.
        self.max_depth = max_depth
        self.rng = random.Random(seed)

    # ---- the field ------------------------------------------------------------------

    def field(self, obs: dict, water: np.ndarray, seen: np.ndarray) -> np.ndarray:
        """Distance from the nearest target, over known-passable cells, wrapping.

        Unseen cells are passable *and* are targets: that is what makes one field drive foraging and
        exploration together, and why a colony with nothing in sight still spreads out rather than
        stalling into `idle_food`.

        **A numpy wavefront, not a queue.** The Python BFS this replaces was 93% of the training
        loop after `dilate` was fixed -- 11.6 million deque operations in a two-minute profile,
        because seeding the frontier makes almost every cell of the board a target. Expanding the
        whole ring at once is the same flood in `max_depth` array operations instead of one Python
        iteration per cell visited.

        Levels are offset so the earliest seed is 0: an enemy hill starts at 0, food at `hill_lead`,
        the frontier two later. Only differences are ever compared, so the offset is free -- and a
        head start is the right way to prefer a target, where a multiplier on a distance field
        breaks the flood's monotonicity and is the usual bug here.
        """
        rows, cols = obs["size"]
        dist = np.full((rows, cols), FAR, dtype=np.int32)
        passable = ~water

        at_level: dict[int, np.ndarray] = {}

        def seed(mask: np.ndarray, level: int) -> None:
            if level in at_level:
                at_level[level] |= mask
            else:
                at_level[level] = mask.copy()

        hills = np.zeros((rows, cols), dtype=bool)
        for r, c, owner in obs["hills"]:
            if owner != 0:
                hills[r, c] = True
        if hills.any():
            seed(hills, 0)

        if obs["food"]:
            food = np.zeros((rows, cols), dtype=bool)
            idx = np.asarray(obs["food"], dtype=np.int64)
            food[idx[:, 0], idx[:, 1]] = True
            seed(food, self.hill_lead)

        unseen = ~seen
        if unseen.any():
            # Every unseen cell, not a sample of them: with the flood vectorised there is no reason
            # to approximate the frontier, and a sample gave the colony a lumpy idea of where the
            # map ran out.
            seed(unseen & passable, self.hill_lead + 2)

        if not at_level:
            return dist

        wave = np.zeros((rows, cols), dtype=bool)
        for depth in range(0, self.max_depth + 1):
            step = at_level.pop(depth, None)
            if wave.any():
                spread = (
                    np.roll(wave, 1, 0) | np.roll(wave, -1, 0)
                    | np.roll(wave, 1, 1) | np.roll(wave, -1, 1)
                )
                step = spread if step is None else (spread | step)
            if step is None:
                continue
            step &= passable & (dist == FAR)
            if not step.any() and not at_level:
                break
            dist[step] = depth
            wave = step
        return dist

    # ---- the move -------------------------------------------------------------------

    def orders(self, obs: dict, water: np.ndarray, seen: np.ndarray) -> np.ndarray:
        """One move index an ant, positionally aligned with `mine`."""
        mine = obs["mine"]
        if not mine:
            return np.zeros(0, dtype=np.int64)

        rows, cols = obs["size"]
        dist = self.field(obs, water, seen)
        # Food blocks movement as surely as water does (`ants/src/turn.rs`), so an ant ordered onto
        # food simply stays -- which wastes the turn and, worse, teaches the network that it does not.
        blocked = np.zeros((rows, cols), dtype=bool)
        for r, c in obs["food"]:
            blocked[r, c] = True

        taken = {tuple(a) for a in mine}
        out = np.full(len(mine), HOLD, dtype=np.int64)

        # Nearest-first, so the ant with the best claim on a square gets it.
        order = sorted(range(len(mine)), key=lambda i: dist[mine[i][0], mine[i][1]])
        for i in order:
            ar, ac = mine[i]
            best, best_d = HOLD, dist[ar, ac]
            scout = self.rng.random() < self.explore
            options = list(range(4))
            self.rng.shuffle(options)                    # break ties without a directional bias
            for k in options:
                dr, dc = DELTAS[k]
                nr, nc = (ar + dr) % rows, (ac + dc) % cols
                if water[nr, nc] or blocked[nr, nc] or (nr, nc) in taken:
                    continue
                d = dist[nr, nc]
                if d < best_d or (scout and best == HOLD and d < FAR):
                    best, best_d = k, d
            if best != HOLD:
                dr, dc = DELTAS[best]
                nr, nc = (ar + dr) % rows, (ac + dc) % cols
                taken.discard((ar, ac))
                taken.add((nr, nc))
            out[i] = best
        return out


def known(obs: dict) -> tuple[np.ndarray, np.ndarray]:
    """Water, and what has ever been seen, from one observation.

    `water` is the only field with memory, so a 1 is water you have seen and a 0 is *either* known
    land or a cell you have never looked at. The teacher needs those apart, and the honest way to
    get it from a single observation is the visibility disk around your own ants — which is the same
    thing plane 6 computes, and the same reason it exists.
    """
    from .planes import VIEW_RADIUS2, Board

    rows, cols = obs["size"]
    b = Board(rows, cols)
    water = b.rle(obs["water"]["rle"]).astype(bool)
    # A cell is "seen" if it is known water or is visible now. This under-counts -- ground you
    # walked over an hour ago reads as unseen -- and that is the right way to be wrong here: it
    # keeps the colony exploring rather than declaring the map finished.
    visible = b.dilate(b.scatter(obs["mine"]), VIEW_RADIUS2).astype(bool)
    return water, water | visible
