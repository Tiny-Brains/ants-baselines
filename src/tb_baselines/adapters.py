"""`adapter.json`, generated from `planes.py` — never hand-written.

An artifact nobody hand-edits cannot drift from the thing it was made from. That is the same rule
`ants/build.sh` applies to `cartridge.json` and `plugin.json`, and it applies here for a sharper
reason: the adapter is one of the two renderings of the encoding, and the whole anti-skew argument
collapses if someone can edit one rendering without the other.

    python -m tb_baselines.adapters > adapter.json

The bytes are what the platform hashes, so this must be **deterministic**: compact separators,
sorted keys, no trailing newline. Change the spacing and `adapter_hash` moves, which is a new
submission for a document that means the same thing.

## The two programs

`in` stacks `planes.PLANES` into one `board` tensor of `[1, planes, rows, cols]`. The size is read
from the observation rather than written down: three presets mean 64x96, 96x96 and 128x128, and an
adapter that hard-codes one of them fails the other two at admission.

`out` is the reference adapter's, kept verbatim rather than re-derived. It turns a dense policy map
`[1, moves, rows, cols]` into one move per ant by flattening the board, computing each ant's flat
index with a `reduce`, gathering those columns and taking the argmax. It costs about 1,300
operations and it is the fiddliest thing in the dialect — `reduce`'s seed is the only channel from
outer scope into a body, which is why the accumulator has to carry the board width along with the
indices it is building, and why `tb.get` had to exist to get them out again.
"""

from __future__ import annotations

import json
import sys

from .planes import MOVES, N_MOVES, N_PLANES, PLANES, DTYPE

DIALECT = 1


def in_program() -> dict:
    """Observation to `{board: [1, planes, rows, cols]}`."""
    size = {"var": "size"}
    stacked = {"tb.stack": [[p.logic(size) for p in PLANES], 0, DTYPE]}
    # `merge` builds the shape list at run time: [1, planes] ++ size. `tb.reshape` is metadata only
    # and costs 1, so the leading batch axis is free.
    return {"board": {"tb.reshape": [stacked, {"merge": [[1, N_PLANES], size]}]}}


def out_program() -> dict:
    """`{policy: [1, moves, rows, cols]}` to one move per ant, positionally aligned with `mine`."""
    width = {"var": "observation.size.1"}
    cells = {"*": [{"var": "observation.size.0"}, width]}

    # Each ant's flat index, row-major. The accumulator carries `w` because a reduce body sees only
    # the element and the accumulator -- the board width is not otherwise in scope in here.
    flat_indices = {
        "tb.get": [
            {
                "reduce": [
                    {"var": "observation.mine"},
                    {
                        "idx": {
                            "merge": [
                                {"tb.get": [{"var": "accumulator"}, "idx"]},
                                [
                                    {
                                        "+": [
                                            {
                                                "*": [
                                                    {"tb.get": [{"var": "accumulator"}, "w"]},
                                                    {"var": "current.0"},
                                                ]
                                            },
                                            {"var": "current.1"},
                                        ]
                                    }
                                ],
                            ]
                        },
                        "w": {"tb.get": [{"var": "accumulator"}, "w"]},
                    },
                    {"idx": [], "w": width},
                ]
            },
            "idx",
        ]
    }

    per_ant = {
        "tb.transpose": [
            {
                "tb.gather": [
                    {"tb.reshape": [{"var": "outputs.policy"}, [N_MOVES, cells]]},
                    flat_indices,
                    1,
                ]
            },
            [1, 0],
        ]
    }
    return {"map": [{"tb.argmax": [per_ant, 1]}, {"tb.at": [list(MOVES), {"var": ""}]}]}


def adapter() -> dict:
    return {"dialect": DIALECT, "in": in_program(), "out": out_program()}


def dumps() -> str:
    """The exact bytes. Sorted and compact, so the same spec always hashes the same."""
    return json.dumps(adapter(), separators=(",", ":"), sort_keys=True)


def main() -> None:
    sys.stdout.write(dumps())


if __name__ == "__main__":
    main()
