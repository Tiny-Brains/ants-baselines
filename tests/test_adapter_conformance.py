"""The trainer's encoder and the ladder's adapter must produce the same tensor. Byte for byte.

**This is the most important test in the repository.** Everything else here is a training script
that can be wrong in ways a metric will show you. This one guards a failure that shows you nothing:
an encoder that disagrees with the adapter trains a model on a distribution the arena never serves,
and the only symptom is a rating lower than the training curve promised.

It is not a re-implementation checking itself. `tinybrains adapt` runs **Axon's own dialect
evaluator** — the same code path, the same `evaluator_digest`, that admission and every match run —
over the cartridge's own reference observations, which is the only set on which agreement decides
anything. If this passes, `planes.encode` and `adapter.json` are the same function on the inputs the
platform actually validates against.

    pytest tests/ -q                    # needs `tinybrains` on PATH, or TINYBRAINS set
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tb_baselines import adapters, planes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CLI = os.environ.get("TINYBRAINS") or shutil.which("tinybrains")


@pytest.fixture(scope="session")
def dumped(tmp_path_factory):
    """The ladder's own answer: `adapter.json` through Axon, over the reference observations."""
    if not CLI:
        pytest.skip("no `tinybrains` on PATH; set TINYBRAINS to the binary")
    out = tmp_path_factory.mktemp("tensors")
    adapter = out / "adapter.json"
    adapter.write_text(adapters.dumps())
    r = subprocess.run(
        [CLI, "adapt", str(adapter), "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert r.returncode == 0, f"tinybrains adapt failed:\n{r.stdout}\n{r.stderr}"
    return json.loads((out / "manifest.json").read_text()), out


def test_every_reference_observation_encodes_identically(dumped):
    manifest, out = dumped
    assert manifest["cases"], "the cartridge shipped no reference observations"

    for case in manifest["cases"]:
        i = case["case"]
        obs = json.loads((out / f"case-{i}" / "observation.json").read_text())
        theirs = np.load(out / f"case-{i}" / "board.npy")
        ours = planes.encode(obs)

        assert ours.shape == theirs.shape, (
            f"case {i}: the trainer builds {ours.shape}, the adapter builds {theirs.shape}"
        )
        assert ours.dtype == theirs.dtype, f"case {i}: {ours.dtype} vs {theirs.dtype}"

        if not np.array_equal(ours, theirs):
            # Name the plane rather than the cell. A whole plane is almost always what is wrong,
            # and "plane 4 (hill_mine) differs in 2 cells" is a diagnosis where a flat index is a
            # puzzle.
            bad = [
                f"{k} ({planes.PLANES[k].name}): {int((ours[0, k] != theirs[0, k]).sum())} cells"
                for k in range(planes.N_PLANES)
                if not np.array_equal(ours[0, k], theirs[0, k])
            ]
            pytest.fail(f"case {i} disagrees on plane(s) -- " + "; ".join(bad))


def test_the_adapter_fits_the_operation_budget(dumped):
    manifest, _ = dumped
    budget = manifest["budget_ops"]
    worst = max(c["ops_in"] for c in manifest["cases"])
    # The worst case is what admission refuses, so the mean is not the number to watch. A comfortable
    # margin, not a passing grade: an observation busier than any of these still has to fit.
    assert worst < budget * 0.6, (
        f"the `in` program costs {worst} of {budget} at its worst -- too little headroom for a "
        f"board busier than the reference set"
    )


def test_the_generated_adapter_is_byte_stable():
    """The platform hashes these bytes, so the same spec must always produce the same document."""
    assert adapters.dumps() == adapters.dumps()
    doc = json.loads(adapters.dumps())
    assert doc["dialect"] == 1
    assert set(doc) == {"dialect", "in", "out"}
    assert set(doc["in"]) == {"board"}, "the graph declares one input, named `board`"


def test_the_encoder_handles_an_empty_colony():
    """Zero ants is always valid (`ants/docs/protocol.md` §1) and is what a wiped-out seat sends
    right up until its match ends. An encoder that indexes into an empty list dies there."""
    obs = {
        "size": [64, 96],
        "mine": [], "foes": [], "food": [], "hills": [],
        "water": {"rle": [0, 64 * 96]},
    }
    board = planes.encode(obs)
    assert board.shape == (1, planes.N_PLANES, 64, 96)
    assert board.sum() == 0, "nothing seen, nothing set -- including the visibility mask"
