# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

The parent `tinybrains/CLAUDE.md` describes the platform and the other repositories. This file
covers only what is specific to ants-baselines.

## What this repository is

**The platform's own trained entries, and the worked example of how they were trained.** It is a
competitor repository that the platform happens to own: it submits ONNX models and declarative
adapters like anyone else, and it has no special access to anything.

Two axes, and they do not cross cleanly:

- **The class ladder** — one fixed teacher distilled into each of the five weight classes, which is
  how the competition's actual question ("how much play fits in N bytes?") gets a measured answer.
- **The method column** — several learners on one class and one dataset, so the comparison is
  between algorithms rather than between algorithms *and* budgets at once.

`models/<class>-<method>/` holds each finished artifact: `model.onnx`, the generated `adapter.json`,
a `metrics.json` of what the platform said about it, and a `card.md` a person can read.

## Working here needs three sibling checkouts

```text
tinybrains/
  ants-baselines/   <- you are here
  ants/             <- games.toml resolves the cartridge from ../ants
  devops/           <- the `tinybrains` CLI is built from devops/cli
  axon/             <- devops/cli depends on it by path
```

```sh
cargo install --path ../devops/cli    # or: export TINYBRAINS=../devops/cli/target/release/tinybrains
pip install -e .                      # torch, numpy, onnx, pytest
```

## Commands

```sh
pytest tests/ -q                                   # THE test. See "What must stay true".
python -m tb_baselines.adapters > /tmp/a.json      # the generated adapter
python -m tb_baselines.collect --seat-turns 250000 # the teacher dataset (about 9 minutes, 90 MB)
python -m tb_baselines.train.bc --class micro --epochs 6
python -m tb_baselines.export --class micro --weights runs/micro-bc/best.pt --out models/micro-bc
python -m tb_baselines.eval models/micro-bc models/nano-bc --boards 3
```

## The three gates, and only two of them are real

```
tinybrains env       training rollouts    fast; NO deadline, NO strikes, NO adapter
tinybrains <match>   eval.py              the real path, minus admission
tinybrains check     export.py            what the platform will actually decide
```

`tinybrains env` is not the referee. A policy that trains happily can still be struck for missing
the turn clock or refused for an adapter that goes over budget, because none of that exists in the
env. Never report a result from the env as a result.

## What must stay true

- **`planes.py` is the only definition of the encoding, and it is rendered twice.** The numpy
  encoder trains the network; `adapters.py` generates the `adapter.json` the ladder runs. Two
  implementations of one encoding is how a model scores worse in the arena than in training, and it
  fails *silently*. `tests/test_adapter_conformance.py` runs the real evaluator (`tinybrains adapt`,
  which is Axon's dialect interpreter) over the cartridge's reference observations and asserts they
  agree element for element. **It is the most important test here.** It has already caught a plane
  filter inverted and a dilation radius off by five.
- **`adapter.json` is generated and never hand-edited.** Same rule as `ants/build.sh`'s manifests,
  for a sharper reason: editing one rendering of the encoding without the other is exactly the bug
  the conformance test exists to catch, and a hand-edit is how it gets reintroduced.
- **Numbers in `classes.toml` are measured, and the measurement is written down beside them.**
  Nothing there is a guess. When something is re-measured, replace the number *and* its note.
- **The teacher is a label source, not a baseline.** It never ships. The class ladder is only
  meaningful if every class distils the *same* teacher, so changing it invalidates the comparison —
  regenerate the whole dataset and retrain everything, or don't change it.
- **The value head is never exported.** `export.py` takes the trunk and the policy head only. It is
  a real saving (at nano a critic would be a third of the budget) and it is what makes privileged
  input safe: a critic may see the true score because a critic is discarded before anything plays.
- **The engine digest belongs on every artifact.** A model trained against one engine and played
  under another is a model nobody can reproduce, and an engine change is a rules change. The dataset
  header, `metrics.json` and `card.md` all carry it.
- **`data/`, `runs/` and `replays/` are gitignored output.** The dataset is 90 MB and regenerable
  from a seed; a checkpoint is not an artifact. Only `models/` is committed.

## Things that were measured here, and cost time to find

- **fp16 initializers are free capacity.** A `Cast` back to float32 at each use, which ORT
  constant-folds at graph optimisation, so the *file* halves and the runtime does not change:
  **2.03x the parameters for the same weight class**, identical play over 396 per-ant orders, same
  three operators. There is no reason to ship fp32.
- **Above `mini` the turn deadline binds before the byte cap does.** A row owns `turn_ms / rows` of
  a play call — 31.2 ms at Kalam's `wave_k` of 16 — and `small` reaches that at about a quarter of
  its four-megabyte cap. `export.py` refuses an artifact over 70% of a seat's share, which is the
  check that turns this from a paragraph into a gate.
- **`ConvTranspose` is not on the operator allowlist.** `Resize` is the upsampler.
- **The board wraps, and padding costs.** Wrapping before every convolution measured 2.3x the FLOP
  model; one wrap per resolution stage is the same arithmetic for a third of the copying.
- **A batch is per board size.** Three presets are three sizes (64x96, 96x96, 128x128) and one
  tensor cannot hold two. `Step.groups` is that, and `Step.boards` raises rather than silently
  handing back a fraction of the batch.
- **`dilate` scatters from the ants, it does not roll the plane.** The obvious reading of
  `tb.dilate` — shift the board by all 241 disk offsets and OR — was 48% of the training loop at
  1.86 million `np.roll` calls. Walking the disk out from each set cell is the same answer for 8,700
  writes instead of 3.9 million.
