# ants-baselines

**The platform's trained entries for TinyBrains Ants, and the worked example of how each was
trained.** Every model here is a real submission: an ONNX graph and a declarative adapter, measured
into a weight class by the same code that measures yours.

It exists because a ladder needs opponents before it needs interesting ones, and because the
competition's actual question — *how much play fits in eight kilobytes?* — deserves a measured
answer rather than an argument.

## Scope

Two axes, and the honest thing to say about them is that they do not cross cleanly.

**The class ladder** takes one fixed teacher and distils it into each of the five weight classes.
Same data, same labels, same loss; only the budget changes. That is the size/fidelity curve, and it
is the reason the classes exist.

**The method column** takes one class and one dataset and changes only the learner. That is the
comparison worth making — a per-cell model, a convolutional trunk and a reinforcement learner on
identical data say something about the algorithms; the same three at different budgets say nothing
about either.

The control in that column is worth naming: `percell` is 1x1 convolutions only, matched to the
convolutional trunk's parameter count to within 1%, so it has the same capacity and **no receptive
field at all**. The gap between them is what looking around is worth in this game, measured rather
than asserted.

They do not cross because compute does not scale with bytes. Above `mini` the **turn deadline**
binds long before the byte cap does, and `large`'s cap is out of reach of any dense architecture.
That is the interesting fact about the top of the ladder, not a defect in it — see
[`classes.toml`](classes.toml), which carries every number and the measurement behind it.

## Where it sits

A competitor repository that the platform happens to own. It has no special access: it reads the
cartridge from a sibling checkout, encodes observations through the published adapter dialect, and
is admitted through the same gate as anyone else.

```text
tinybrains/
  ants-baselines/   <- you are here
  ants/             <- the cartridge; games.toml resolves it from ../ants
  devops/           <- the `tinybrains` CLI, which is the whole toolchain
  axon/             <- devops/cli depends on it by path
```

## Interface

What this repository produces, and what reads it:

| Artifact | Read by |
|---|---|
| `models/<class>-<method>/model.onnx` + `adapter.json` | a GitHub release; admission fetches both |
| `models/<class>-<method>/metrics.json` | the seeding script, for `size_bytes`, `param_count`, `infer_us` |
| `models/<class>-<method>/card.md` | people |

## Run it, test it

```sh
cargo install --path ../devops/cli     # or export TINYBRAINS=../devops/cli/target/release/tinybrains
pip install -e '.[dev]'

pytest tests/ -q                                    # the conformance gate; see below
python -m tb_baselines.collect --seat-turns 250000  # the teacher dataset, ~9 min, ~90 MB
python -m tb_baselines.train.bc --class micro --epochs 6
python -m tb_baselines.export --class micro --weights runs/micro-bc/best.pt --out models/micro-bc
python -m tb_baselines.eval models/micro-bc models/nano-bc --boards 3
```

### The one test that matters

A competitor training in Python encodes each observation twice: once in `adapter.json`, which is
what the ladder runs, and once in numpy, which is what the optimiser sees. When those two disagree,
both halves work and the model simply scores worse in the arena than its training curve promised.
Nothing tells you.

So the encoding is declared once, in [`planes.py`](src/tb_baselines/planes.py), with both renderings
side by side — and `tests/test_adapter_conformance.py` runs **Axon's own dialect evaluator** over the
cartridge's own reference observations and asserts they agree element for element. If you take one
idea from this repository, take that one.

## What a deployment owes it

Nothing at run time. These are ordinary submissions.

At seed time, `devops/compose/db-init/30-seed.sql` names this repository and a release tag, and the
seeding script needs each artifact's `metrics.json` for the values admission would otherwise have
measured.

## Layout

```text
classes.toml                    the class table: every number, and its measurement
src/tb_baselines/
  planes.py                     THE encoding, rendered twice
  adapters.py                   generates adapter.json from planes.py
  env.py                        client for `tinybrains env`
  teacher.py                    the scripted bot the class ladder is distilled from
  collect.py                    teacher rollouts to a dataset
  nets.py                       one architecture per class
  train/bc.py                   behaviour cloning
  export.py                     torch -> ONNX -> fp16 -> the platform's verdict
  eval.py                       round robin through `tinybrains <match>`
models/                         the finished artifacts, committed
tests/                          the conformance gate
```

### What cost the most to learn

[`docs/receptive-field.md`](docs/receptive-field.md) is the record of the mistake this repository
existed to catch: the first two models were designed against the byte cap, spent it on width, and
could see one and two cells respectively while the teacher they imitated plans 32 cells deep. Nine
times the parameters bought 2.6 points of agreement and the bigger one lost the round robin.
Dilating the convolutions took micro from 47.2% to 91.1% with fewer parameters.

## What must stay true

- **`adapter.json` is generated, never hand-edited.** It is one of two renderings of the encoding;
  editing it alone reintroduces exactly the skew the conformance test exists to catch.
- **The teacher never ships.** It is a label source. Changing it invalidates the class ladder, which
  is only a comparison because every class distils the same one.
- **The value head is never exported.** A critic may see privileged information precisely because it
  is discarded before anything plays.
- **`tinybrains env` is not the referee.** No deadline, no strikes, no adapter. A result from the env
  is not a result.
- **Every artifact names its engine digest.** An engine change is a rules change, and a model that
  cannot say which engine it was trained against cannot be reproduced.

## Status

**10 September 2026 — the pipeline is built and measured; the artifacts are not finished.**

Working end to end: the environment client, the generated adapter (245,839 operations at the worst
reference board, 24% of the budget), the conformance test, the teacher, dataset collection
(250,000 seat-turns and 7.4 million ant labels in nine minutes), behaviour cloning on MPS, and
export with the platform's own verdict.

Measured and settled: **fp16 initializers give 2.03x the parameters for the same weight class** with
identical play; **the turn deadline binds before the byte cap above `mini`**, and `small` reaches its
31.2 ms seat share at about a quarter of its cap.

Not built: the `large` class (no dense architecture reaches its cap inside the deadline — the plan
is a learned pattern table), PPO self-play, the method column, and the platform seeding.

## More

- [The competitor guide](https://github.com/Tiny-Brains/docs) — the rules, the model format, the
  adapter dialect, submitting, ranking and seasons.
- [drill](https://github.com/Tiny-Brains/drill) — match files, boards and a place to try a model
  without a database.
- Apache-2.0: see [LICENSE](LICENSE).
