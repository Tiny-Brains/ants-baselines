# Receptive field, not parameter count

The first two models trained here were bad, and the reason was not the thing the weight classes are
about. This is the record of finding that out, kept because the numbers are the argument.

## What was built first

`Trunk(channels, blocks)` as originally written: one 3x3 stem, `blocks` further 3x3s, then 1x1
layers. The 1x1s were a deliberate choice — they buy depth for a ninth of what a 3x3 costs, which
seemed like the right trade when the whole budget is a few thousand parameters.

It is the right trade for *capacity* and the wrong one for *reach*. A 1x1 layer adds none. So:

| | channels | 3x3 layers | parameters | **receptive field** |
|---|---:|---:|---:|---:|
| nano | 30 | 1 | 3,005 | **3x3** — one cell |
| micro | 48 | 2 | 26,453 | **5x5** — two cells |

The teacher these were imitating plans over a breadth-first flood **32 cells deep**. An ant that can
see one cell cannot follow a trail; it can only tell whether it is standing on something.

## What that cost, measured

Both were trained by behaviour cloning for six epochs over 250,000 seat-turns (7.3 million ant
decisions), against a deterministic teacher whose majority-class floor is 38.1%.

| | held-out agreement | final ants a seat | round robin, 18 boards |
|---|---:|---:|---|
| nano-bc, reach 1 | 44.6% | 5.2 | **4 wins, 14 draws, 0 losses** |
| micro-bc, reach 2 | 47.2% | 14.7 | 0 wins, 14 draws, 4 losses |
| the teacher | — | 32.9 | — |

Three things in that table are worth sitting with.

**Both plateaued inside one epoch.** Train and held-out tracked each other the whole way — 1.2124 to
1.1918 over six epochs for micro. That is not overfitting and it is not a learning-rate problem. It
is a model that has extracted everything its input window contains by the end of the first pass.

**Nine times the parameters bought 2.6 points.** Capacity was not the binding constraint, so
spending the class on it did nothing.

**The bigger model lost.** micro grew colonies nearly three times the size of nano's — a better food
economy — and still went 0-4. Whatever nano was doing with its one cell of context, more capacity at
two cells did not beat it. That is the shape of a measurement telling you that you are optimising
the wrong axis.

## The fix costs nothing

ONNX carries **dilation as an attribute of `Conv`**, not as an operator of its own, so a dilated
graph inspects as `Cast, Concat, Constant, Conv, Relu, Slice` — every one already on the platform's
allowlist. Nothing had to be asked for.

Doubling the dilation each layer grows the receptive field exponentially instead of by two cells a
layer: dilations 1, 2, 4, 8 reach **15 cells each way in four layers**, where an undilated stack
would need sixteen layers to get there.

| | channels | layers | dilations | parameters | reach | S | of a seat share |
|---|---:|---:|---|---:|---:|---:|---:|
| nano | 9 | 4 | 1,2,4,8 | 2,930 | 15 | 5,785 (71% of cap) | 9% |

(Sizes are the trained artifacts'. Trained weights compress slightly *worse* than random ones —
micro measured 44,008 bytes untrained and 45,642 trained — which is worth knowing before sizing a
model to the last kilobyte of its class.)
| micro | 28 | 4 | 1,2,4,8 | 24,001 | 15 | 45,642 (70% of cap) | 46% |

**Nano went from 30 channels to 9** — a third of the width — to pay for four layers instead of one.
Micro went from 48 to 28. Both got smaller in every dimension except the one that mattered.

The first epoch of the retrained micro answered the question on its own: **held-out agreement went
from 47.2% to 91.1%**, against a floor of 38.1%, on the same data with fewer parameters. Five epochs
finished at **93.8%**.

## And then they played each other

Twenty-four matches, every pair of the three presets' first four boards, both seat orders.

| | parameters | S | agreement | peak colony | W–D–L |
|---|---:|---:|---:|---:|---|
| micro, reach 15 | 24,001 | 45,642 (70% of cap) | 93.8% | **31.6** | **19–5–0** |
| micro, reach 2 | 26,453 | 50,143 (77% of cap) | 47.2% | 10.7 | 0–5–19 |

Mean score 2.58 against 0.21. A match starts every seat on one point and the only way up is razing
an enemy hill, so the wide-but-blind model was **losing hills faster than it took them** while the
dilated one takes them — and the end reasons say the same thing: eleven matches settled early on
rank, eight by extermination, and only three of the twenty-four stalled into `idle_food`, which was
the dominant ending before.

The colony number is the one to keep. **31.6 ants a seat against the teacher's 32.9**: at reach 15 a
24,000-parameter network recovers essentially all of the teacher's food economy, and at reach 2 a
larger one recovered a third of it. Nothing about the byte budget changed between those two rows.

## Why this belongs in a repository about weight classes

Because it is the thing a competitor will get wrong, and the class table will not warn them. The
platform measures bytes, so a first model is designed against bytes — and at these budgets the
temptation is exactly the one taken here: spend the bytes on width, because width is what a
parameter count is made of.

Reach is nearly free. It is layers and dilation, and at nine channels a layer costs almost nothing.
Buying it with width is the trade, and until this was measured the trade was being made backwards.
