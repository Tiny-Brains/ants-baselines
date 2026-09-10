# micro-bc

A micro-class Ants policy.

| | |
|---|---|
| Weight class | **micro** — 50,143 of 65,536 bytes (77% of the cap) |
| Parameters | 26,453 (fp16 initializers) |
| Method | behaviour cloning from the deterministic potential-field teacher, 6 epochs over 250,000 seat-turns |
| Adapter | 245,839 of 1,000,000 operations at its worst reference case |
| Inference | 10.54 ms at the worst reference case — **34%** of a 31.2 ms seat share |
| Operators | Cast, Concat, Constant, Conv, Relu, Slice |
| Engine | `sha256:f17b51b6c92b066d9a354578b2774e98d88ddcece1db3ac7aa6b3e3271d72865` |
| Evaluator | `sha256:44cfc91cb1f20f9a4b46d742169c22f970a96801faa843416f4d6776c9b3c505` |
| Model hash | `sha256:22c0baaba8c85ea93e2a578e7134f1f44714791c3f0397f609e7391e3d8dc5cd` |
| Adapter hash | `sha256:9c022ff50c7210213afa4a88ffb56cd2dee188f99b949335a037bdc60ab2d468` |



Reproduce with `see README.md`.

Inference time is measured on whatever machine ran the check and is **reported, never a gate**:
there is no compute cap (devops decision 46). It is here because the turn deadline is what a graph
too expensive to play runs into, and a seat's share of it is `turn_ms / rows in the call`.
