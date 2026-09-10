# micro-bc

A micro-class Ants policy.

| | |
|---|---|
| Weight class | **micro** — 45,642 of 65,536 bytes (70% of the cap) |
| Parameters | 24,001 (fp16 initializers) |
| Architecture | `Trunk`, receptive field **15 cells** each way (dilations [1, 2, 4, 8]) |
| Method | behaviour cloning, 5 epochs over 250,000 seat-turns, 93.8% held-out agreement |
| Adapter | 245,839 of 1,000,000 operations at its worst reference case |
| Inference | 14.33 ms at the worst reference case — **46%** of a 31.2 ms seat share |
| Operators | Cast, Concat, Constant, Conv, Relu, Slice |
| Engine | `sha256:f17b51b6c92b066d9a354578b2774e98d88ddcece1db3ac7aa6b3e3271d72865` |
| Evaluator | `sha256:44cfc91cb1f20f9a4b46d742169c22f970a96801faa843416f4d6776c9b3c505` |
| Model hash | `sha256:1e51646a3987eb7844a7cd8e9c3f12f36d2a5d5a678286d856ac1c1be9801268` |
| Adapter hash | `sha256:9c022ff50c7210213afa4a88ffb56cd2dee188f99b949335a037bdc60ab2d468` |



Reproduce with `see README.md`.

Inference time is measured on whatever machine ran the check and is **reported, never a gate**:
there is no compute cap (devops decision 46). It is here because the turn deadline is what a graph
too expensive to play runs into, and a seat's share of it is `turn_ms / rows in the call`.
