# micro-percell

A micro-class Ants policy.

| | |
|---|---|
| Weight class | **micro** — 45,942 of 65,536 bytes (70% of the cap) |
| Parameters | 24,953 (fp16 initializers) |
| Architecture | `PerCell`, receptive field **0 cells** each way |
| Method | behaviour cloning, same data as micro-bc; the reach-0 control, 40.7% held-out agreement |
| Adapter | 245,839 of 1,000,000 operations at its worst reference case |
| Inference | 10.75 ms at the worst reference case — **34%** of a 31.2 ms seat share |
| Operators | Cast, Conv, Relu |
| Engine | `sha256:f17b51b6c92b066d9a354578b2774e98d88ddcece1db3ac7aa6b3e3271d72865` |
| Evaluator | `sha256:44cfc91cb1f20f9a4b46d742169c22f970a96801faa843416f4d6776c9b3c505` |
| Model hash | `sha256:85713895274b6d1040ab14205c6f9e00eaf3657c6ed61c10444d6acabfcae193` |
| Adapter hash | `sha256:9c022ff50c7210213afa4a88ffb56cd2dee188f99b949335a037bdc60ab2d468` |



Reproduce with `see README.md`.

Inference time is measured on whatever machine ran the check and is **reported, never a gate**:
there is no compute cap (devops decision 46). It is here because the turn deadline is what a graph
too expensive to play runs into, and a seat's share of it is `turn_ms / rows in the call`.
