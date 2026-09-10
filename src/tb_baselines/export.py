"""Torch to a submittable artifact: ONNX, fp16 initializers, and the platform's own verdict.

The last step is not a check this repository invents. It shells out to `tinybrains check --json`,
which is Axon's `/inspect` and `/validate` in the same crate the fleet runs — so "does it fit its
class" is answered by the code that will actually answer it, over the cartridge's own reference
observations. What this module adds is the loop around that: build, measure, and refuse an artifact
that misses its class.

    python -m tb_baselines.export --class micro --weights runs/micro-bc/best.pt --out models/micro-bc

Every export writes three files beside the model: `adapter.json` (generated, never edited),
`metrics.json` (what the platform said), and `card.md` (what a person needs to reproduce it). The
card names the **engine digest** and the **evaluator digest**, because a baseline that cannot say
which engine it was trained against is a baseline nobody can reproduce.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tomllib
import warnings
from pathlib import Path

import onnx
import torch
from onnx import numpy_helper

from . import adapters, nets
from .planes import N_PLANES

ROOT = Path(__file__).resolve().parents[2]
OPSET = 17  # inside the deployment's 13-19, and what the reference fixtures use


# ---- the class table --------------------------------------------------------------------

def classes() -> dict:
    with open(ROOT / "classes.toml", "rb") as f:
        doc = tomllib.load(f)
    return {c["name"]: c for c in doc["class"]} | {"_": doc}


def budget(name: str) -> dict:
    """What one class allows, worked out from `classes.toml` rather than written down twice."""
    table = classes()
    spec = table[name]
    doc = table["_"]
    per_param = doc["metric"][f"{doc['dtype']['default']}_bytes_per_param"]
    # A class may set its own fill: `small` cannot reach the global 70% inside the turn deadline,
    # and pretending otherwise would make every small artifact fail certification for a reason that
    # is not the artifact's fault.
    fraction = spec.get("fraction", doc["target"]["fraction"])
    target = int(spec["max_bytes"] * fraction)
    weights = target - doc["metric"]["adapter_zstd_bytes"]
    c = doc["compute"]
    return {
        "name": name,
        "max_bytes": spec["max_bytes"],
        "fraction": fraction,
        "target_bytes": target,
        "weight_bytes": weights,
        "max_share": doc["target"]["max_share"],
        "share_us": c["turn_ms"] * 1000 / c["rows_per_call"],
        "params_at_target": int(weights / per_param),
        "flops_per_seat_turn": c["turn_ms"] / 1000 / c["rows_per_call"] * c["flops_per_second"],
        "spec": spec,
    }


# ---- torch to onnx ----------------------------------------------------------------------

def to_onnx(trunk: torch.nn.Module, path: Path, planes: int = N_PLANES) -> None:
    """The policy half only.

    `board` and `policy` both declare a dynamic leading dimension, which is what makes the graph
    *batchable*: Axon stacks every row of a wave that shares a `weights_hash` into one inference,
    and a graph without that axis costs one inference a seat. H and W are dynamic too, because the
    three presets are three board sizes.
    """
    trunk.eval()
    example = torch.zeros(1, planes, 128, 128, dtype=torch.int8)
    # The TorchScript exporter is deprecated in favour of dynamo, which does not yet produce the
    # dynamic H/W axes this graph needs on every board size. Pinned deliberately; revisit when it
    # does.
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch.onnx")
    torch.onnx.export(
        trunk, (example,), str(path),
        input_names=["board"], output_names=["policy"],
        dynamic_axes={"board": {0: "B", 2: "H", 3: "W"},
                      "policy": {0: "B", 2: "H", 3: "W"}},
        opset_version=OPSET, dynamo=False,
    )


def halve(path: Path) -> None:
    """Rewrite every float32 initializer as float16, with a `Cast` back at its use.

    The size metric compresses **initializer data**, so this halves what the class is measured on.
    The graph's compute dtype is untouched: ORT constant-folds `Cast(initializer)` during graph
    optimisation, so the fp32 tensor is rebuilt once at session load and never per inference.

    Measured 10 September 2026: 2.03x the parameters for the same S, no measurable change in
    inference time, the same three operators, and 396 of 396 per-ant orders identical to the fp32
    graph over three played matches.
    """
    model = onnx.load(str(path))
    g = model.graph
    casts, kept = [], []
    for init in g.initializer:
        array = numpy_helper.to_array(init)
        if array.dtype != "float32":
            kept.append(init)
            continue
        kept.append(numpy_helper.from_array(array.astype("float16"), init.name + "_h"))
        casts.append(
            onnx.helper.make_node(
                "Cast", [init.name + "_h"], [init.name],
                to=onnx.TensorProto.FLOAT, name="cast_" + init.name,
            )
        )
    if not casts:
        return
    del g.initializer[:]
    g.initializer.extend(kept)
    # ONNX requires topological order and a Cast must precede its consumer, so they all go first.
    rest = list(g.node)
    del g.node[:]
    g.node.extend(casts + rest)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


# ---- the platform's verdict -------------------------------------------------------------

def cli() -> str:
    found = os.environ.get("TINYBRAINS") or shutil.which("tinybrains")
    if not found:
        raise SystemExit(
            "no `tinybrains` on PATH.\n"
            "  cargo install --path ../devops/cli   (or set TINYBRAINS to the binary)"
        )
    return found


def verdict(model: Path, adapter: Path) -> dict:
    """`tinybrains check --json`: Axon's own inspect and validate, over the reference set."""
    r = subprocess.run(
        [cli(), "check", str(model), str(adapter), "--json"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if not r.stdout.strip():
        raise SystemExit(f"tinybrains check produced nothing:\n{r.stderr}")
    return json.loads(r.stdout)


def certify(name: str, said: dict) -> list[str]:
    """Everything wrong with this artifact, in the order a person would want to hear it."""
    b = budget(name)
    ins, val = said["inspect"], said["validate"]
    size = ins["size_metric_bytes"]
    problems = []

    if not said["ok"]:
        problems.append(
            f"the platform refused it: {val.get('reason')} {val.get('detail') or ''}".strip()
        )
    if size > b["max_bytes"]:
        problems.append(
            f"S is {size:,} bytes and {name} caps at {b['max_bytes']:,} -- over by "
            f"{size - b['max_bytes']:,}"
        )
    if ins["unsupported_ops"]:
        problems.append(f"operators outside the allowlist: {', '.join(ins['unsupported_ops'])}")
    # The deadline, made a check rather than a paragraph. Above `mini` this is what refuses an
    # artifact, and the byte cap never gets a say.
    share = val["infer_us_max"] / b["share_us"]
    if share > b["max_share"]:
        problems.append(
            f"it spends {share:.0%} of a seat's {b['share_us'] / 1000:.1f} ms deadline share at the "
            f"worst reference board, over the {b['max_share']:.0%} this repository allows. The class "
            f"has bytes left; the turn does not."
        )
    # Batchability is not something `check` reports, so read it off the declared shape: a leading
    # dimension that is not dynamic means one inference per seat instead of one per wave.
    board = next((p for p in ins["inputs"] if p["name"] == "board"), None)
    if board and isinstance(board["shape"][0], int):
        problems.append(
            "the graph fixes its leading dimension, so it cannot batch -- every seat costs its own "
            "inference. Export with a dynamic axis 0."
        )
    return problems


def report(name: str, said: dict) -> dict:
    b = budget(name)
    ins, val = said["inspect"], said["validate"]
    return {
        "class": name,
        "size_metric_bytes": ins["size_metric_bytes"],
        "class_max_bytes": b["max_bytes"],
        "class_target_bytes": b["target_bytes"],
        "fill_of_cap": round(ins["size_metric_bytes"] / b["max_bytes"], 4),
        "weights_zstd_bytes": ins["weights_zstd_bytes"],
        "adapter_zstd_bytes": ins["adapter_zstd_bytes"],
        "params": ins["params"],
        "opset": ins["opset"],
        "operators": ins["ops"],
        "adapter_ops_max": val["ops_max"],
        "adapter_ops_budget": said["budget_ops"],
        "infer_us_max": val["infer_us_max"],
        "deadline_share_ms": round(b["share_us"] / 1000, 2),
        "share_used": round(val["infer_us_max"] / b["share_us"], 4),
        "target_fraction": b["fraction"],
        "engine_digest": said["engine_digest"],
        "evaluator_digest": val["evaluator_digest"],
        "weights_hash": said["weights_hash"],
        "adapter_hash": said["adapter_hash"],
    }


CARD = """# {name}

{summary}

| | |
|---|---|
| Weight class | **{cls}** — {size:,} of {cap:,} bytes ({fill:.0%} of the cap) |
| Parameters | {params:,} ({dtype} initializers) |
| Method | {method} |
| Adapter | {ops:,} of {opsmax:,} operations at its worst reference case |
| Inference | {infer:.2f} ms at the worst reference case — **{shareuse:.0%}** of a {share:.1f} ms seat share |
| Operators | {operators} |
| Engine | `{engine}` |
| Evaluator | `{evaluator}` |
| Model hash | `{whash}` |
| Adapter hash | `{ahash}` |

{notes}

Reproduce with `{repro}`.

Inference time is measured on whatever machine ran the check and is **reported, never a gate**:
there is no compute cap (devops decision 46). It is here because the turn deadline is what a graph
too expensive to play runs into, and a seat's share of it is `turn_ms / rows in the call`.
"""


def write_card(out: Path, name: str, method: str, metrics: dict, summary: str,
               notes: str, repro: str) -> None:
    (out / "card.md").write_text(CARD.format(
        name=name, summary=summary, cls=metrics["class"],
        size=metrics["size_metric_bytes"], cap=metrics["class_max_bytes"],
        fill=metrics["fill_of_cap"], params=metrics["params"],
        dtype=classes()["_"]["dtype"]["default"], method=method,
        ops=metrics["adapter_ops_max"], opsmax=metrics["adapter_ops_budget"],
        infer=metrics["infer_us_max"] / 1000, share=metrics["deadline_share_ms"],
        shareuse=metrics["share_used"],
        operators=", ".join(metrics["operators"]), engine=metrics["engine_digest"],
        evaluator=metrics["evaluator_digest"], whash=metrics["weights_hash"],
        ahash=metrics["adapter_hash"], notes=notes, repro=repro,
    ))


def export(trunk: torch.nn.Module, name: str, out: Path, method: str,
           summary: str = "", notes: str = "", repro: str = "") -> dict:
    """The whole pipeline. Raises if the artifact misses its class."""
    out.mkdir(parents=True, exist_ok=True)
    model_path, adapter_path = out / "model.onnx", out / "adapter.json"
    adapter_path.write_text(adapters.dumps())
    to_onnx(trunk, model_path)
    if classes()["_"]["dtype"]["default"] == "fp16":
        halve(model_path)

    said = verdict(model_path, adapter_path)
    problems = certify(name, said)
    metrics = report(name, said)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    write_card(out, out.name, method, metrics,
               summary or f"A {name}-class Ants policy.", notes, repro or "see README.md")

    if problems:
        raise SystemExit(
            f"{out}: this artifact does not belong in {name}\n  "
            + "\n  ".join(problems)
        )
    return metrics


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--class", dest="cls", required=True)
    # A variant is a command line, not a second table: the run's history.json and the model card
    # both record what was actually built, which is the copy that matters.
    ap.add_argument("--arch", default=None, help="override the class's architecture")
    ap.add_argument("--channels", type=int, default=None)
    ap.add_argument("--blocks", type=int, default=None)
    ap.add_argument("--weights", type=Path, help="a .pt state dict; omitted means random init")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--method", default="untrained")
    a = ap.parse_args(argv)

    from .train.bc import override
    spec = override(classes()[a.cls], a)
    trunk = nets.build(spec)
    if a.weights:
        state = torch.load(a.weights, map_location="cpu")
        trunk.load_state_dict(state.get("trunk", state))

    b = budget(a.cls)
    print(f"{a.cls}: {nets.policy_params(trunk):,} parameters, "
          f"budget about {b['params_at_target']:,} at {b['target_bytes']:,} bytes")
    m = export(trunk, a.cls, a.out, a.method)
    print(f"  S = {m['size_metric_bytes']:,} bytes, {m['fill_of_cap']:.0%} of the {a.cls} cap "
          f"(aiming at {m['target_fraction']:.0%})")
    print(f"  adapter {m['adapter_ops_max']:,} ops, inference {m['infer_us_max'] / 1000:.2f} ms "
          f"= {m['share_used']:.0%} of a seat share")
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
