"""PPO self-play: learn Ants from a reward instead of from a teacher.

    python -m tb_baselines.train.ppo --class micro --iters 200

This is the interesting half of the matrix and the risky one. Ants is sparse-reward, long-horizon,
multi-agent and fogged; behaviour cloning is none of those. Read `eval.py`'s table before believing
a rising return.

## Both seats are the same policy, and that is the point

Every match seats the learner against itself, so one match yields two trajectories and the opponent
improves exactly as fast as the learner does. It also means the observation must be **symmetric** —
which it now is: engine `sha256:f17b51b6c92b` made a view's owner labels relative to the observer,
and before that the two seats of one match were, quietly, two different games.

## The action space is factorised, not joint

A seat commands up to 180 ants from one forward pass. The joint action space is 5^180, and the joint
distribution is not what the network produces: it produces a per-cell distribution, and each ant
reads its own cell. So

    log pi(a | s) = sum over ants of log pi(a_i | s, pos_i)

which is the standard factorised policy, and the entropy sums the same way. The consequence to keep
in mind is that **a seat's log-probability scales with its ant count**, so a colony of 90 has a
gradient signal thirty times a colony of 3. The ratio in the PPO objective is formed per ANT rather
than per seat for exactly that reason: a per-seat ratio is `exp(sum of 90 differences)`, which
overflows the clip range on the first update and makes the algorithm a no-op.

## The critic is privileged and is thrown away

`tinybrains env --scores every` reports each match's true score every turn — `f_finish` answers it
for a live match — and the value head reads it. That is asymmetric actor-critic, not a hole in the
fog: `export.py` ships the trunk and the policy head only, so nothing that saw a score survives into
anything that plays.

## Rollouts hold observations, not boards

A `[7, 128, 128]` int8 board is 114 KiB, so a rollout of 64 steps across 100 live seats would be
730 MB of tensors. The observations behind them are about 2 KB each. They are re-encoded during the
update, which costs about a second per epoch and makes the rollout length a free parameter instead
of a memory budget.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .. import nets
from ..env import Env, orders_from_indices
from ..export import budget, classes
from ..planes import N_MOVES
from .bc import device


@dataclass
class Reward:
    """What a turn is worth. Every term is a training-time choice and none is the ladder's.

    The ladder pays for rank, and rank comes from score: +2 for razing an enemy hill, -1 for losing
    one of yours. That signal arrives a handful of times in three hundred turns, so the shaping
    terms exist to make the first thousand updates about something. They are declared here rather
    than buried so that a run's card can name them.
    """

    score: float = 1.0          # your own score, turn on turn
    rival: float = 0.5          # minus theirs: Ants is not zero-sum, but taking a hill is
    ants: float = 0.02          # the food economy, which is the only dense signal there is
    win: float = 1.0            # at the end, on rank rather than on margin
    gamma: float = 0.997        # about a 300-turn horizon
    lam: float = 0.95


@dataclass
class Trajectory:
    """One seat's episode, keyed by `(ep, seat)`. `ep` is stable across a wave refill and `(w, m)`
    is not."""

    obs: list[dict] = field(default_factory=list)
    # The encoded board, kept from the rollout. `encode` is not free and the update would otherwise
    # redo it once per epoch for every sample -- work that was already done to choose the action.
    boards: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    logp: list[np.ndarray] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    last_score: int = 0
    last_rival: int = 0
    last_ants: int = 0
    primed: bool = False        # a match opens at score 1, so the first delta must not count it
    finished: bool = False


def gae(rewards: list[float], values: list[float], bootstrap: float, r: Reward):
    """Advantages and returns for one seat. `bootstrap` is 0 at a real terminal and V(s_T) when the
    rollout simply stopped -- collapsing the two is how a truncated episode teaches the agent that
    the world ends."""
    adv, out = [], 0.0
    nxt = bootstrap
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + r.gamma * nxt - values[t]
        out = delta + r.gamma * r.lam * out
        adv.append(out)
        nxt = values[t]
    adv.reverse()
    returns = [a + v for a, v in zip(adv, values)]
    return adv, returns


class Runner:
    """The rollout loop: env in, trajectories out."""

    def __init__(self, model: nets.ActorCritic, env: Env, dev: torch.device, reward: Reward):
        self.model = model
        self.env = env
        self.dev = dev
        self.reward = reward
        self.step = env.reset()
        self.open: dict[tuple[int, int], Trajectory] = {}
        self.done: list[Trajectory] = []
        self.returns: list[float] = []

    @torch.no_grad()
    def act(self, boards: np.ndarray, rows: "_AntIndex"):
        """Sample a move for every ant on every board in one group."""
        t = torch.from_numpy(boards).to(self.dev)
        logits, values = self.model(t)
        per_ant = logits[rows.board, :, rows.r, rows.c]
        # validate_args=False: the checks cost 8% of the whole loop proving that a tensor of
        # logits is a tensor of logits, on every construction, on every minibatch.
        dist = torch.distributions.Categorical(logits=per_ant, validate_args=False)
        picks = dist.sample()
        return picks.cpu().numpy(), dist.log_prob(picks).cpu().numpy(), values.cpu().numpy()

    def collect(self, steps: int) -> dict:
        """`steps` env turns. Returns statistics; the trajectories accumulate on `self`."""
        self.done, self.returns = [], []
        for _ in range(steps):
            actions: list[str | None] = [None] * len(self.step.seats)

            for group in self.step.groups:
                idx = np.array(group.indices)
                counts = [len(self.step.seats[i].obs["mine"]) for i in idx]
                rows = _AntIndex(self.step.seats, group.indices).to(self.dev)
                picks, logp, values = self.act(group.boards, rows)

                at = 0
                for k, i in enumerate(idx):
                    n = counts[k]
                    seat = self.step.seats[i]
                    mine = picks[at : at + n]
                    actions[i] = orders_from_indices(mine, [n])[0]
                    self._record(seat, group.boards[k : k + 1], mine,
                                 logp[at : at + n], float(values[k]))
                    at += n

            prev = self.step
            self.step = self.env.step([a if a is not None else "" for a in actions])
            self._reward(prev)
            self._close(self.step.ended)

        # Whatever is still open bootstraps rather than pretending to have ended.
        return {
            "episodes": len(self.returns),
            "mean_return": float(np.mean(self.returns)) if self.returns else 0.0,
        }

    def _record(self, seat, board, picks, logp, value) -> None:
        key = (seat.ep, seat.seat)
        traj = self.open.setdefault(key, Trajectory())
        if not traj.obs:
            traj.last_ants = len(seat.obs["mine"])
        traj.obs.append(seat.obs)
        traj.boards.append(board)
        traj.actions.append(picks.copy())
        traj.logp.append(logp.copy())
        traj.values.append(value)

    def _reward(self, prev) -> None:
        """One reward a seat, from the score the env reported and the colony it kept."""
        r = self.reward
        for seat in prev.seats:
            key = (seat.ep, seat.seat)
            traj = self.open.get(key)
            if traj is None or len(traj.rewards) >= len(traj.obs):
                continue
            scores = self.step.scores.get(seat.ep) or {e.ep: e.scores for e in self.step.ended}.get(seat.ep)
            if scores is None:
                traj.rewards.append(0.0)
                continue
            mine = scores[seat.seat]
            rival = max(s for i, s in enumerate(scores) if i != seat.seat)
            ants = len(seat.obs["mine"])
            if not traj.primed:
                # Ants opens every seat on 1 point, and a colony starts with one ant. Priming from
                # zero would pay a point and an ant for turning up.
                traj.last_score, traj.last_rival, traj.last_ants = mine, rival, ants
                traj.primed = True
            value = (
                r.score * (mine - traj.last_score)
                - r.rival * (rival - traj.last_rival)
                + r.ants * (ants - traj.last_ants)
            )
            traj.last_score, traj.last_rival, traj.last_ants = mine, rival, ants
            traj.rewards.append(value)

    def _close(self, ended) -> None:
        for e in ended:
            for seat in range(e.seat_count):
                traj = self.open.pop((e.ep, seat), None)
                if traj is None or not traj.obs:
                    continue
                best = min(e.ranks)
                outcome = 1.0 if e.ranks[seat] == best and e.ranks.count(best) == 1 else (
                    0.0 if e.ranks[seat] == best else -1.0
                )
                while len(traj.rewards) < len(traj.obs):
                    traj.rewards.append(0.0)
                traj.rewards[-1] += self.reward.win * outcome
                traj.finished = True
                self.done.append(traj)
                self.returns.append(float(sum(traj.rewards)))

    def harvest(self) -> list[Trajectory]:
        """Finished episodes, plus a snapshot of everything still running so a long match still
        contributes. The open ones keep their running score, so nothing is collected twice and the
        next segment's first delta is measured from where this one stopped."""
        batch = list(self.done)
        for key, traj in list(self.open.items()):
            if len(traj.rewards) >= len(traj.obs) and traj.obs:
                batch.append(traj)
                self.open[key] = Trajectory(
                    last_score=traj.last_score, last_rival=traj.last_rival,
                    last_ants=traj.last_ants, primed=True,
                )
        return batch


class _AntIndex:
    """Flat `(board, row, col)` for every ant in a group, which is the projection the policy head
    and the adapter's `out` program both do."""

    def __init__(self, seats, indices):
        b, r, c = [], [], []
        for k, i in enumerate(indices):
            for row, col in seats[i].obs["mine"]:
                b.append(k)
                r.append(row)
                c.append(col)
        self.board = torch.tensor(b, dtype=torch.long)
        self.r = torch.tensor(r, dtype=torch.long)
        self.c = torch.tensor(c, dtype=torch.long)

    def to(self, dev):
        self.board, self.r, self.c = self.board.to(dev), self.r.to(dev), self.c.to(dev)
        return self


def update(model, opt, batch: list[Trajectory], r: Reward, dev, epochs: int,
           clip: float, entropy: float, minibatch: int) -> dict:
    """PPO, with the ratio formed per ant.

    A per-seat ratio would be `exp(sum over 90 ants of the log-probability difference)`, which
    leaves any sane clip range on the first update. Per ant is both correct for a factorised policy
    and numerically the only version that does anything.
    """
    samples = []
    for traj in batch:
        adv, ret = gae(traj.rewards, traj.values, 0.0 if traj.finished else traj.values[-1], r)
        for t in range(len(traj.obs)):
            if len(traj.actions[t]) == 0:
                continue
            samples.append((traj.obs[t], traj.boards[t], traj.actions[t], traj.logp[t],
                            adv[t], ret[t]))
    if not samples:
        return {"samples": 0}

    advs = np.array([s[4] for s in samples], dtype=np.float32)
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

    by_size: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, sample in enumerate(samples):
        by_size[tuple(sample[0]["size"])].append(i)

    stats = defaultdict(float)
    batches = 0
    for _ in range(epochs):
        for group in by_size.values():
            order = np.random.permutation(len(group))
            for start in range(0, len(order), minibatch):
                chunk = [group[j] for j in order[start : start + minibatch]]
                bt = torch.from_numpy(np.concatenate([samples[i][1] for i in chunk], 0)).to(dev)

                b_idx, rr, cc, acts, olp, adv_rep = [], [], [], [], [], []
                rets = []
                for k, i in enumerate(chunk):
                    obs, _, a, lp, _, ret = samples[i]
                    mine = np.asarray(obs["mine"], dtype=np.int64).reshape(-1, 2)
                    b_idx.append(np.full(len(mine), k, dtype=np.int64))
                    rr.append(mine[:, 0])
                    cc.append(mine[:, 1])
                    acts.append(a)
                    olp.append(lp)
                    adv_rep.append(np.full(len(a), advs[i], dtype=np.float32))
                    rets.append(ret)

                nd = lambda parts, dt: torch.from_numpy(
                    np.concatenate(parts).astype(dt, copy=False)).to(dev)
                logits, values = model(bt)
                per_ant = logits[nd(b_idx, np.int64), :, nd(rr, np.int64), nd(cc, np.int64)]
                dist = torch.distributions.Categorical(logits=per_ant, validate_args=False)
                a_t = nd(acts, np.int64)
                new_lp = dist.log_prob(a_t)
                old_lp = nd(olp, np.float32)
                adv_t = nd(adv_rep, np.float32)

                ratio = torch.exp(new_lp - old_lp)
                policy_loss = -torch.min(
                    ratio * adv_t, torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t
                ).mean()
                value_loss = F.mse_loss(
                    values, torch.from_numpy(np.asarray(rets, dtype=np.float32)).to(dev)
                )
                ent = dist.entropy().mean()
                loss = policy_loss + 0.5 * value_loss - entropy * ent

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()

                batches += 1
                stats["policy"] += float(policy_loss.detach())
                stats["value"] += float(value_loss.detach())
                stats["entropy"] += float(ent.detach())
                stats["clipped"] += float(
                    ((ratio - 1).abs() > clip).float().mean().detach()
                )
    return {k: v / max(batches, 1) for k, v in stats.items()} | {"samples": len(samples)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--class", dest="cls", required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--rollout", type=int, default=48, help="env turns an iteration")
    ap.add_argument("--waves", type=int, default=4)
    ap.add_argument("--matches-per-wave", type=int, default=8)
    ap.add_argument("--max-turns", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", type=Path, default=None, help="a bc checkpoint to start from")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = device()

    spec = classes()[a.cls]
    trunk = nets.build(spec)
    if a.init:
        state = torch.load(a.init, map_location="cpu")
        trunk.load_state_dict(state.get("trunk", state))
    model = nets.ActorCritic(trunk).to(dev)

    b = budget(a.cls)
    method = "bc-ppo" if a.init else "ppo"
    print(f"{a.cls} {method}: {nets.policy_params(model):,} policy parameters "
          f"(budget about {b['params_at_target']:,}), {dev.type}")

    out = a.out or Path("runs") / f"{a.cls}-{method}"
    out.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    reward = Reward()

    history = []
    best = -1e9
    with Env(waves=a.waves, matches_per_wave=a.matches_per_wave,
             max_turns=a.max_turns, seed=1000 + a.seed) as env:
        runner = Runner(model, env, dev, reward)
        print(f"engine {env.engine_digest[:20]}")
        for it in range(1, a.iters + 1):
            t0 = time.time()
            rolled = runner.collect(a.rollout)
            batch = runner.harvest()
            stats = update(model, opt, batch, reward, dev, a.epochs, a.clip,
                           a.entropy, a.minibatch)
            took = time.time() - t0
            row = {"iter": it, **rolled, **stats, "seconds": round(took, 1)}
            history.append(row)
            print(f"  {it:4d}  return {rolled['mean_return']:+7.2f} over {rolled['episodes']:3d} eps"
                  f"   pi {stats.get('policy', 0):+.4f}  V {stats.get('value', 0):.3f}"
                  f"  H {stats.get('entropy', 0):.3f}  {stats['samples']:5d} samples  {took:.0f}s")
            if rolled["episodes"] and rolled["mean_return"] > best:
                best = rolled["mean_return"]
                torch.save({"trunk": trunk.state_dict(), "class": a.cls,
                            "engine_digest": env.engine_digest}, out / "best.pt")
            if it % 10 == 0:
                (out / "history.json").write_text(json.dumps(
                    {"class": a.cls, "method": method, "reward": vars(reward),
                     "iters": history}, indent=2) + "\n")

    (out / "history.json").write_text(json.dumps(
        {"class": a.cls, "method": method, "reward": vars(reward), "iters": history}, indent=2) + "\n")
    print(f"  -> {out / 'best.pt'}")
    print("  a rising return is not strength. Play it: `python -m tb_baselines.eval`")


if __name__ == "__main__":
    main()
