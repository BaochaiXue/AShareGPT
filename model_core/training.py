"""
Unified PPO training workflow for alpha formula discovery.

Merges: ppo_training_service, reward_orchestrator, training_workflow_service, factories.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

from .config import ModelConfig
from .data_loader import ChinaMinuteDataLoader, DataSlice, WalkForwardFold


# ---------------------------------------------------------------------------
# Token tables
# ---------------------------------------------------------------------------

def build_token_tables(
    *,
    vocab_size: int,
    feat_offset: int,
    arity_map: dict[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build arity / stack-delta tables consumed by PPO sampling."""
    token_arity = torch.zeros(vocab_size, dtype=torch.long)
    token_arity[:feat_offset] = 0
    token_delta = torch.ones(vocab_size, dtype=torch.long)
    for token, arity in arity_map.items():
        if 0 <= token < vocab_size:
            arity_int = int(arity)
            token_arity[token] = arity_int
            token_delta[token] = 1 - arity_int
    return token_arity, token_delta


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FormulaEvaluation:
    """Reward and selection metrics for one candidate formula."""
    reward: float
    selection_score: Optional[float]
    mean_return: float
    train_score: Optional[float] = None
    val_score: Optional[float] = None


@dataclass
class RolloutBatch:
    seqs: torch.Tensor
    rollout_inputs: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    stack_depth_steps: list[torch.Tensor]


@dataclass
class EvaluationSnapshot:
    label: str
    score: float
    mean_return: float
    sharpe: float
    max_drawdown: float


@dataclass
class TrainingResult:
    best_score: float
    best_formula: Optional[list[int]]
    history: dict[str, list[Any]]
    evaluations: list[EvaluationSnapshot]


# ---------------------------------------------------------------------------
# Reward orchestrator
# ---------------------------------------------------------------------------

class FormulaRewardOrchestrator:
    """Score candidate formulas against backtest slices."""

    def __init__(
        self,
        *,
        vm,
        backtest_engine,
        train_slice: DataSlice,
        val_slice: Optional[DataSlice],
        walk_forward_folds: list[WalkForwardFold],
        use_wfo: bool,
        reward_mode: str = "selection",
    ):
        self._vm = vm
        self._bt = backtest_engine
        self._train = train_slice
        self._val = val_slice
        self._wf_folds = walk_forward_folds
        self._use_wfo = use_wfo
        mode = reward_mode.strip().lower()
        if mode not in {"train", "selection"}:
            raise ValueError(f"Unsupported reward_mode={reward_mode!r}")
        self._mode = mode

        if self._use_wfo:
            score_split = "train" if self._mode == "train" else "val"
            has_window = any(
                getattr(f, score_split).end_idx > getattr(f, score_split).start_idx
                for f in self._wf_folds
            )
            if not has_window:
                raise ValueError(
                    f"Walk-forward requires non-empty {score_split} windows "
                    f"for reward_mode={self._mode!r}."
                )

    @staticmethod
    def _to_float(score: object) -> float:
        return float(score.item()) if hasattr(score, "item") else float(score)

    @torch.no_grad()
    def evaluate_formula(self, formula: list[int], full_feat: torch.Tensor) -> FormulaEvaluation:
        res = self._vm.execute(formula, full_feat)
        if res is None:
            return FormulaEvaluation(reward=-5.0, selection_score=None, mean_return=0.0)
        if res.std() < 1e-4:
            return FormulaEvaluation(reward=-2.0, selection_score=None, mean_return=0.0)
        return self._eval_wfo(res) if self._use_wfo else self._eval_tv(res)

    def _eval_wfo(self, res: torch.Tensor) -> FormulaEvaluation:
        scores, rets = [], []
        split_name = "train" if self._mode == "train" else "val"
        for fold in self._wf_folds:
            s = getattr(fold, split_name)
            if s.end_idx <= s.start_idx:
                continue
            r = res[:, s.start_idx:s.end_idx]
            if r.numel() == 0:
                continue
            result = self._bt.evaluate(r, s.raw_data_cache, s.target_ret)
            scores.append(self._to_float(result.score))
            rets.append(result.mean_return)
        if not scores:
            return FormulaEvaluation(reward=-2.0, selection_score=None, mean_return=0.0)
        reward = sum(scores) / len(scores)
        mr = sum(rets) / len(rets)
        return FormulaEvaluation(
            reward=reward, selection_score=reward, mean_return=mr,
            train_score=reward if self._mode == "train" else None,
            val_score=reward if self._mode == "selection" else None,
        )

    def _eval_tv(self, res: torch.Tensor) -> FormulaEvaluation:
        r_train = res[:, self._train.start_idx:self._train.end_idx]
        if r_train.numel() == 0 or r_train.std() < 1e-4:
            return FormulaEvaluation(reward=-2.0, selection_score=None, mean_return=0.0)
        tr = self._bt.evaluate(r_train, self._train.raw_data_cache, self._train.target_ret)
        ts = self._to_float(tr.score)
        sel, mr, vs = ts, float(tr.mean_return), None

        if self._val and self._val.end_idx > self._val.start_idx:
            r_val = res[:, self._val.start_idx:self._val.end_idx]
            if r_val.numel() > 0:
                vr = self._bt.evaluate(r_val, self._val.raw_data_cache, self._val.target_ret)
                vs = self._to_float(vr.score)
                if self._mode == "selection":
                    sel, mr = vs, float(vr.mean_return)

        reward = ts if self._mode == "train" else sel
        return FormulaEvaluation(
            reward=reward, selection_score=sel, mean_return=mr,
            train_score=ts, val_score=vs,
        )


# ---------------------------------------------------------------------------
# PPO training loop
# ---------------------------------------------------------------------------

class _PpoLoop:
    """PPO sampling + update loop (internal)."""

    def __init__(self, *, model, optimizer, bos_id, token_arity, token_delta,
                 device, reward_orch, use_lord=False, lord_opt=None):
        self.model = model
        self.optimizer = optimizer
        self.bos_id = int(bos_id)
        self.token_arity = token_arity
        self.token_delta = token_delta
        self.device = device
        self.reward_orch = reward_orch
        self.use_lord = use_lord
        self.lord_opt = lord_opt

    def run(self, *, full_feat, train_steps, batch_size, max_len,
            ppo_epochs, clip_eps, value_coef, entropy_coef, max_grad_norm,
            rank_monitor=None, rank_every=100, on_new_best=None):
        history = self._init_history()
        best_score, best_formula = -float("inf"), None

        pbar = tqdm(range(train_steps))
        for step in pbar:
            rollout = self._sample(batch_size, max_len)
            rewards, t_scores, v_scores, best_score, best_formula = self._evaluate(
                rollout.seqs, full_feat, best_score, best_formula, on_new_best,
            )
            returns_steps, advantages = self._advantages(rewards, rollout.old_values, max_len)
            pl, vl, ent = self._ppo_update(
                rollout, returns_steps, advantages, max_len,
                ppo_epochs, clip_eps, value_coef, entropy_coef, max_grad_norm,
            )
            sr = None
            if rank_monitor and step % rank_every == 0:
                sr = float(rank_monitor.compute())
            self._record(history, step, best_score, rewards, t_scores, v_scores, pl, vl, ent, sr)
            pbar.set_postfix(self._postfix(best_score, rewards, pl, vl, sr, t_scores, v_scores))

        return best_score, best_formula, history

    @staticmethod
    def _init_history():
        return {k: [] for k in (
            "step", "avg_reward", "best_score", "stable_rank",
            "avg_train_score", "avg_val_score", "policy_loss", "value_loss", "entropy",
        )}

    def _sample(self, bs, max_len):
        inp = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self.device)
        sd = torch.zeros(bs, dtype=torch.long, device=self.device)
        lps, vals, toks, sds = [], [], [], []
        for t in range(max_len):
            logits, v, _ = self.model(inp)
            sds.append(sd.clone())
            vals.append(v.squeeze(-1).detach())
            rem = max_len - t
            mask = self._legal_mask(sd, rem)
            logits = logits.masked_fill(~mask, -1e9)
            dist = Categorical(logits=logits)
            a = dist.sample()
            lps.append(dist.log_prob(a).detach())
            toks.append(a)
            sd = sd + self.token_delta[a]
            inp = torch.cat([inp, a.unsqueeze(1)], dim=1)
        return RolloutBatch(
            seqs=torch.stack(toks, 1), rollout_inputs=inp.detach(),
            old_log_probs=torch.stack(lps, 1).detach(),
            old_values=torch.stack(vals, 1), stack_depth_steps=sds,
        )

    def _evaluate(self, seqs, full_feat, best_score, best_formula, on_new_best):
        bs = seqs.shape[0]
        rewards = torch.zeros(bs, device=self.device)
        ts, vs = [], []
        for i in range(bs):
            f = seqs[i].tolist()
            ev = self.reward_orch.evaluate_formula(f, full_feat)
            rewards[i] = ev.reward
            if ev.train_score is not None:
                ts.append(ev.train_score)
            if ev.val_score is not None:
                vs.append(ev.val_score)
            if ev.selection_score is not None and ev.selection_score > best_score:
                best_score = ev.selection_score
                best_formula = f
                if on_new_best:
                    on_new_best(best_score, ev.mean_return, f)
        return rewards, ts, vs, best_score, best_formula

    @staticmethod
    def _advantages(rewards, old_values, max_len):
        ret = torch.nan_to_num(rewards.detach(), nan=-2.0, posinf=5.0, neginf=-5.0)
        ret_steps = ret.unsqueeze(1).expand(-1, max_len)
        adv = ret_steps - old_values
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-5)
        return ret_steps, adv.detach()

    def _ppo_update(self, rollout, ret_steps, adv, max_len,
                    epochs, clip_eps, value_coef, entropy_coef, max_grad_norm):
        pl = vl = ent = float("nan")
        for _ in range(max(1, epochs)):
            nlp, vp, eb = self._policy_tensors(rollout, max_len)
            ratio = torch.exp(nlp - rollout.old_log_probs)
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            ploss = -torch.min(s1, s2).mean()
            vloss = F.mse_loss(vp, ret_steps)
            loss = ploss + value_coef * vloss - entropy_coef * eb
            self.optimizer.zero_grad()
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()
            if self.use_lord and self.lord_opt:
                self.lord_opt.step()
            pl, vl, ent = ploss.item(), vloss.item(), eb.item()
        return pl, vl, ent

    def _policy_tensors(self, rollout, max_len):
        nlps, vps, ents = [], [], []
        for t in range(max_len):
            prefix = rollout.rollout_inputs[:, :t + 1]
            logits, v, _ = self.model(prefix)
            rem = max_len - t
            mask = self._legal_mask(rollout.stack_depth_steps[t], rem)
            logits = logits.masked_fill(~mask, -1e9)
            dist = Categorical(logits=logits)
            nlps.append(dist.log_prob(rollout.seqs[:, t]))
            vps.append(v.squeeze(-1))
            ents.append(dist.entropy())
        return torch.stack(nlps, 1), torch.stack(vps, 1), torch.stack(ents, 1).mean()

    def _legal_mask(self, sd, remaining):
        legal = sd.unsqueeze(1) >= self.token_arity.unsqueeze(0)
        nd = sd.unsqueeze(1) + self.token_delta.unsqueeze(0)
        legal = legal & (nd <= remaining) if remaining > 1 else legal & (nd == 1)
        legal[:, self.bos_id] = False
        return legal

    @staticmethod
    def _avg_or_nan(values):
        return sum(values) / len(values) if values else float("nan")

    @staticmethod
    def _record(h, step, best, rewards, ts, vs, pl, vl, ent, sr):
        h["step"].append(step)
        h["avg_reward"].append(float(rewards.mean().item()))
        h["best_score"].append(best)
        h["policy_loss"].append(pl)
        h["value_loss"].append(vl)
        h["entropy"].append(ent)
        h["avg_train_score"].append(_PpoLoop._avg_or_nan(ts))
        h["avg_val_score"].append(_PpoLoop._avg_or_nan(vs))
        h["stable_rank"].append(float("nan") if sr is None else sr)

    @staticmethod
    def _postfix(best, rewards, pl, vl, sr, ts, vs):
        p = {"AvgRew": f"{rewards.mean():.3f}", "Best": f"{best:.3f}",
             "PLoss": f"{pl:.3f}", "VLoss": f"{vl:.3f}"}
        if sr is not None:
            p["Rank"] = f"{sr:.2f}"
        at, av = _PpoLoop._avg_or_nan(ts), _PpoLoop._avg_or_nan(vs)
        if not math.isnan(at):
            p["Train"] = f"{at:.3f}"
        if not math.isnan(av):
            p["Val"] = f"{av:.3f}"
        return p


# ---------------------------------------------------------------------------
# TrainingWorkflow  (top-level entry-point)
# ---------------------------------------------------------------------------

class TrainingWorkflow:
    """End-to-end alpha mining: load data → PPO train → evaluate → save."""

    def __init__(
        self,
        *,
        loader: ChinaMinuteDataLoader,
        model,
        optimizer,
        vm,
        backtest_engine,
        bos_id: int,
        token_arity: torch.Tensor,
        token_delta: torch.Tensor,
        device: torch.device,
        use_lord: bool = False,
        lord_opt=None,
        rank_monitor=None,
    ):
        self.loader = loader
        self.model = model
        self.optimizer = optimizer
        self.vm = vm
        self.bt = backtest_engine
        self.bos_id = int(bos_id)
        self.token_arity = token_arity
        self.token_delta = token_delta
        self.device = device
        self.use_lord = use_lord
        self.lord_opt = lord_opt
        self.rank_monitor = rank_monitor

        self.splits = loader.train_val_test_split()
        self.train_slice = self.splits.get("train")
        self.val_slice = self.splits.get("val")
        self.test_slice = self.splits.get("test")
        self.wf_folds = loader.walk_forward_splits() if ModelConfig.CN_WALK_FORWARD else []
        self.use_wfo = ModelConfig.CN_WALK_FORWARD and len(self.wf_folds) > 0

    def run(
        self,
        *,
        strategy_path: str = "",
        history_path: str = "training_history.json",
        on_new_best: Optional[Callable[[float, float, list[int]], None]] = None,
    ) -> TrainingResult:
        full_feat = self.loader.feat_tensor
        if full_feat is None or self.loader.raw_data_cache is None or self.loader.target_ret is None:
            raise ValueError("Data not loaded.")

        train_slice = self.train_slice or self.loader.get_slice(0, full_feat.shape[-1])
        orch = FormulaRewardOrchestrator(
            vm=self.vm, backtest_engine=self.bt,
            train_slice=train_slice, val_slice=self.val_slice,
            walk_forward_folds=self.wf_folds, use_wfo=self.use_wfo,
            reward_mode=ModelConfig.CN_REWARD_MODE,
        )
        ppo = _PpoLoop(
            model=self.model, optimizer=self.optimizer,
            bos_id=self.bos_id, token_arity=self.token_arity,
            token_delta=self.token_delta, device=self.device,
            reward_orch=orch, use_lord=self.use_lord, lord_opt=self.lord_opt,
        )
        best_score, best_formula, history = ppo.run(
            full_feat=full_feat,
            train_steps=ModelConfig.TRAIN_STEPS,
            batch_size=ModelConfig.BATCH_SIZE,
            max_len=ModelConfig.MAX_FORMULA_LEN,
            ppo_epochs=ModelConfig.PPO_EPOCHS,
            clip_eps=ModelConfig.PPO_CLIP_EPS,
            value_coef=ModelConfig.PPO_VALUE_COEF,
            entropy_coef=ModelConfig.PPO_ENTROPY_COEF,
            max_grad_norm=ModelConfig.PPO_MAX_GRAD_NORM,
            rank_monitor=self.rank_monitor if self.use_lord else None,
            on_new_best=on_new_best,
        )

        # Save
        sp = strategy_path or ModelConfig.STRATEGY_FILE
        if best_formula is not None:
            Path(sp).write_text(json.dumps(best_formula), encoding="utf-8")
        Path(history_path).write_text(json.dumps(history), encoding="utf-8")

        evals = self._final_eval(best_formula, full_feat)
        return TrainingResult(
            best_score=best_score, best_formula=best_formula,
            history=history, evaluations=evals,
        )

    def _final_eval(self, formula, full_feat):
        if not formula or self.use_wfo:
            return []
        res = self.vm.execute(formula, full_feat)
        if res is None:
            return []
        snaps = []
        for label, sl in [("Train", self.train_slice), ("Val", self.val_slice), ("Test", self.test_slice)]:
            if sl is None:
                continue
            sig = res[:, sl.start_idx:sl.end_idx]
            r = self.bt.evaluate(sig, sl.raw_data_cache, sl.target_ret, return_details=True)
            m = r.metrics or {}
            snaps.append(EvaluationSnapshot(
                label=label,
                score=float(r.score.item()) if hasattr(r.score, "item") else float(r.score),
                mean_return=float(r.mean_return),
                sharpe=float(m.get("sharpe", float("nan"))),
                max_drawdown=float(m.get("max_drawdown", float("nan"))),
            ))
        return snaps

    def window_descriptions(self) -> list[str]:
        if self.use_wfo:
            return [f"   Walk-forward: {len(self.wf_folds)} folds"]
        lines = []
        for label, sl in [("Train", self.train_slice), ("Val", self.val_slice), ("Test", self.test_slice)]:
            if sl:
                lines.append(f"   {label}: {sl.dates.min()} -> {sl.dates.max()}")
        return lines


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_training_workflow(
    *,
    use_lord: bool = True,
    lord_decay_rate: float = 1e-3,
    lord_num_iterations: int = 5,
    loader: Optional[ChinaMinuteDataLoader] = None,
    data_kwargs: Optional[dict[str, Any]] = None,
) -> TrainingWorkflow:
    """One-call factory: build everything from config + optional overrides."""
    from .model import NeuralSymbolicAlphaGenerator, NewtonSchulzLowRankDecay, StableRankMonitor
    from .backtest import ChinaBacktest
    from .vm import StackVM

    loader = loader or ChinaMinuteDataLoader()
    if loader.feat_tensor is None:
        kw = data_kwargs or {
            "codes": ModelConfig.CN_CODES,
            "years": ModelConfig.CN_MINUTE_YEARS,
            "start_date": ModelConfig.CN_MINUTE_START_DATE,
            "end_date": ModelConfig.CN_MINUTE_END_DATE,
            "signal_time": ModelConfig.CN_SIGNAL_TIME,
            "exit_time": ModelConfig.CN_EXIT_TIME,
            "pool_size": ModelConfig.CN_POOL_SIZE,
        }
        loader.load_data(**kw)

    model = NeuralSymbolicAlphaGenerator().to(ModelConfig.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    vm = StackVM()
    bt = ChinaBacktest()

    ta, td = build_token_tables(
        vocab_size=model.vocab_size, feat_offset=vm.feat_offset, arity_map=vm.arity_map,
    )

    lord_opt = rank_mon = None
    if use_lord:
        lord_opt = NewtonSchulzLowRankDecay(
            model.named_parameters(), decay_rate=lord_decay_rate,
            num_iterations=lord_num_iterations,
            target_keywords=["q_proj", "k_proj", "attention"],
        )
        rank_mon = StableRankMonitor(
            model, target_keywords=["attention", "in_proj", "out_proj"],
        )

    return TrainingWorkflow(
        loader=loader, model=model, optimizer=optimizer, vm=vm,
        backtest_engine=bt, bos_id=model.bos_id,
        token_arity=ta.to(ModelConfig.DEVICE),
        token_delta=td.to(ModelConfig.DEVICE),
        device=ModelConfig.DEVICE,
        use_lord=use_lord, lord_opt=lord_opt, rank_monitor=rank_mon,
    )
