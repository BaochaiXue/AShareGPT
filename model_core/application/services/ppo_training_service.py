from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

from .reward_orchestrator import FormulaRewardOrchestrator


@dataclass
class RolloutBatch:
    """Collected rollout tensors for one PPO step."""

    seqs: torch.Tensor
    rollout_inputs: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    stack_depth_steps: list[torch.Tensor]


@dataclass
class TrainingRunState:
    """Training outputs consumed by the compatibility engine."""

    best_score: float
    best_formula: Optional[list[int]]
    history: dict[str, list[Any]]


@dataclass
class TrainingStepSummary:
    """One training-step scalar summary for history and progress logging."""

    avg_reward: float
    policy_loss: float
    value_loss: float
    entropy: float
    avg_train_score: float
    avg_val_score: float
    stable_rank: Optional[float]


class PpoTrainingService:
    """PPO loop extracted from `AlphaEngine`."""

    def __init__(
        self,
        *,
        model,
        optimizer,
        bos_id: int,
        token_arity: torch.Tensor,
        token_delta: torch.Tensor,
        device: torch.device,
        reward_orchestrator: FormulaRewardOrchestrator,
        use_lord: bool = False,
        lord_opt=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.bos_id = int(bos_id)
        self.token_arity = token_arity
        self.token_delta = token_delta
        self.device = device
        self.reward_orchestrator = reward_orchestrator
        self.use_lord = use_lord
        self.lord_opt = lord_opt

    def train(
        self,
        *,
        full_feat: torch.Tensor,
        train_steps: int,
        batch_size: int,
        max_formula_len: int,
        ppo_epochs: int,
        ppo_clip_eps: float,
        ppo_value_coef: float,
        ppo_entropy_coef: float,
        ppo_max_grad_norm: float,
        rank_monitor=None,
        rank_every: int = 100,
        on_new_best: Optional[Callable[[float, float, list[int]], None]] = None,
    ) -> TrainingRunState:
        history = self._init_history()
        best_score = -float("inf")
        best_formula: Optional[list[int]] = None

        pbar = tqdm(range(train_steps))
        for step in pbar:
            rollout = self._sample_rollout(batch_size=batch_size, max_formula_len=max_formula_len)
            rewards, train_scores, val_scores, best_score, best_formula = self._evaluate_batch(
                seqs=rollout.seqs,
                full_feat=full_feat,
                best_score=best_score,
                best_formula=best_formula,
                on_new_best=on_new_best,
            )

            returns_steps, advantages = self._compute_advantages(
                rewards=rewards,
                old_values=rollout.old_values,
                max_formula_len=max_formula_len,
            )
            policy_loss, value_loss, entropy = self._run_ppo_updates(
                rollout=rollout,
                returns_steps=returns_steps,
                advantages=advantages,
                max_formula_len=max_formula_len,
                ppo_epochs=ppo_epochs,
                ppo_clip_eps=ppo_clip_eps,
                ppo_value_coef=ppo_value_coef,
                ppo_entropy_coef=ppo_entropy_coef,
                ppo_max_grad_norm=ppo_max_grad_norm,
            )

            summary = self._build_step_summary(
                rewards=rewards,
                train_scores=train_scores,
                val_scores=val_scores,
                policy_loss=policy_loss,
                value_loss=value_loss,
                entropy=entropy,
                rank_monitor=rank_monitor,
                step=step,
                rank_every=rank_every,
            )
            self._append_history(history=history, step=step, best_score=best_score, summary=summary)
            pbar.set_postfix(self._build_postfix(best_score=best_score, summary=summary))

        return TrainingRunState(best_score=best_score, best_formula=best_formula, history=history)

    @staticmethod
    def _init_history() -> dict[str, list[Any]]:
        return {
            "step": [],
            "avg_reward": [],
            "best_score": [],
            "stable_rank": [],
            "avg_train_score": [],
            "avg_val_score": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
        }

    @staticmethod
    def _score_average(scores: list[float]) -> float:
        if not scores:
            return float("nan")
        return float(sum(scores) / len(scores))

    def _build_step_summary(
        self,
        *,
        rewards: torch.Tensor,
        train_scores: list[float],
        val_scores: list[float],
        policy_loss: float,
        value_loss: float,
        entropy: float,
        rank_monitor,
        step: int,
        rank_every: int,
    ) -> TrainingStepSummary:
        stable_rank: Optional[float] = None
        if rank_monitor and step % rank_every == 0:
            stable_rank = float(rank_monitor.compute())
        return TrainingStepSummary(
            avg_reward=float(rewards.mean().item()),
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy=entropy,
            avg_train_score=self._score_average(train_scores),
            avg_val_score=self._score_average(val_scores),
            stable_rank=stable_rank,
        )

    @staticmethod
    def _append_history(
        *,
        history: dict[str, list[Any]],
        step: int,
        best_score: float,
        summary: TrainingStepSummary,
    ) -> None:
        history["step"].append(step)
        history["avg_reward"].append(summary.avg_reward)
        history["best_score"].append(best_score)
        history["policy_loss"].append(summary.policy_loss)
        history["value_loss"].append(summary.value_loss)
        history["entropy"].append(summary.entropy)
        history["avg_train_score"].append(summary.avg_train_score)
        history["avg_val_score"].append(summary.avg_val_score)
        if summary.stable_rank is not None:
            history["stable_rank"].append(summary.stable_rank)

    @staticmethod
    def _build_postfix(*, best_score: float, summary: TrainingStepSummary) -> dict[str, str]:
        postfix = {
            "AvgRew": f"{summary.avg_reward:.3f}",
            "BestScore": f"{best_score:.3f}",
            "PLoss": f"{summary.policy_loss:.3f}",
            "VLoss": f"{summary.value_loss:.3f}",
        }
        if summary.stable_rank is not None:
            postfix["Rank"] = f"{summary.stable_rank:.2f}"
        if summary.avg_train_score == summary.avg_train_score:
            postfix["Train"] = f"{summary.avg_train_score:.3f}"
        if summary.avg_val_score == summary.avg_val_score:
            postfix["Val"] = f"{summary.avg_val_score:.3f}"
        return postfix

    def _sample_rollout(self, *, batch_size: int, max_formula_len: int) -> RolloutBatch:
        inp = torch.full((batch_size, 1), self.bos_id, dtype=torch.long, device=self.device)
        stack_depth = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        old_log_probs: list[torch.Tensor] = []
        old_values_steps: list[torch.Tensor] = []
        tokens_list: list[torch.Tensor] = []
        stack_depth_steps: list[torch.Tensor] = []

        for t in range(max_formula_len):
            logits, value_t, _ = self.model(inp)
            stack_depth_steps.append(stack_depth.clone())
            old_values_steps.append(value_t.squeeze(-1).detach())

            remaining_steps = max_formula_len - t
            legal_mask = self._legal_action_mask(stack_depth=stack_depth, remaining_steps=remaining_steps)
            masked_logits = logits.masked_fill(~legal_mask, -1e9)
            dist = Categorical(logits=masked_logits)
            action = dist.sample()

            old_log_probs.append(dist.log_prob(action).detach())
            tokens_list.append(action)
            stack_depth = stack_depth + self.token_delta[action]
            inp = torch.cat([inp, action.unsqueeze(1)], dim=1)

        return RolloutBatch(
            seqs=torch.stack(tokens_list, dim=1),
            rollout_inputs=inp.detach(),
            old_log_probs=torch.stack(old_log_probs, dim=1).detach(),
            old_values=torch.stack(old_values_steps, dim=1),
            stack_depth_steps=stack_depth_steps,
        )

    def _evaluate_batch(
        self,
        *,
        seqs: torch.Tensor,
        full_feat: torch.Tensor,
        best_score: float,
        best_formula: Optional[list[int]],
        on_new_best: Optional[Callable[[float, float, list[int]], None]],
    ) -> tuple[torch.Tensor, list[float], list[float], float, Optional[list[int]]]:
        batch_size = seqs.shape[0]
        rewards = torch.zeros(batch_size, device=self.device)
        train_scores: list[float] = []
        val_scores: list[float] = []

        for i in range(batch_size):
            formula = seqs[i].tolist()
            eval_out = self.reward_orchestrator.evaluate_formula(formula, full_feat)
            rewards[i] = eval_out.reward
            if eval_out.train_score is not None:
                train_scores.append(eval_out.train_score)
            if eval_out.val_score is not None:
                val_scores.append(eval_out.val_score)
            if eval_out.selection_score is None:
                continue
            if eval_out.selection_score > best_score:
                best_score = eval_out.selection_score
                best_formula = formula
                if on_new_best:
                    on_new_best(best_score, eval_out.mean_return, formula)

        return rewards, train_scores, val_scores, best_score, best_formula

    def _compute_advantages(
        self,
        *,
        rewards: torch.Tensor,
        old_values: torch.Tensor,
        max_formula_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        returns = torch.nan_to_num(rewards.detach(), nan=-2.0, posinf=5.0, neginf=-5.0)
        returns_steps = returns.unsqueeze(1).expand(-1, max_formula_len)
        advantages = returns_steps - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-5)
        return returns_steps, advantages.detach()

    def _run_ppo_updates(
        self,
        *,
        rollout: RolloutBatch,
        returns_steps: torch.Tensor,
        advantages: torch.Tensor,
        max_formula_len: int,
        ppo_epochs: int,
        ppo_clip_eps: float,
        ppo_value_coef: float,
        ppo_entropy_coef: float,
        ppo_max_grad_norm: float,
    ) -> tuple[float, float, float]:
        policy_loss_value = float("nan")
        value_loss_value = float("nan")
        entropy_value = float("nan")

        for _ in range(max(1, ppo_epochs)):
            new_log_probs, values_pred, entropy_bonus = self._collect_policy_tensors(
                rollout=rollout,
                max_formula_len=max_formula_len,
            )
            ratio = torch.exp(new_log_probs - rollout.old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - ppo_clip_eps, 1.0 + ppo_clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values_pred, returns_steps)
            loss = policy_loss + ppo_value_coef * value_loss - ppo_entropy_coef * entropy_bonus
            self._apply_optimizer_step(loss=loss, ppo_max_grad_norm=ppo_max_grad_norm)

            policy_loss_value = float(policy_loss.item())
            value_loss_value = float(value_loss.item())
            entropy_value = float(entropy_bonus.item())

        return policy_loss_value, value_loss_value, entropy_value

    def _collect_policy_tensors(
        self,
        *,
        rollout: RolloutBatch,
        max_formula_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        new_log_probs_steps: list[torch.Tensor] = []
        values_pred_steps: list[torch.Tensor] = []
        entropy_steps: list[torch.Tensor] = []

        for t in range(max_formula_len):
            prefix = rollout.rollout_inputs[:, : t + 1]
            logits_t, value_t, _ = self.model(prefix)
            remaining_steps = max_formula_len - t
            legal_mask_t = self._legal_action_mask(
                stack_depth=rollout.stack_depth_steps[t],
                remaining_steps=remaining_steps,
            )
            masked_logits_t = logits_t.masked_fill(~legal_mask_t, -1e9)
            dist_t = Categorical(logits=masked_logits_t)
            actions_t = rollout.seqs[:, t]
            new_log_probs_steps.append(dist_t.log_prob(actions_t))
            values_pred_steps.append(value_t.squeeze(-1))
            entropy_steps.append(dist_t.entropy())

        new_log_probs = torch.stack(new_log_probs_steps, dim=1)
        values_pred = torch.stack(values_pred_steps, dim=1)
        entropy_bonus = torch.stack(entropy_steps, dim=1).mean()
        return new_log_probs, values_pred, entropy_bonus

    def _apply_optimizer_step(self, *, loss: torch.Tensor, ppo_max_grad_norm: float) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        if ppo_max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), ppo_max_grad_norm)
        self.optimizer.step()
        if self.use_lord and self.lord_opt:
            self.lord_opt.step()

    def _legal_action_mask(self, *, stack_depth: torch.Tensor, remaining_steps: int) -> torch.Tensor:
        legal = stack_depth.unsqueeze(1) >= self.token_arity.unsqueeze(0)
        next_depth = stack_depth.unsqueeze(1) + self.token_delta.unsqueeze(0)
        if remaining_steps > 1:
            legal = legal & (next_depth <= remaining_steps)
        else:
            legal = legal & (next_depth == 1)
        legal[:, self.bos_id] = False
        return legal
