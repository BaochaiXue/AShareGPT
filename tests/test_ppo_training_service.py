from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_core.application.services.ppo_training_service import PpoTrainingService


class _DummyModel(nn.Module):
    def __init__(self, vocab_size: int, bos_id: int):
        super().__init__()
        self.logit_head = nn.Linear(vocab_size, vocab_size, bias=False)
        self.value_head = nn.Linear(vocab_size, 1, bias=False)
        with torch.no_grad():
            self.logit_head.weight.zero_()
            self.value_head.weight.zero_()
            # Make token 3 strongly preferred on BOS input for deterministic-ish tests.
            self.logit_head.weight[3, bos_id] = 20.0

    def forward(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        one_hot = F.one_hot(idx[:, -1], num_classes=self.logit_head.in_features).float()
        logits = self.logit_head(one_hot)
        value = self.value_head(one_hot)
        task_probs = torch.zeros((idx.shape[0], 1), dtype=one_hot.dtype, device=one_hot.device)
        return logits, value, task_probs


class _DummyRewardOrchestrator:
    @staticmethod
    def evaluate_formula(formula: list[int], full_feat: torch.Tensor):
        del full_feat
        score = float(formula[0])
        return SimpleNamespace(
            reward=score,
            train_score=score,
            val_score=score + 0.1,
            selection_score=score,
            mean_return=score / 10.0,
        )


def test_train_history_and_best_formula_shapes() -> None:
    torch.manual_seed(0)
    vocab_size = 5
    bos_id = 4
    model = _DummyModel(vocab_size=vocab_size, bos_id=bos_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    token_arity = torch.zeros(vocab_size, dtype=torch.long)
    token_delta = torch.ones(vocab_size, dtype=torch.long)

    service = PpoTrainingService(
        model=model,
        optimizer=optimizer,
        bos_id=bos_id,
        token_arity=token_arity,
        token_delta=token_delta,
        device=torch.device("cpu"),
        reward_orchestrator=_DummyRewardOrchestrator(),
    )

    state = service.train(
        full_feat=torch.zeros((1, 1, 1)),
        train_steps=3,
        batch_size=6,
        max_formula_len=1,
        ppo_epochs=1,
        ppo_clip_eps=0.2,
        ppo_value_coef=0.5,
        ppo_entropy_coef=0.01,
        ppo_max_grad_norm=1.0,
    )

    assert state.best_formula is not None
    assert len(state.best_formula) == 1
    assert len(state.history["step"]) == 3
    assert len(state.history["avg_reward"]) == 3
    assert len(state.history["avg_train_score"]) == 3
    assert len(state.history["avg_val_score"]) == 3
    assert len(state.history["policy_loss"]) == 3
    assert len(state.history["value_loss"]) == 3
    assert len(state.history["entropy"]) == 3


def test_on_new_best_callback_invoked() -> None:
    torch.manual_seed(0)
    vocab_size = 5
    bos_id = 4
    model = _DummyModel(vocab_size=vocab_size, bos_id=bos_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    token_arity = torch.zeros(vocab_size, dtype=torch.long)
    token_delta = torch.ones(vocab_size, dtype=torch.long)

    service = PpoTrainingService(
        model=model,
        optimizer=optimizer,
        bos_id=bos_id,
        token_arity=token_arity,
        token_delta=token_delta,
        device=torch.device("cpu"),
        reward_orchestrator=_DummyRewardOrchestrator(),
    )

    seen: list[tuple[float, float, list[int]]] = []

    def _on_new_best(score: float, mean_return: float, formula: list[int]) -> None:
        seen.append((score, mean_return, formula))

    service.train(
        full_feat=torch.zeros((1, 1, 1)),
        train_steps=2,
        batch_size=4,
        max_formula_len=1,
        ppo_epochs=1,
        ppo_clip_eps=0.2,
        ppo_value_coef=0.5,
        ppo_entropy_coef=0.01,
        ppo_max_grad_norm=1.0,
        on_new_best=_on_new_best,
    )

    assert seen
    best_score, mean_return, formula = seen[-1]
    assert isinstance(best_score, float)
    assert isinstance(mean_return, float)
    assert len(formula) == 1
