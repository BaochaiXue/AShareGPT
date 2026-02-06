from typing import Optional, Any
import json

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

from .config import ModelConfig
from .data_loader import ChinaMinuteDataLoader
from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .vm import StackVM
from .backtest import ChinaBacktest


class AlphaEngine:
    """
    Core training engine for AlphaGPT on A-share minute data.
    """

    def __init__(self, use_lord_regularization: bool = True,
                 lord_decay_rate: float = 1e-3,
                 lord_num_iterations: int = 5):
        self.loader = ChinaMinuteDataLoader()
        self.loader.load_data(
            codes=ModelConfig.CN_CODES,
            years=ModelConfig.CN_MINUTE_YEARS,
            start_date=ModelConfig.CN_MINUTE_START_DATE,
            end_date=ModelConfig.CN_MINUTE_END_DATE,
            signal_time=ModelConfig.CN_SIGNAL_TIME,
            exit_time=ModelConfig.CN_EXIT_TIME,
            limit_codes=ModelConfig.CN_MAX_CODES,
        )

        self.model = AlphaGPT().to(ModelConfig.DEVICE)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        self.use_lord = use_lord_regularization
        if self.use_lord:
            self.lord_opt: Optional[NewtonSchulzLowRankDecay] = NewtonSchulzLowRankDecay(
                self.model.named_parameters(),
                decay_rate=lord_decay_rate,
                num_iterations=lord_num_iterations,
                target_keywords=["q_proj", "k_proj", "attention", "qk_norm"],
            )
            self.rank_monitor: Optional[StableRankMonitor] = StableRankMonitor(
                self.model,
                target_keywords=["attention", "in_proj", "out_proj"],
            )
        else:
            self.lord_opt = None
            self.rank_monitor = None

        self.vm = StackVM()
        self.bt = ChinaBacktest()
        self.feat_offset = self.vm.feat_offset
        self.vocab_size = self.model.vocab_size
        self.bos_id = self.model.bos_id
        self.token_arity = self._build_token_arity().to(ModelConfig.DEVICE)
        self.token_delta = self._build_token_delta().to(ModelConfig.DEVICE)

        self.splits = self.loader.train_val_test_split()
        self.train_slice = self.splits.get("train")
        self.val_slice = self.splits.get("val")
        self.test_slice = self.splits.get("test")
        self.walk_forward_folds = self.loader.walk_forward_splits() if ModelConfig.CN_WALK_FORWARD else []
        self.use_wfo = ModelConfig.CN_WALK_FORWARD and len(self.walk_forward_folds) > 0

        self.best_score: float = -float('inf')
        self.best_formula: Optional[list[int]] = None

        self.training_history: dict[str, list[Any]] = {
            'step': [],
            'avg_reward': [],
            'best_score': [],
            'stable_rank': [],
            'avg_val_score': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
        }

    def _build_token_arity(self) -> torch.Tensor:
        token_arity = torch.zeros(self.vocab_size, dtype=torch.long)
        # Default to "always legal" for feature tokens.
        token_arity[: self.feat_offset] = 0
        # Fill operator arity from VM definition.
        for token, arity in self.vm.arity_map.items():
            if 0 <= token < self.vocab_size:
                token_arity[token] = int(arity)
        return token_arity

    def _build_token_delta(self) -> torch.Tensor:
        # Feature token pushes one tensor onto stack.
        token_delta = torch.ones(self.vocab_size, dtype=torch.long)
        # Operator pops `arity`, then pushes one result => delta = 1 - arity.
        for token, arity in self.vm.arity_map.items():
            if 0 <= token < self.vocab_size:
                token_delta[token] = 1 - int(arity)
        return token_delta

    def _legal_action_mask(self, stack_depth: torch.Tensor, remaining_steps: int) -> torch.Tensor:
        # A token is legal iff current depth >= token arity (no underflow).
        legal = stack_depth.unsqueeze(1) >= self.token_arity.unsqueeze(0)
        next_depth = stack_depth.unsqueeze(1) + self.token_delta.unsqueeze(0)
        # Keep only actions that can still finish with stack depth 1.
        if remaining_steps > 1:
            legal = legal & (next_depth <= remaining_steps)
        else:
            legal = legal & (next_depth == 1)
        legal[:, self.bos_id] = False
        return legal

    def train(self) -> None:
        print("🚀 Starting Alpha Mining with PPO + LoRD..." if self.use_lord else "🚀 Starting Alpha Mining with PPO...")
        if self.use_lord:
            print("   LoRD Regularization enabled")
            print("   Target keywords: ['q_proj', 'k_proj', 'attention', 'qk_norm']")
        if self.use_wfo:
            print(f"   Walk-forward validation: {len(self.walk_forward_folds)} folds")
        elif self.train_slice:
            print(f"   Train window: {self.train_slice.dates.min()} -> {self.train_slice.dates.max()}")
            if self.val_slice:
                print(f"   Val window:   {self.val_slice.dates.min()} -> {self.val_slice.dates.max()}")
            if self.test_slice:
                print(f"   Test window:  {self.test_slice.dates.min()} -> {self.test_slice.dates.max()}")

        pbar = tqdm(range(ModelConfig.TRAIN_STEPS))
        full_feat = self.loader.feat_tensor
        if full_feat is None or self.loader.raw_data_cache is None or self.loader.target_ret is None:
            raise ValueError("Data not loaded. Check data loader.")

        for step in pbar:
            bs = ModelConfig.BATCH_SIZE
            inp = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=ModelConfig.DEVICE)
            stack_depth = torch.zeros(bs, dtype=torch.long, device=ModelConfig.DEVICE)

            old_log_probs: list[torch.Tensor] = []
            old_values_steps: list[torch.Tensor] = []
            tokens_list: list[torch.Tensor] = []
            stack_depth_steps: list[torch.Tensor] = []

            for t in range(ModelConfig.MAX_FORMULA_LEN):
                logits, value_t, _ = self.model(inp)
                stack_depth_steps.append(stack_depth.clone())
                old_values_steps.append(value_t.squeeze(-1).detach())
                remaining_steps = ModelConfig.MAX_FORMULA_LEN - t
                legal_mask = self._legal_action_mask(stack_depth, remaining_steps)
                masked_logits = logits.masked_fill(~legal_mask, -1e9)
                dist = Categorical(logits=masked_logits)
                action = dist.sample()

                old_log_probs.append(dist.log_prob(action).detach())
                tokens_list.append(action)
                stack_depth = stack_depth + self.token_delta[action]
                inp = torch.cat([inp, action.unsqueeze(1)], dim=1)

            seqs = torch.stack(tokens_list, dim=1)
            rollout_inputs = inp.detach()
            old_log_probs_tensor = torch.stack(old_log_probs, dim=1).detach()
            old_values = torch.stack(old_values_steps, dim=1)

            rewards = torch.zeros(bs, device=ModelConfig.DEVICE)
            val_scores: list[float] = []

            for i in range(bs):
                formula = seqs[i].tolist()

                res = self.vm.execute(formula, full_feat)
                if res is None:
                    rewards[i] = -5.0
                    continue

                if res.std() < 1e-4:
                    rewards[i] = -2.0
                    continue

                ret_val = 0.0
                selection_score = None

                if self.use_wfo:
                    fold_scores = []
                    fold_returns = []
                    for fold in self.walk_forward_folds:
                        if fold.val.end_idx <= fold.val.start_idx:
                            continue
                        res_val = res[:, fold.val.start_idx:fold.val.end_idx]
                        if res_val.numel() == 0:
                            continue
                        result = self.bt.evaluate(res_val, fold.val.raw_data_cache, fold.val.target_ret)
                        fold_scores.append(result.score)
                        fold_returns.append(result.mean_return)
                    if not fold_scores:
                        rewards[i] = -2.0
                        continue
                    reward = torch.stack(fold_scores).mean()
                    rewards[i] = reward
                    selection_score = reward
                    ret_val = float(sum(fold_returns) / len(fold_returns))
                else:
                    train_slice = self.train_slice
                    if train_slice is None:
                        train_slice = self.loader.get_slice(0, res.shape[1])
                    res_train = res[:, train_slice.start_idx:train_slice.end_idx]
                    if res_train.std() < 1e-4:
                        rewards[i] = -2.0
                        continue
                    train_result = self.bt.evaluate(
                        res_train,
                        train_slice.raw_data_cache,
                        train_slice.target_ret,
                    )
                    rewards[i] = train_result.score
                    selection_score = train_result.score
                    ret_val = train_result.mean_return

                    if self.val_slice and self.val_slice.end_idx > self.val_slice.start_idx:
                        res_val = res[:, self.val_slice.start_idx:self.val_slice.end_idx]
                        if res_val.numel() > 0:
                            val_result = self.bt.evaluate(
                                res_val,
                                self.val_slice.raw_data_cache,
                                self.val_slice.target_ret,
                            )
                            selection_score = val_result.score
                            ret_val = val_result.mean_return
                            val_scores.append(val_result.score.item())

                if selection_score is not None and selection_score.item() > self.best_score:
                    self.best_score = selection_score.item()
                    self.best_formula = formula
                    tqdm.write(f"[!] New King: Score {selection_score:.2f} | Ret {ret_val:.2%} | Formula {formula}")

            returns = torch.nan_to_num(rewards.detach(), nan=-2.0, posinf=5.0, neginf=-5.0)
            returns_steps = returns.unsqueeze(1).expand(-1, ModelConfig.MAX_FORMULA_LEN)
            advantages = returns_steps - old_values
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-5)
            advantages = advantages.detach()

            policy_loss_value = float("nan")
            value_loss_value = float("nan")
            entropy_value = float("nan")

            for _ in range(max(1, ModelConfig.PPO_EPOCHS)):
                new_log_probs_steps: list[torch.Tensor] = []
                values_pred_steps: list[torch.Tensor] = []
                entropy_steps: list[torch.Tensor] = []

                for t in range(ModelConfig.MAX_FORMULA_LEN):
                    prefix = rollout_inputs[:, : t + 1]
                    logits_t, value_t, _ = self.model(prefix)
                    remaining_steps = ModelConfig.MAX_FORMULA_LEN - t
                    legal_mask_t = self._legal_action_mask(stack_depth_steps[t], remaining_steps)
                    masked_logits_t = logits_t.masked_fill(~legal_mask_t, -1e9)
                    dist_t = Categorical(logits=masked_logits_t)
                    actions_t = seqs[:, t]
                    new_log_probs_steps.append(dist_t.log_prob(actions_t))
                    values_pred_steps.append(value_t.squeeze(-1))
                    entropy_steps.append(dist_t.entropy())

                new_log_probs = torch.stack(new_log_probs_steps, dim=1)
                ratio = torch.exp(new_log_probs - old_log_probs_tensor)

                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - ModelConfig.PPO_CLIP_EPS,
                    1.0 + ModelConfig.PPO_CLIP_EPS,
                ) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                values_pred = torch.stack(values_pred_steps, dim=1)
                value_loss = F.mse_loss(values_pred, returns_steps)

                entropy_bonus = torch.stack(entropy_steps, dim=1).mean()
                loss = (
                    policy_loss
                    + ModelConfig.PPO_VALUE_COEF * value_loss
                    - ModelConfig.PPO_ENTROPY_COEF * entropy_bonus
                )

                self.opt.zero_grad()
                loss.backward()
                if ModelConfig.PPO_MAX_GRAD_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), ModelConfig.PPO_MAX_GRAD_NORM)
                self.opt.step()

                if self.use_lord and self.lord_opt:
                    self.lord_opt.step()

                policy_loss_value = policy_loss.item()
                value_loss_value = value_loss.item()
                entropy_value = entropy_bonus.item()

            avg_reward = rewards.mean().item()
            postfix_dict = {'AvgRew': f"{avg_reward:.3f}", 'BestScore': f"{self.best_score:.3f}"}
            postfix_dict['PLoss'] = f"{policy_loss_value:.3f}"
            postfix_dict['VLoss'] = f"{value_loss_value:.3f}"

            if self.use_lord and self.rank_monitor and step % 100 == 0:
                stable_rank = self.rank_monitor.compute()
                postfix_dict['Rank'] = f"{stable_rank:.2f}"
                self.training_history['stable_rank'].append(stable_rank)
            if val_scores:
                avg_val = float(sum(val_scores) / len(val_scores))
                postfix_dict['Val'] = f"{avg_val:.3f}"
                self.training_history['avg_val_score'].append(avg_val)
            else:
                self.training_history['avg_val_score'].append(float("nan"))

            self.training_history['step'].append(step)
            self.training_history['avg_reward'].append(avg_reward)
            self.training_history['best_score'].append(self.best_score)
            self.training_history['policy_loss'].append(policy_loss_value)
            self.training_history['value_loss'].append(value_loss_value)
            self.training_history['entropy'].append(entropy_value)

            pbar.set_postfix(postfix_dict)

        with open(ModelConfig.STRATEGY_FILE, "w") as f:
            json.dump(self.best_formula, f)

        with open("training_history.json", "w") as f:
            json.dump(self.training_history, f)

        print("\n✓ Training completed!")
        print(f"  Best score: {self.best_score:.4f}")
        print(f"  Best formula: {self.best_formula}")

        if self.best_formula and not self.use_wfo:
            res = self.vm.execute(self.best_formula, full_feat)
            if res is not None:
                def _print_eval(label: str, result) -> None:
                    metrics = result.metrics or {}
                    sharpe = metrics.get("sharpe", float("nan"))
                    max_dd = metrics.get("max_drawdown", float("nan"))
                    print(f"  {label}: Score {result.score.item():.4f} | MeanRet {result.mean_return:.2%} | Sharpe {sharpe:.2f} | MaxDD {max_dd:.2%}")

                if self.train_slice:
                    train_res = res[:, self.train_slice.start_idx:self.train_slice.end_idx]
                    train_result = self.bt.evaluate(
                        train_res,
                        self.train_slice.raw_data_cache,
                        self.train_slice.target_ret,
                        return_details=True,
                    )
                    _print_eval("Train", train_result)
                if self.val_slice:
                    val_res = res[:, self.val_slice.start_idx:self.val_slice.end_idx]
                    val_result = self.bt.evaluate(
                        val_res,
                        self.val_slice.raw_data_cache,
                        self.val_slice.target_ret,
                        return_details=True,
                    )
                    _print_eval("Val", val_result)
                if self.test_slice:
                    test_res = res[:, self.test_slice.start_idx:self.test_slice.end_idx]
                    test_result = self.bt.evaluate(
                        test_res,
                        self.test_slice.raw_data_cache,
                        self.test_slice.target_ret,
                        return_details=True,
                    )
                    _print_eval("Test", test_result)
