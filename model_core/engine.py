from typing import Optional, Any
import json

import torch
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
                target_keywords=["q_proj", "k_proj"],
            )
        else:
            self.lord_opt = None
            self.rank_monitor = None

        self.vm = StackVM()
        self.bt = ChinaBacktest()

        self.best_score: float = -float('inf')
        self.best_formula: Optional[list[int]] = None

        self.training_history: dict[str, list[Any]] = {
            'step': [],
            'avg_reward': [],
            'best_score': [],
            'stable_rank': [],
        }

    def train(self) -> None:
        print("🚀 Starting Alpha Mining with LoRD Regularization..." if self.use_lord else "🚀 Starting Alpha Mining...")
        if self.use_lord:
            print("   LoRD Regularization enabled")
            print("   Target keywords: ['q_proj', 'k_proj', 'attention', 'qk_norm']")

        pbar = tqdm(range(ModelConfig.TRAIN_STEPS))

        for step in pbar:
            bs = ModelConfig.BATCH_SIZE
            inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)

            log_probs: list[torch.Tensor] = []
            tokens_list: list[torch.Tensor] = []

            for _ in range(ModelConfig.MAX_FORMULA_LEN):
                logits, _, _ = self.model(inp)
                dist = Categorical(logits=logits)
                action = dist.sample()

                log_probs.append(dist.log_prob(action))
                tokens_list.append(action)
                inp = torch.cat([inp, action.unsqueeze(1)], dim=1)

            seqs = torch.stack(tokens_list, dim=1)

            rewards = torch.zeros(bs, device=ModelConfig.DEVICE)

            for i in range(bs):
                formula = seqs[i].tolist()

                res = self.vm.execute(formula, self.loader.feat_tensor)
                if res is None:
                    rewards[i] = -5.0
                    continue

                if res.std() < 1e-4:
                    rewards[i] = -2.0
                    continue

                score, ret_val = self.bt.evaluate(res, self.loader.raw_data_cache, self.loader.target_ret)
                rewards[i] = score

                if score.item() > self.best_score:
                    self.best_score = score.item()
                    self.best_formula = formula
                    tqdm.write(f"[!] New King: Score {score:.2f} | Ret {ret_val:.2%} | Formula {formula}")

            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

            loss = torch.tensor(0.0, device=ModelConfig.DEVICE)
            for t in range(len(log_probs)):
                loss = loss + (-log_probs[t] * adv)

            loss = loss.mean()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            if self.use_lord and self.lord_opt:
                self.lord_opt.step()

            avg_reward = rewards.mean().item()
            postfix_dict = {'AvgRew': f"{avg_reward:.3f}", 'BestScore': f"{self.best_score:.3f}"}

            if self.use_lord and self.rank_monitor and step % 100 == 0:
                stable_rank = self.rank_monitor.compute()
                postfix_dict['Rank'] = f"{stable_rank:.2f}"
                self.training_history['stable_rank'].append(stable_rank)

            self.training_history['step'].append(step)
            self.training_history['avg_reward'].append(avg_reward)
            self.training_history['best_score'].append(self.best_score)

            pbar.set_postfix(postfix_dict)

        with open(ModelConfig.STRATEGY_FILE, "w") as f:
            json.dump(self.best_formula, f)

        with open("training_history.json", "w") as f:
            json.dump(self.training_history, f)

        print("\n✓ Training completed!")
        print(f"  Best score: {self.best_score:.4f}")
        print(f"  Best formula: {self.best_formula}")
