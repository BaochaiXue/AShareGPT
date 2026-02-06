from typing import Any, Optional
import json

from tqdm import tqdm

import torch

from .application.services import FormulaRewardOrchestrator, PpoTrainingService
from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .backtest import ChinaBacktest

from .data_loader import ChinaMinuteDataLoader
from .config import ModelConfig
from .vm import StackVM


class AlphaEngine:
    """Compatibility engine that delegates the PPO loop to application services."""

    def __init__(
        self,
        use_lord_regularization: bool = True,
        lord_decay_rate: float = 1e-3,
        lord_num_iterations: int = 5,
        *,
        loader: Optional[ChinaMinuteDataLoader] = None,
        model: Optional[AlphaGPT] = None,
        optimizer=None,
        vm: Optional[StackVM] = None,
        backtest: Optional[ChinaBacktest] = None,
        auto_load_data: bool = True,
        data_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.loader = loader or ChinaMinuteDataLoader()
        if auto_load_data and self.loader.feat_tensor is None:
            kwargs = data_kwargs or {
                "codes": ModelConfig.CN_CODES,
                "years": ModelConfig.CN_MINUTE_YEARS,
                "start_date": ModelConfig.CN_MINUTE_START_DATE,
                "end_date": ModelConfig.CN_MINUTE_END_DATE,
                "signal_time": ModelConfig.CN_SIGNAL_TIME,
                "exit_time": ModelConfig.CN_EXIT_TIME,
                "limit_codes": ModelConfig.CN_MAX_CODES,
            }
            self.loader.load_data(**kwargs)

        self.model = model or AlphaGPT().to(ModelConfig.DEVICE)
        self.opt = optimizer or torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        self.use_lord = bool(use_lord_regularization)
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

        self.vm = vm or StackVM()
        self.bt = backtest or ChinaBacktest()
        self.feat_offset = self.vm.feat_offset
        self.vocab_size = self.model.vocab_size
        self.bos_id = self.model.bos_id
        self.token_arity = self._build_token_arity().to(ModelConfig.DEVICE)
        self.token_delta = self._build_token_delta().to(ModelConfig.DEVICE)

        if self.loader.dates is None:
            raise ValueError("Data not loaded. Provide a loaded loader or enable auto_load_data.")

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
        token_arity[: self.feat_offset] = 0
        for token, arity in self.vm.arity_map.items():
            if 0 <= token < self.vocab_size:
                token_arity[token] = int(arity)
        return token_arity

    def _build_token_delta(self) -> torch.Tensor:
        token_delta = torch.ones(self.vocab_size, dtype=torch.long)
        for token, arity in self.vm.arity_map.items():
            if 0 <= token < self.vocab_size:
                token_delta[token] = 1 - int(arity)
        return token_delta

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

        full_feat = self.loader.feat_tensor
        if full_feat is None or self.loader.raw_data_cache is None or self.loader.target_ret is None:
            raise ValueError("Data not loaded. Check data loader.")

        train_slice = self.train_slice or self.loader.get_slice(0, full_feat.shape[1])
        reward_orchestrator = FormulaRewardOrchestrator(
            vm=self.vm,
            backtest_engine=self.bt,
            train_slice=train_slice,
            val_slice=self.val_slice,
            walk_forward_folds=self.walk_forward_folds,
            use_wfo=self.use_wfo,
        )
        training_service = PpoTrainingService(
            model=self.model,
            optimizer=self.opt,
            bos_id=self.bos_id,
            token_arity=self.token_arity,
            token_delta=self.token_delta,
            device=ModelConfig.DEVICE,
            reward_orchestrator=reward_orchestrator,
            use_lord=self.use_lord,
            lord_opt=self.lord_opt,
        )

        def _on_new_best(score: float, mean_return: float, formula: list[int]) -> None:
            tqdm.write(f"[!] New King: Score {score:.2f} | Ret {mean_return:.2%} | Formula {formula}")

        run_state = training_service.train(
            full_feat=full_feat,
            train_steps=ModelConfig.TRAIN_STEPS,
            batch_size=ModelConfig.BATCH_SIZE,
            max_formula_len=ModelConfig.MAX_FORMULA_LEN,
            ppo_epochs=ModelConfig.PPO_EPOCHS,
            ppo_clip_eps=ModelConfig.PPO_CLIP_EPS,
            ppo_value_coef=ModelConfig.PPO_VALUE_COEF,
            ppo_entropy_coef=ModelConfig.PPO_ENTROPY_COEF,
            ppo_max_grad_norm=ModelConfig.PPO_MAX_GRAD_NORM,
            rank_monitor=self.rank_monitor if self.use_lord else None,
            rank_every=100,
            on_new_best=_on_new_best,
        )

        self.best_score = run_state.best_score
        self.best_formula = run_state.best_formula
        self.training_history = run_state.history

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
