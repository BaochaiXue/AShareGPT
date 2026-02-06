clean_adj_factors.py
```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ENCODINGS = ("utf-8", "utf-8-sig", "gbk", "gb18030")


@dataclass
class FileStats:
    rows_in: int = 0
    rows_out: int = 0
    duplicate_dates: int = 0
    order_issues: int = 0
    invalid_rows: int = 0
    code_mismatch: int = 0
    changed: bool = False


def normalize_date(value: str) -> str:
    digits = "".join(ch for ch in value if ch.isdigit())
    return digits if len(digits) == 8 else value.strip()


def is_header(row: List[str]) -> bool:
    if len(row) < 2:
        return False
    c0 = row[0].strip().lower()
    c1 = row[1].strip().lower()
    return c0 in {"code"} and c1 in {"date"}


def read_csv_rows(path: Path) -> Iterable[List[str]]:
    for enc in ENCODINGS:
        try:
            with path.open("r", encoding=enc, errors="strict", newline="") as handle:
                return list(csv.reader(handle))
        except UnicodeDecodeError:
            continue
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.reader(handle))


def write_csv_rows(path: Path, rows: Iterable[Tuple[str, str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["code", "date", "adj_factor"])
        for row in rows:
            writer.writerow(row)


def process_file(path: Path, dry_run: bool) -> FileStats:
    stats = FileStats()
    file_code = path.stem.strip()

    rows = read_csv_rows(path)
    if not rows:
        return stats

    data: Dict[str, Tuple[str, str, str]] = {}
    start_idx = 1 if is_header(rows[0]) else 0
    prev_date = None

    for row in rows[start_idx:]:
        if not row:
            continue
        stats.rows_in += 1
        if len(row) < 3:
            stats.invalid_rows += 1
            continue
        code = row[0].strip()
        date_raw = row[1].strip()
        factor = row[2].strip()
        if not date_raw or not factor:
            stats.invalid_rows += 1
            continue

        date = normalize_date(date_raw)
        if prev_date is not None and date < prev_date:
            stats.order_issues += 1
        prev_date = date

        if date in data:
            stats.duplicate_dates += 1
        if code and file_code and code != file_code:
            stats.code_mismatch += 1

        out_code = file_code if file_code else code
        data[date] = (out_code, date, factor)

    if not data:
        return stats

    sorted_dates = sorted(data.keys())
    rows_out = [data[date] for date in sorted_dates]
    stats.rows_out = len(rows_out)

    stats.changed = (
        stats.duplicate_dates > 0
        or stats.order_issues > 0
        or stats.invalid_rows > 0
        or stats.code_mismatch > 0
        or stats.rows_out != stats.rows_in
    )

    if not dry_run and stats.changed:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        write_csv_rows(tmp_path, rows_out)
        tmp_path.replace(path)

    return stats


def iter_csv_files(root: Path, max_files: int) -> Iterable[Path]:
    count = 0
    for path in sorted(root.glob("*.csv")):
        yield path
        count += 1
        if max_files and count >= max_files:
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean adj_factor CSVs: dedupe by date and sort ascending."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "复权因子",
        help="Directory containing per-code adj_factor CSV files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report only, do not modify files.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write per-file cleanup stats to CSV.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on number of files to process.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print progress every N files.",
    )
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        raise SystemExit(f"root not found: {root}")

    totals = FileStats()
    files = 0
    changed_files = 0
    report_writer = None
    report_handle = None

    if args.report:
        report_handle = args.report.open("w", encoding="utf-8", newline="")
        report_writer = csv.writer(report_handle, lineterminator="\n")
        report_writer.writerow(
            [
                "file",
                "rows_in",
                "rows_out",
                "rows_fixed",
                "duplicate_dates",
                "order_issues",
                "invalid_rows",
                "code_mismatch",
                "changed",
            ]
        )

    try:
        for path in iter_csv_files(root, args.max_files):
            files += 1
            stats = process_file(path, args.dry_run)
            totals.rows_in += stats.rows_in
            totals.rows_out += stats.rows_out
            totals.duplicate_dates += stats.duplicate_dates
            totals.order_issues += stats.order_issues
            totals.invalid_rows += stats.invalid_rows
            totals.code_mismatch += stats.code_mismatch
            if stats.changed:
                changed_files += 1

            if report_writer:
                report_writer.writerow(
                    [
                        str(path),
                        stats.rows_in,
                        stats.rows_out,
                        stats.rows_in - stats.rows_out,
                        stats.duplicate_dates,
                        stats.order_issues,
                        stats.invalid_rows,
                        stats.code_mismatch,
                        int(stats.changed),
                    ]
                )

            if args.progress_every and files % args.progress_every == 0:
                print(f"processed {files} files...", flush=True)
                if report_handle:
                    report_handle.flush()
    finally:
        if report_handle:
            report_handle.close()

    print("done")
    print(f"files: {files}")
    print(f"changed_files: {changed_files}")
    print(f"rows_in: {totals.rows_in}")
    print(f"rows_out: {totals.rows_out}")
    print(f"duplicate_dates: {totals.duplicate_dates}")
    print(f"order_issues: {totals.order_issues}")
    print(f"invalid_rows: {totals.invalid_rows}")
    print(f"code_mismatch: {totals.code_mismatch}")
    if args.dry_run:
        print("dry_run: no files modified")


if __name__ == "__main__":
    main()
```

environment.yml
```yaml
name: asharegpt
channels:
  - conda-forge
  - pytorch
  - nvidia
  - defaults
dependencies:
  - python=3.11
  - pip
  - pip:
      - pandas-ta-classic
      - -r requirements.txt
```

genearte_all_code.py
```python
#!/usr/bin/env python3
"""Generate a Markdown snapshot of all Python, YAML, and shell source files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_IGNORE_DIRS: tuple[str, ...] = (
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    ".idea",
    ".vscode",
    ".venv",
    "env",
    "envs",
    "venv",
    "build",
    "dist",
    "node_modules",
    "tests",
)

SUFFIX_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".sh": "bash",
}


def collect_files(root: Path, suffixes: Iterable[str], ignore_dirs: Sequence[str]) -> list[Path]:
    """Return all files under root that match the specified suffixes while respecting the ignore list."""
    suffix_set: set[str] = {suffix.lower() for suffix in suffixes}
    ignore_set: set[str] = set(ignore_dirs)
    matches: list[Path] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Trim directories in-place so os.walk does not descend into ignored folders.
        dirnames[:] = [directory for directory in dirnames if directory not in ignore_set]

        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() not in suffix_set:
                continue
            matches.append(path)

    matches.sort()
    return matches


def write_markdown(files: Iterable[Path], root: Path, output_path: Path) -> None:
    """Write a Markdown file containing each source file surrounded by fenced code blocks."""
    root = root.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for path in files:
            rel_path = path.resolve().relative_to(root)
            language = SUFFIX_TO_LANG.get(path.suffix.lower(), "")
            handle.write(f"{rel_path.as_posix()}\n")
            handle.write(f"```{language}\n")
            text = path.read_text(encoding="utf-8", errors="replace")
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
            handle.write("```\n\n")


def main() -> None:
    """Entry point for generating the Markdown snapshot."""
    parser = argparse.ArgumentParser(
        description="Generate a Markdown file containing all Python, YAML, and shell source code."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Root directory to scan (defaults to current working directory).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("all_code.md"),
        help="Path to the Markdown file to generate (defaults to ./all_code.md).",
    )
    parser.add_argument(
        "--ignore",
        nargs="*",
        default=list(DEFAULT_IGNORE_DIRS),
        metavar="DIR",
        help="Directory names to skip anywhere under the root.",
    )
    args = parser.parse_args()

    files = collect_files(args.root, SUFFIX_TO_LANG.keys(), args.ignore)
    write_markdown(files, args.root, args.output)


if __name__ == "__main__":
    main()
```

model_core/__init__.py
```python
"""AShareGPT core package (A-share minute CSV only)."""
```

model_core/alphagpt.py
```python

from typing import Optional, Iterator
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ModelConfig
from .factors import FeatureEngineer
from .ops import OPS_CONFIG


class NewtonSchulzLowRankDecay:
    """
    Low-Rank Decay (LoRD) using Newton-Schulz iteration.
    
    A more efficient regularization method that targets low-rank structure
    in attention and key parameters. Uses Newton-Schulz iteration to compute
    the minimum singular vectors without explicit SVD.

    This optimizer wrapper identifies specific 2D parameters (like Attention projection matrices),
    computes an orthogonal matrix approximating their singular vectors, and decays the weights
    towards a lower-rank approximation. This encourages the model to learn "cleaner", low-rank
    representations, which is associated with better generalization (Grokking).
    
    Args:
        named_parameters: Iterator yielding (name, parameter) tuples from the model.
        decay_rate: Strength of low-rank decay (lambda).
        num_iterations: Number of Newton-Schulz iterations (default: 5). High values are more accurate but slower.
        target_keywords: list of strings. Only parameters containing these keywords will be decayed.
    """
    def __init__(self, 
                 named_parameters: Iterator[tuple[str, nn.Parameter]], 
                 decay_rate: float = 1e-3, 
                 num_iterations: int = 5, 
                 target_keywords: Optional[list[str]] = None):
        self.decay_rate = decay_rate
        self.num_iterations = num_iterations
        self.target_keywords = target_keywords or ["attention"]
        self.params_to_decay: list[tuple[str, nn.Parameter]] = []
        
        for name, param in named_parameters:
            if not param.requires_grad or param.ndim != 2:
                continue
            if not any(k in name for k in self.target_keywords):
                continue
            self.params_to_decay.append((name, param))
    
    @torch.no_grad()
    def step(self) -> None:
        """Apply Newton-Schulz low-rank decay to target parameters."""
        for _, W in self.params_to_decay:
            orig_dtype = W.dtype
            X = W.float()
            r, c = X.shape
            
            # Transpose if needed for efficiency (we want the smaller dimension to be the rank bottleneck)
            transposed = False
            if r > c:
                X = X.T
                transposed = True
            
            # Normalize by spectral norm to ensure convergence of Newton-Schulz
            norm = X.norm() + 1e-8
            X = X / norm
            
            # Initialize Y for Newton-Schulz iteration
            Y = X
            identity = torch.eye(X.shape[-1], device=X.device, dtype=X.dtype)
            
            # Newton-Schulz iteration: Y_{k+1} = 0.5 * Y_k * (3*I - Y_k^T * Y_k)
            # This converges to an orthogonal matrix sharing the same singular vectors as X.
            # It essentially "whitens" the singular values, pushing them towards 1 or 0.
            for _ in range(self.num_iterations):
                A = Y.T @ Y
                Y = 0.5 * Y @ (3.0 * identity - A)
            
            if transposed:
                Y = Y.T
            
            # Apply low-rank decay: Weights are pushed away from this orthogonal basis
            W.sub_(self.decay_rate * Y.to(orig_dtype))



class StableRankMonitor:
    """Monitor the effective rank (stable rank) of model parameters during training."""
    
    def __init__(self, model: nn.Module, target_keywords: Optional[list[str]] = None):
        self.model = model
        self.target_keywords = target_keywords or ["attention", "in_proj", "out_proj"]
        self.history: list[float] = []
    
    @torch.no_grad()
    def compute(self) -> float:
        """
        Compute average stable rank of target parameters.
        Stable Rank is defined as ||W||_F^2 / ||W||_2^2.
        Lower stable rank indicates the matrix is closer to low-rank.
        """
        ranks = []
        for name, param in self.model.named_parameters():
            if param.ndim != 2:
                continue
            if not any(k in name for k in self.target_keywords):
                continue
            
            W = param.detach().float()
            S = torch.linalg.svdvals(W)
            # Stable Rank = ||W||_F^2 / ||W||_2^2
            stable_rank = (S.norm() ** 2) / (S[0] ** 2 + 1e-9)
            ranks.append(stable_rank.item())
        
        avg_rank = sum(ranks) / len(ranks) if ranks else 0.0
        self.history.append(avg_rank)
        return avg_rank


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization. Stable and efficient."""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class SwiGLU(nn.Module):
    """Swish Gated Linear Unit (SwiGLU) activation function."""
    
    def __init__(self, d_in: int, d_ff: int):
        super().__init__()
        self.w = nn.Linear(d_in, d_ff * 2)
        self.fc = nn.Linear(d_ff, d_in)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split the projection into gate and value
        x_glu = self.w(x)
        x_val, x_gate = x_glu.chunk(2, dim=-1)
        x_act = x_val * F.silu(x_gate)  # SiLU (Swish) activation: x * sigmoid(x)
        return self.fc(x_act)


class MTPHead(nn.Module):
    """
    Multi-Task Pooling Head for multi-objective learning.
    Dynamically routes information to different task heads (e.g. Return, Volatility, Risk)
    and combines them.
    """
    
    def __init__(self, d_model: int, vocab_size: int, num_tasks: int = 3):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for _ in range(num_tasks)
        ])
        # Learnable global task prior, fused with per-sample router probabilities.
        self.task_weights = nn.Parameter(torch.ones(num_tasks) / num_tasks)
        
        # Router network to decide task importance for each token
        self.task_router = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_tasks)
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [Batch, D_Model]
        
        # Route to appropriate task heads
        task_logits = self.task_router(x)
        router_probs = F.softmax(task_logits, dim=-1)  # [Batch, NumTasks]
        task_prior = F.softmax(self.task_weights, dim=0)  # [NumTasks]
        task_probs = router_probs * task_prior.unsqueeze(0)
        task_probs = task_probs / task_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        
        # Compute all task outputs
        # task_outputs: list of [Batch, VocabSize]
        task_outputs_list = [head(x) for head in self.task_heads]
        task_outputs = torch.stack(task_outputs_list, dim=1)  # [Batch, NumTasks, VocabSize]
        
        # Weighted combination of task outputs based on Router's probability
        # [Batch, NumTasks, 1] * [Batch, NumTasks, VocabSize] -> Sum over Tasks
        weighted = (task_probs.unsqueeze(-1) * task_outputs).sum(dim=1) # [Batch, VocabSize]
        return weighted, task_probs



class LoopedTransformerLayer(nn.Module):
    """
    Looped Transformer Layer - recurrent processing within a layer.
    
    Instead of stacking many unique layers (depth), we reuse the same layer multiple times (recurrence).
    This promotes parameter efficiency and algorithmic reasoning (iterative refinement of thought).
    """
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, num_loops: int = 3, dropout: float = 0.1):
        super().__init__()
        self.num_loops = num_loops
        self.d_model = d_model
        self.nhead = nhead

        # Standard attention components
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        
        # RMSNorm instead of LayerNorm (usually more stable for deep/recurrent nets)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        # SwiGLU FFN instead of standard FFN (better performance in LLMs)
        self.ffn = SwiGLU(d_model, dim_feedforward)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, is_causal: bool = False) -> torch.Tensor:
        # Looped processing - recurrent refinement:
        # The same input x runs through this block `num_loops` times.
        # This allows the model to "think harder" without adding new parameters.
        for _ in range(self.num_loops):
            # Self-attention with residual (Pre-Norm architecture)
            x_norm = self.norm1(x)
            attn_out, _ = self.attention(x_norm, x_norm, x_norm, attn_mask=mask, is_causal=is_causal)
            x = x + self.dropout(attn_out)
            
            # FFN with residual
            x_norm = self.norm2(x)
            ffn_out = self.ffn(x_norm)
            x = x + self.dropout(ffn_out)
        
        return x


class LoopedTransformer(nn.Module):
    """
    Looped Transformer Encoder with multiple loop iterations.
    Effectively acts as a very deep network (num_layers * num_loops)
    but with parameter count = num_layers.
    """
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, num_loops: int = 3, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            LoopedTransformerLayer(d_model, nhead, dim_feedforward, num_loops, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, is_causal: bool = False) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask, is_causal=is_causal)
        return x


class AlphaGPT(nn.Module):
    """
    AlphaGPT: Symbolic Formula Generative Model.
    
    A specialized Transformer designed to generate Reverse Polish Notation (RPN) formulas
    for quantitative trading. It treats formula generation as a sequence generation task.
    """
    def __init__(self):
        super().__init__()
        self.d_model = 64
        self.features_list = FeatureEngineer.FEATURES
        self.ops_list = [cfg[0] for cfg in OPS_CONFIG]
        
        self.vocab = self.features_list + self.ops_list
        self.bos_id = len(self.vocab)
        self.vocab_size = len(self.vocab) + 1
        
        # Embedding
        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, ModelConfig.MAX_FORMULA_LEN + 1, self.d_model))
        
        # Enhanced Transformer with Looped Transformer
        # This improves reasoning depth for complex formula logic
        self.blocks = LoopedTransformer(
            d_model=self.d_model,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            num_loops=3,
            dropout=0.1
        )
        
        # RMSNorm instead of LayerNorm
        self.ln_f = RMSNorm(self.d_model)
        
        # MTPHead for multi-task output
        # Predicts not just the next token policy, but also auxiliary tasks if needed
        self.mtp_head = MTPHead(self.d_model, self.vocab_size, num_tasks=3)
        
        # Critic head for RL value estimation (Baseline)
        self.head_critic = nn.Linear(self.d_model, 1)

    def forward(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            idx: Input token indices [Batch, SeqLen]
        Returns:
            logits: Next token probabilities [Batch, SeqLen, Vocab] (Weighted mix)
            value: Value estimate for RL baseline [Batch, SeqLen, 1]
            task_probs: Weights assigned to each task head [Batch, SeqLen, NumTasks]
        """
        _, T = idx.size()
        
        x = self.token_emb(idx) + self.pos_emb[:, :T, :]
        
        # Causal Mask (prevent peeking into future)
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(idx.device)
        
        # Process through looped transformer
        x = self.blocks(x, mask=mask)
        x = self.ln_f(x)
        
        # Use the representation of the *last* token to predict the next token
        last_emb = x[:, -1, :] # [Batch, D_Model]
        
        # Multi-task pooling head for logits
        logits, task_probs = self.mtp_head(last_emb) # logits: [Batch, Vocab], task_probs: [Batch, Tasks]
        value = self.head_critic(last_emb)       # [Batch, 1]
        
        return logits, value, task_probs
```

model_core/application/__init__.py
```python
"""Application-layer orchestration for AShareGPT."""

```

model_core/application/services/__init__.py
```python
"""Training and evaluation services."""

from .ppo_training_service import PpoTrainingService, TrainingRunState
from .reward_orchestrator import FormulaEvaluation, FormulaRewardOrchestrator
from .training_workflow_service import (
    EvaluationSnapshot,
    TrainingWorkflowResult,
    TrainingWorkflowService,
    build_token_tables,
)

__all__ = [
    "build_token_tables",
    "EvaluationSnapshot",
    "FormulaEvaluation",
    "FormulaRewardOrchestrator",
    "PpoTrainingService",
    "TrainingWorkflowResult",
    "TrainingWorkflowService",
    "TrainingRunState",
]
```

model_core/application/services/ppo_training_service.py
```python
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
```

model_core/application/services/reward_orchestrator.py
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from model_core.data_loader import DataSlice, WalkForwardFold


@dataclass
class FormulaEvaluation:
    """Reward and selection metrics for one candidate formula."""

    reward: float
    selection_score: Optional[float]
    mean_return: float
    train_score: Optional[float] = None
    val_score: Optional[float] = None


class FormulaRewardOrchestrator:
    """
    Pure orchestration for formula scoring.

    This service extracts reward logic from the training loop so it can be tested
    without running PPO end-to-end.
    """

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
        self._backtest_engine = backtest_engine
        self._train_slice = train_slice
        self._val_slice = val_slice
        self._walk_forward_folds = walk_forward_folds
        self._use_wfo = use_wfo
        mode = reward_mode.strip().lower()
        if mode not in {"train", "selection"}:
            raise ValueError(f"Unsupported reward_mode={reward_mode!r}; expected 'train' or 'selection'.")
        self._reward_mode = mode
        if self._use_wfo:
            score_split = "train" if self._reward_mode == "train" else "val"
            has_scoring_window = any(
                getattr(fold, score_split).end_idx > getattr(fold, score_split).start_idx
                for fold in self._walk_forward_folds
            )
            if not has_scoring_window:
                raise ValueError(
                    f"Walk-forward requires non-empty {score_split} windows for reward_mode={self._reward_mode!r}. "
                    "Adjust CN_WFO_*_DAYS or disable CN_WALK_FORWARD."
                )

    @staticmethod
    def _score_to_float(score: object) -> float:
        """Accept either tensor-like or numeric score values from backtest engines."""
        if hasattr(score, "item"):
            return float(score.item())  # type: ignore[call-arg]
        return float(score)

    @torch.no_grad()
    def evaluate_formula(self, formula: list[int], full_feat: torch.Tensor) -> FormulaEvaluation:
        res = self._vm.execute(formula, full_feat)
        if res is None:
            return FormulaEvaluation(reward=-5.0, selection_score=None, mean_return=0.0)
        if res.std() < 1e-4:
            return FormulaEvaluation(reward=-2.0, selection_score=None, mean_return=0.0)

        if self._use_wfo:
            return self._evaluate_wfo(res)
        return self._evaluate_train_val(res)

    def _evaluate_wfo(self, res: torch.Tensor) -> FormulaEvaluation:
        fold_scores: list[float] = []
        fold_returns: list[float] = []
        score_split = "train" if self._reward_mode == "train" else "val"
        for fold in self._walk_forward_folds:
            split = getattr(fold, score_split)
            if split.end_idx <= split.start_idx:
                continue
            res_split = res[:, split.start_idx : split.end_idx]
            if res_split.numel() == 0:
                continue
            result = self._backtest_engine.evaluate(
                res_split,
                split.raw_data_cache,
                split.target_ret,
            )
            fold_scores.append(self._score_to_float(result.score))
            fold_returns.append(result.mean_return)

        if not fold_scores:
            return FormulaEvaluation(reward=-2.0, selection_score=None, mean_return=0.0)

        reward = float(sum(fold_scores) / len(fold_scores))
        mean_return = float(sum(fold_returns) / len(fold_returns))
        train_score = reward if self._reward_mode == "train" else None
        val_score = reward if self._reward_mode == "selection" else None
        return FormulaEvaluation(
            reward=reward,
            selection_score=reward,
            mean_return=mean_return,
            train_score=train_score,
            val_score=val_score,
        )

    def _evaluate_train_val(self, res: torch.Tensor) -> FormulaEvaluation:
        res_train = res[:, self._train_slice.start_idx : self._train_slice.end_idx]
        if res_train.numel() == 0 or res_train.std() < 1e-4:
            return FormulaEvaluation(reward=-2.0, selection_score=None, mean_return=0.0)

        train_result = self._backtest_engine.evaluate(
            res_train,
            self._train_slice.raw_data_cache,
            self._train_slice.target_ret,
        )
        train_score = self._score_to_float(train_result.score)
        selection_score = train_score
        mean_return = float(train_result.mean_return)
        val_score: Optional[float] = None

        if self._val_slice and self._val_slice.end_idx > self._val_slice.start_idx:
            res_val = res[:, self._val_slice.start_idx : self._val_slice.end_idx]
            if res_val.numel() > 0:
                val_result = self._backtest_engine.evaluate(
                    res_val,
                    self._val_slice.raw_data_cache,
                    self._val_slice.target_ret,
                )
                val_score = self._score_to_float(val_result.score)
                if self._reward_mode == "selection":
                    selection_score = val_score
                    mean_return = float(val_result.mean_return)

        reward = train_score if self._reward_mode == "train" else selection_score

        return FormulaEvaluation(
            reward=reward,
            selection_score=selection_score,
            mean_return=mean_return,
            train_score=train_score,
            val_score=val_score,
        )
```

model_core/application/services/training_workflow_service.py
```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from model_core.config import ModelConfig
from model_core.data_loader import ChinaMinuteDataLoader, DataSlice
from .ppo_training_service import PpoTrainingService
from .reward_orchestrator import FormulaRewardOrchestrator


def build_token_tables(
    *,
    vocab_size: int,
    feat_offset: int,
    arity_map: dict[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build token arity/delta tables consumed by PPO sampling."""

    token_arity = torch.zeros(vocab_size, dtype=torch.long)
    token_arity[:feat_offset] = 0

    token_delta = torch.ones(vocab_size, dtype=torch.long)
    for token, arity in arity_map.items():
        if 0 <= token < vocab_size:
            arity_int = int(arity)
            token_arity[token] = arity_int
            token_delta[token] = 1 - arity_int
    return token_arity, token_delta


@dataclass
class EvaluationSnapshot:
    """Post-training evaluation line for one dataset window."""

    label: str
    score: float
    mean_return: float
    sharpe: float
    max_drawdown: float


@dataclass
class TrainingWorkflowResult:
    """Structured output of a training run."""

    best_score: float
    best_formula: Optional[list[int]]
    history: dict[str, list[Any]]
    evaluations: list[EvaluationSnapshot]


class TrainingWorkflowService:
    """High-level training workflow extracted from the compatibility engine."""

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
        use_lord: bool,
        lord_opt=None,
        rank_monitor=None,
        train_steps: int,
        batch_size: int,
        max_formula_len: int,
        ppo_epochs: int,
        ppo_clip_eps: float,
        ppo_value_coef: float,
        ppo_entropy_coef: float,
        ppo_max_grad_norm: float,
        rank_every: int = 100,
    ):
        self.loader = loader
        self.model = model
        self.optimizer = optimizer
        self.vm = vm
        self.backtest_engine = backtest_engine
        self.bos_id = int(bos_id)
        self.token_arity = token_arity
        self.token_delta = token_delta
        self.device = device

        self.use_lord = bool(use_lord)
        self.lord_opt = lord_opt
        self.rank_monitor = rank_monitor

        self.train_steps = int(train_steps)
        self.batch_size = int(batch_size)
        self.max_formula_len = int(max_formula_len)
        self.ppo_epochs = int(ppo_epochs)
        self.ppo_clip_eps = float(ppo_clip_eps)
        self.ppo_value_coef = float(ppo_value_coef)
        self.ppo_entropy_coef = float(ppo_entropy_coef)
        self.ppo_max_grad_norm = float(ppo_max_grad_norm)
        self.rank_every = int(rank_every)

        self.splits = self.loader.train_val_test_split()
        self.train_slice = self.splits.get("train")
        self.val_slice = self.splits.get("val")
        self.test_slice = self.splits.get("test")
        self.walk_forward_folds = self.loader.walk_forward_splits() if ModelConfig.CN_WALK_FORWARD else []
        self.use_wfo = ModelConfig.CN_WALK_FORWARD and len(self.walk_forward_folds) > 0

    def run(
        self,
        *,
        strategy_path: str,
        history_path: str = "training_history.json",
        on_new_best: Optional[Callable[[float, float, list[int]], None]] = None,
    ) -> TrainingWorkflowResult:
        full_feat = self.loader.feat_tensor
        if full_feat is None or self.loader.raw_data_cache is None or self.loader.target_ret is None:
            raise ValueError("Data not loaded. Check data loader.")

        train_slice = self.train_slice
        if train_slice is None:
            train_slice = self.loader.get_slice(0, full_feat.shape[-1])

        reward_orchestrator = FormulaRewardOrchestrator(
            vm=self.vm,
            backtest_engine=self.backtest_engine,
            train_slice=train_slice,
            val_slice=self.val_slice,
            walk_forward_folds=self.walk_forward_folds,
            use_wfo=self.use_wfo,
            reward_mode=ModelConfig.CN_REWARD_MODE,
        )
        ppo_service = PpoTrainingService(
            model=self.model,
            optimizer=self.optimizer,
            bos_id=self.bos_id,
            token_arity=self.token_arity,
            token_delta=self.token_delta,
            device=self.device,
            reward_orchestrator=reward_orchestrator,
            use_lord=self.use_lord,
            lord_opt=self.lord_opt,
        )

        run_state = ppo_service.train(
            full_feat=full_feat,
            train_steps=self.train_steps,
            batch_size=self.batch_size,
            max_formula_len=self.max_formula_len,
            ppo_epochs=self.ppo_epochs,
            ppo_clip_eps=self.ppo_clip_eps,
            ppo_value_coef=self.ppo_value_coef,
            ppo_entropy_coef=self.ppo_entropy_coef,
            ppo_max_grad_norm=self.ppo_max_grad_norm,
            rank_monitor=self.rank_monitor if self.use_lord else None,
            rank_every=self.rank_every,
            on_new_best=on_new_best,
        )

        strategy_file = Path(strategy_path)
        if run_state.best_formula is not None:
            with strategy_file.open("w", encoding="utf-8") as handle:
                json.dump(run_state.best_formula, handle)
        elif strategy_file.exists():
            strategy_file.unlink()
        with open(history_path, "w", encoding="utf-8") as handle:
            json.dump(run_state.history, handle)

        evaluations = self._evaluate_best_formula(run_state.best_formula, full_feat)
        return TrainingWorkflowResult(
            best_score=run_state.best_score,
            best_formula=run_state.best_formula,
            history=run_state.history,
            evaluations=evaluations,
        )

    def _evaluate_best_formula(
        self,
        best_formula: Optional[list[int]],
        full_feat: torch.Tensor,
    ) -> list[EvaluationSnapshot]:
        if not best_formula or self.use_wfo:
            return []

        res = self.vm.execute(best_formula, full_feat)
        if res is None:
            return []

        snapshots: list[EvaluationSnapshot] = []
        for label, data_slice in (
            ("Train", self.train_slice),
            ("Val", self.val_slice),
            ("Test", self.test_slice),
        ):
            if data_slice is None:
                continue
            snapshots.append(self._evaluate_slice(label=label, signal=res, data_slice=data_slice))
        return snapshots

    def _evaluate_slice(
        self,
        *,
        label: str,
        signal: torch.Tensor,
        data_slice: DataSlice,
    ) -> EvaluationSnapshot:
        sig_slice = signal[:, data_slice.start_idx : data_slice.end_idx]
        result = self.backtest_engine.evaluate(
            sig_slice,
            data_slice.raw_data_cache,
            data_slice.target_ret,
            return_details=True,
        )
        metrics = result.metrics or {}
        score = float(result.score.item()) if hasattr(result.score, "item") else float(result.score)
        return EvaluationSnapshot(
            label=label,
            score=score,
            mean_return=float(result.mean_return),
            sharpe=float(metrics.get("sharpe", float("nan"))),
            max_drawdown=float(metrics.get("max_drawdown", float("nan"))),
        )

    def train_window_descriptions(self) -> list[str]:
        """Return printable window descriptions for CLI compatibility logs."""

        if self.use_wfo:
            return [f"   Walk-forward validation: {len(self.walk_forward_folds)} folds"]

        lines: list[str] = []
        if self.train_slice is not None:
            lines.append(
                f"   Train window: {self.train_slice.dates.min()} -> {self.train_slice.dates.max()}"
            )
        if self.val_slice is not None:
            lines.append(f"   Val window:   {self.val_slice.dates.min()} -> {self.val_slice.dates.max()}")
        if self.test_slice is not None:
            lines.append(
                f"   Test window:  {self.test_slice.dates.min()} -> {self.test_slice.dates.max()}"
            )
        return lines
```

model_core/application/use_cases/__init__.py
```python
"""Use-case implementations."""

from .backtest_formula import BacktestFormulaUseCase
from .train_alpha import TrainAlphaUseCase

__all__ = ["BacktestFormulaUseCase", "TrainAlphaUseCase"]

```

model_core/application/use_cases/backtest_formula.py
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from model_core.domain.models import Formula
from model_core.ports.interfaces import (
    BacktestEnginePort,
    DataGatewayPort,
    FormulaExecutorPort,
)


@dataclass
class BacktestUseCaseResult:
    """Result envelope for orchestrated backtest calls."""

    ok: bool
    message: str
    payload: Optional[dict] = None


class BacktestFormulaUseCase:
    """Application service that orchestrates load -> execute -> evaluate."""

    def __init__(
        self,
        data_gateway: DataGatewayPort,
        executor: FormulaExecutorPort,
        backtest_engine: BacktestEnginePort,
    ):
        self._data_gateway = data_gateway
        self._executor = executor
        self._backtest_engine = backtest_engine

    def run(
        self,
        *,
        formula: Formula,
        mode: str = "full",
        symbols: Optional[list[str]] = None,
        years: Optional[list[int]] = None,
        start_date: str = "",
        end_date: str = "",
        signal_time: str = "",
        exit_time: str = "",
        limit_codes: int = 50,
        return_details: bool = True,
    ) -> BacktestUseCaseResult:
        if mode not in {"full", "split", "walk_forward"}:
            return BacktestUseCaseResult(ok=False, message=f"Unsupported mode: {mode}")

        self._data_gateway.load(
            codes=symbols,
            years=years,
            start_date=start_date,
            end_date=end_date,
            signal_time=signal_time,
            exit_time=exit_time,
            limit_codes=limit_codes,
        )
        bundle = self._data_gateway.bundle()
        factors = self._executor.execute(formula, bundle.feat_tensor)
        if factors is None:
            return BacktestUseCaseResult(
                ok=False,
                message="Invalid formula - execution failed.",
            )

        warnings: list[str] = []
        if torch.std(factors) < 1e-4:
            warnings.append("Factor has near-zero variance (trivial formula).")

        payload: dict[str, Any] = {
            "mode": mode,
            "symbols": bundle.symbols,
            "dates": bundle.dates,
            "feat_shape": tuple(bundle.feat_tensor.shape),
            "warnings": warnings,
        }

        if mode == "full":
            result = self._backtest_engine.evaluate(
                factors,
                bundle.raw_data_cache,
                bundle.target_ret,
                return_details=return_details,
            )
            payload["result"] = result
            return BacktestUseCaseResult(
                ok=True,
                message="Backtest completed.",
                payload=payload,
            )

        if mode == "split":
            split_results: dict[str, Any] = {}
            splits = self._data_gateway.train_val_test_split()
            for name in ("train", "val", "test"):
                split_slice = splits.get(name)
                if split_slice is None:
                    continue
                if split_slice.end_idx <= split_slice.start_idx:
                    continue
                res_slice = factors[:, split_slice.start_idx : split_slice.end_idx]
                result = self._backtest_engine.evaluate(
                    res_slice,
                    split_slice.raw_data_cache,
                    split_slice.target_ret,
                    return_details=return_details,
                )
                split_results[name] = {
                    "result": result,
                    "dates": split_slice.dates,
                }
            payload["splits"] = split_results
            return BacktestUseCaseResult(
                ok=True,
                message="Split backtest completed.",
                payload=payload,
            )

        fold_results: list[dict[str, Any]] = []
        val_scores: list[float] = []
        test_scores: list[float] = []
        folds = self._data_gateway.walk_forward_splits()
        for idx, fold in enumerate(folds, 1):
            fold_out: dict[str, Any] = {"index": idx}
            if fold.val.end_idx > fold.val.start_idx:
                res_val = factors[:, fold.val.start_idx : fold.val.end_idx]
                val_result = self._backtest_engine.evaluate(
                    res_val,
                    fold.val.raw_data_cache,
                    fold.val.target_ret,
                    return_details=return_details,
                )
                fold_out["val"] = val_result
                val_scores.append(float(val_result.score))
            if fold.test.end_idx > fold.test.start_idx:
                res_test = factors[:, fold.test.start_idx : fold.test.end_idx]
                test_result = self._backtest_engine.evaluate(
                    res_test,
                    fold.test.raw_data_cache,
                    fold.test.target_ret,
                    return_details=return_details,
                )
                fold_out["test"] = test_result
                test_scores.append(float(test_result.score))
            if "val" in fold_out or "test" in fold_out:
                fold_results.append(fold_out)

        if not fold_results:
            payload["folds"] = []
            return BacktestUseCaseResult(
                ok=True,
                message="Walk-forward disabled: not enough data for configured windows.",
                payload=payload,
            )

        payload["folds"] = fold_results
        payload["avg_val_score"] = float(sum(val_scores) / len(val_scores)) if val_scores else None
        payload["avg_test_score"] = float(sum(test_scores) / len(test_scores)) if test_scores else None
        return BacktestUseCaseResult(
            ok=True,
            message="Walk-forward backtest completed.",
            payload=payload,
        )
```

model_core/application/use_cases/train_alpha.py
```python
from __future__ import annotations

from model_core.domain.models import TrainingArtifact
from model_core.ports.interfaces import TrainerPort


class TrainAlphaUseCase:
    """Application service for model training orchestration."""

    def __init__(self, trainer: TrainerPort):
        self._trainer = trainer

    def run(self) -> TrainingArtifact:
        return self._trainer.train()

```

model_core/backtest.py
```python
import math
from dataclasses import dataclass
from typing import Optional

import torch

from .config import ModelConfig


@dataclass
class BacktestResult:
    score: torch.Tensor
    mean_return: float
    metrics: Optional[dict[str, float]] = None
    equity_curve: Optional[torch.Tensor] = None
    portfolio_returns: Optional[torch.Tensor] = None


@dataclass
class TradingPath:
    valid: torch.Tensor
    valid_f: torch.Tensor
    position: torch.Tensor
    turnover: torch.Tensor
    pnl: torch.Tensor


class ChinaBacktest:
    """
    Vectorized backtest for China A-share/ETF.
    Uses open-to-open returns with turnover-based transaction cost,
    signal lag, and optional slippage.
    """

    def __init__(self):
        self.cost_rate = ModelConfig.COST_RATE
        self.slippage_rate = ModelConfig.SLIPPAGE_RATE
        self.slippage_impact = ModelConfig.SLIPPAGE_IMPACT
        self.allow_short = ModelConfig.ALLOW_SHORT
        self.signal_lag = max(0, ModelConfig.SIGNAL_LAG)
        self.annualization_factor = max(1, ModelConfig.ANNUALIZATION_FACTOR)

    def _compute_slippage(
        self,
        turnover: torch.Tensor,
        raw_data: Optional[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        if (self.slippage_rate <= 0) and (self.slippage_impact <= 0):
            return torch.zeros_like(turnover)

        slip = turnover * self.slippage_rate
        if (
            self.slippage_impact > 0
            and raw_data
            and {"high", "low", "open"}.issubset(raw_data.keys())
        ):
            hl_range = (raw_data["high"] - raw_data["low"]).abs() / (raw_data["open"].abs() + 1e-6)
            # Missing OHLC should not poison pnl with NaN slippage.
            hl_range = torch.nan_to_num(hl_range, nan=0.0, posinf=0.0, neginf=0.0)
            slip = slip + turnover * self.slippage_impact * hl_range
        return torch.nan_to_num(slip, nan=0.0, posinf=0.0, neginf=0.0)

    def _compute_risk_metrics(
        self,
        portfolio_ret: torch.Tensor,
        equity_curve: torch.Tensor,
        turnover: torch.Tensor,
        position: torch.Tensor,
    ) -> dict[str, float]:
        eps = 1e-12
        n = portfolio_ret.numel()
        if n == 0:
            return {}

        mean = portfolio_ret.mean()
        std = portfolio_ret.std(unbiased=False)
        ann_factor = float(self.annualization_factor)

        ann_return = torch.pow(torch.clamp(1.0 + mean, min=eps), ann_factor) - 1.0
        ann_vol = std * math.sqrt(ann_factor)
        sharpe = (mean / (std + eps)) * math.sqrt(ann_factor)

        downside = torch.clamp(portfolio_ret, max=0.0)
        down_std = downside.std(unbiased=False)
        sortino = (mean / (down_std + eps)) * math.sqrt(ann_factor)

        equity_end = equity_curve[-1]
        total_return = equity_end - 1.0
        years = n / ann_factor if ann_factor > 0 else 0.0
        if years > 0:
            cagr = torch.pow(torch.clamp(equity_end, min=eps), 1.0 / years) - 1.0
        else:
            cagr = torch.tensor(0.0, device=portfolio_ret.device)

        peak = torch.cummax(equity_curve, dim=0)[0]
        drawdown = equity_curve / peak - 1.0
        max_drawdown = drawdown.min()
        calmar = cagr / (max_drawdown.abs() + eps)

        pos = portfolio_ret[portfolio_ret > 0]
        neg = portfolio_ret[portfolio_ret < 0]
        win_rate = (portfolio_ret > 0).float().mean()
        avg_win = pos.mean() if pos.numel() > 0 else torch.tensor(0.0, device=portfolio_ret.device)
        avg_loss = neg.mean() if neg.numel() > 0 else torch.tensor(0.0, device=portfolio_ret.device)
        profit_factor = pos.sum() / (neg.abs().sum() + eps) if neg.numel() > 0 else torch.tensor(float("inf"))
        expectancy = win_rate * avg_win + (1.0 - win_rate) * avg_loss

        centered = portfolio_ret - mean
        m3 = (centered ** 3).mean()
        m4 = (centered ** 4).mean()
        skew = m3 / (std ** 3 + eps)
        kurtosis = m4 / (std ** 4 + eps) - 3.0

        avg_turnover = turnover.mean()
        gross_exposure = position.abs().mean()
        long_ratio = (position > 0).float().mean()
        short_ratio = (position < 0).float().mean()
        flat_ratio = (position == 0).float().mean()

        def to_float(value: torch.Tensor) -> float:
            return float(value.detach().cpu().item())

        return {
            "total_return": to_float(total_return),
            "cagr": to_float(cagr),
            "annual_return": to_float(ann_return),
            "annual_vol": to_float(ann_vol),
            "sharpe": to_float(sharpe),
            "sortino": to_float(sortino),
            "max_drawdown": to_float(max_drawdown),
            "calmar": to_float(calmar),
            "win_rate": to_float(win_rate),
            "profit_factor": float(to_float(profit_factor)) if torch.isfinite(profit_factor).item() else float("inf"),
            "avg_win": to_float(avg_win),
            "avg_loss": to_float(avg_loss),
            "expectancy": to_float(expectancy),
            "skew": to_float(skew),
            "kurtosis": to_float(kurtosis),
            "avg_turnover": to_float(avg_turnover),
            "gross_exposure": to_float(gross_exposure),
            "long_ratio": to_float(long_ratio),
            "short_ratio": to_float(short_ratio),
            "flat_ratio": to_float(flat_ratio),
        }

    def _build_position(self, signal: torch.Tensor, valid_f: torch.Tensor) -> torch.Tensor:
        if self.allow_short:
            position = torch.sign(signal)
        else:
            position = (signal > 0).float()

        if self.signal_lag > 0:
            position = torch.roll(position, self.signal_lag, dims=1)
            position[:, : self.signal_lag] = 0.0

        # Do not hold positions when target return is missing.
        return position * valid_f

    def _compute_turnover(self, position: torch.Tensor, valid_f: torch.Tensor) -> torch.Tensor:
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0.0
        turnover = torch.abs(position - prev_pos)
        # Charge turnover only on bars with realized return; this still costs
        # re-entry after missing-return gaps (valid -> missing -> valid).
        return turnover * valid_f

    def _compute_pnl(
        self,
        *,
        position: torch.Tensor,
        turnover: torch.Tensor,
        target_ret: torch.Tensor,
        raw_data: dict[str, torch.Tensor],
        valid_f: torch.Tensor,
    ) -> torch.Tensor:
        slippage = self._compute_slippage(turnover, raw_data)
        safe_target = torch.nan_to_num(target_ret, nan=0.0, posinf=0.0, neginf=0.0)
        pnl = position * safe_target - turnover * self.cost_rate - slippage
        return pnl * valid_f

    def _build_trading_path(
        self,
        *,
        factors: torch.Tensor,
        raw_data: dict[str, torch.Tensor],
        target_ret: torch.Tensor,
    ) -> TradingPath:
        valid = torch.isfinite(target_ret)
        valid_f = valid.float()
        signal = torch.tanh(factors)
        position = self._build_position(signal, valid_f)
        turnover = self._compute_turnover(position, valid_f)
        pnl = self._compute_pnl(
            position=position,
            turnover=turnover,
            target_ret=target_ret,
            raw_data=raw_data,
            valid_f=valid_f,
        )
        return TradingPath(valid=valid, valid_f=valid_f, position=position, turnover=turnover, pnl=pnl)

    def _compute_score(self, path: TradingPath) -> tuple[torch.Tensor, float]:
        valid = path.valid
        valid_f = path.valid_f
        pnl = path.pnl
        position = path.position
        turnover = path.turnover

        valid_count = valid_f.sum(dim=1)
        has_obs = valid_count > 0
        valid_count_safe = torch.clamp(valid_count, min=1.0)

        mu = pnl.sum(dim=1) / valid_count_safe
        centered = (pnl - mu.unsqueeze(1)) * valid_f
        std = torch.sqrt((centered ** 2).sum(dim=1) / valid_count_safe + 1e-6)

        neg_mask = (pnl < 0) & valid
        neg_count = neg_mask.sum(dim=1)
        downside = torch.where(neg_mask, pnl, torch.zeros_like(pnl))
        down_mean = downside.sum(dim=1) / torch.clamp(neg_count, min=1)
        down_var = (neg_mask.float() * (downside - down_mean.unsqueeze(1)) ** 2).sum(dim=1)
        down_var = down_var / torch.clamp(neg_count - 1, min=1)
        down_std = torch.sqrt(down_var + 1e-6)

        use_down = neg_count > 5
        sortino = torch.where(use_down, mu / down_std, mu / std) * math.sqrt(self.annualization_factor)
        sortino = torch.where(mu < 0, torch.full_like(sortino, -2.0), sortino)
        sortino = torch.where(turnover.mean(dim=1) > 0.5, sortino - 1.0, sortino)
        sortino = torch.where(position.abs().sum(dim=1) == 0, torch.full_like(sortino, -2.0), sortino)
        sortino = torch.where(has_obs, sortino, torch.full_like(sortino, -2.0))
        sortino = torch.clamp(sortino, -3.0, 5.0)

        final_fitness = torch.median(sortino)
        total_valid = torch.clamp(valid_f.sum(), min=1.0)
        mean_return = (pnl.sum() / total_valid).item()
        return final_fitness, mean_return

    @staticmethod
    def _compute_portfolio_curve(path: TradingPath) -> tuple[torch.Tensor, torch.Tensor]:
        valid_per_t = path.valid_f.sum(dim=0)
        portfolio_ret = torch.where(
            valid_per_t > 0,
            path.pnl.sum(dim=0) / torch.clamp(valid_per_t, min=1.0),
            torch.zeros_like(valid_per_t),
        )
        equity_curve = torch.cumprod(torch.clamp(1.0 + portfolio_ret, min=1e-6), dim=0)
        return portfolio_ret, equity_curve

    def evaluate(
        self,
        factors: torch.Tensor,
        raw_data: dict[str, torch.Tensor],
        target_ret: torch.Tensor,
        *,
        return_details: bool = False,
    ) -> BacktestResult:
        if factors.numel() == 0:
            return BacktestResult(score=torch.tensor(-2.0, device=target_ret.device), mean_return=0.0)

        path = self._build_trading_path(factors=factors, raw_data=raw_data, target_ret=target_ret)
        final_fitness, mean_return = self._compute_score(path)

        if not return_details:
            return BacktestResult(score=final_fitness, mean_return=mean_return)

        portfolio_ret, equity_curve = self._compute_portfolio_curve(path)
        metrics = self._compute_risk_metrics(portfolio_ret, equity_curve, path.turnover, path.position)

        return BacktestResult(
            score=final_fitness,
            mean_return=mean_return,
            metrics=metrics,
            equity_curve=equity_curve.detach().cpu(),
            portfolio_returns=portfolio_ret.detach().cpu(),
        )
```

model_core/bootstrap/__init__.py
```python
"""Composition root utilities."""

from .container import LegacyContainer, create_legacy_container
from .factories import (
    create_training_workflow_service,
    create_training_workflow_service_from_components,
)

__all__ = [
    "LegacyContainer",
    "create_legacy_container",
    "create_training_workflow_service",
    "create_training_workflow_service_from_components",
]
```

model_core/bootstrap/container.py
```python
from __future__ import annotations

from dataclasses import dataclass

from model_core.application.use_cases import BacktestFormulaUseCase, TrainAlphaUseCase
from model_core.infrastructure import (
    ChinaBacktestEngineAdapter,
    ChinaDataGatewayAdapter,
    LegacyAlphaTrainer,
    StackVmFormulaExecutorAdapter,
)
from .factories import create_training_workflow_service


@dataclass
class LegacyContainer:
    """
    Composition root for the compatibility layer.

    This keeps old implementations but exposes them through explicit ports.
    """

    data_gateway: ChinaDataGatewayAdapter
    formula_executor: StackVmFormulaExecutorAdapter
    backtest_engine: ChinaBacktestEngineAdapter
    trainer: LegacyAlphaTrainer

    def backtest_use_case(self) -> BacktestFormulaUseCase:
        return BacktestFormulaUseCase(
            data_gateway=self.data_gateway,
            executor=self.formula_executor,
            backtest_engine=self.backtest_engine,
        )

    def train_use_case(self) -> TrainAlphaUseCase:
        return TrainAlphaUseCase(trainer=self.trainer)


def create_legacy_container(
    *,
    use_lord_regularization: bool = True,
    lord_decay_rate: float = 1e-3,
    lord_num_iterations: int = 5,
) -> LegacyContainer:
    return LegacyContainer(
        data_gateway=ChinaDataGatewayAdapter(),
        formula_executor=StackVmFormulaExecutorAdapter(),
        backtest_engine=ChinaBacktestEngineAdapter(),
        trainer=LegacyAlphaTrainer(
            use_lord_regularization=use_lord_regularization,
            lord_decay_rate=lord_decay_rate,
            lord_num_iterations=lord_num_iterations,
            workflow_factory=create_training_workflow_service,
        ),
    )
```

model_core/bootstrap/factories.py
```python
from __future__ import annotations

from typing import Any, Optional

import torch

from model_core.alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from model_core.application.services import TrainingWorkflowService, build_token_tables
from model_core.backtest import ChinaBacktest
from model_core.config import ModelConfig
from model_core.data_loader import ChinaMinuteDataLoader
from model_core.vm import StackVM


def _default_data_kwargs() -> dict[str, Any]:
    return {
        "codes": ModelConfig.CN_CODES,
        "years": ModelConfig.CN_MINUTE_YEARS,
        "start_date": ModelConfig.CN_MINUTE_START_DATE,
        "end_date": ModelConfig.CN_MINUTE_END_DATE,
        "signal_time": ModelConfig.CN_SIGNAL_TIME,
        "exit_time": ModelConfig.CN_EXIT_TIME,
        "limit_codes": ModelConfig.CN_MAX_CODES,
    }


def create_training_workflow_service_from_components(
    *,
    use_lord_regularization: bool = True,
    lord_decay_rate: float = 1e-3,
    lord_num_iterations: int = 5,
    loader: Optional[ChinaMinuteDataLoader] = None,
    model: Optional[AlphaGPT] = None,
    optimizer=None,
    vm: Optional[StackVM] = None,
    backtest: Optional[ChinaBacktest] = None,
    auto_load_data: bool = True,
    data_kwargs: Optional[dict[str, Any]] = None,
) -> TrainingWorkflowService:
    """Build training workflow from optional runtime components."""

    loader = loader or ChinaMinuteDataLoader()
    if auto_load_data and loader.feat_tensor is None:
        loader.load_data(**(data_kwargs or _default_data_kwargs()))
    if loader.dates is None:
        raise ValueError("Data not loaded. Provide a loaded loader or enable auto_load_data.")

    model = model or AlphaGPT().to(ModelConfig.DEVICE)
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=1e-3)
    vm = vm or StackVM()
    backtest = backtest or ChinaBacktest()
    token_arity, token_delta = build_token_tables(
        vocab_size=model.vocab_size,
        feat_offset=vm.feat_offset,
        arity_map=vm.arity_map,
    )

    use_lord = bool(use_lord_regularization)
    if use_lord:
        lord_opt = NewtonSchulzLowRankDecay(
            model.named_parameters(),
            decay_rate=lord_decay_rate,
            num_iterations=lord_num_iterations,
            target_keywords=["q_proj", "k_proj", "attention"],
        )
        rank_monitor = StableRankMonitor(
            model,
            target_keywords=["attention", "in_proj", "out_proj"],
        )
    else:
        lord_opt = None
        rank_monitor = None

    return TrainingWorkflowService(
        loader=loader,
        model=model,
        optimizer=optimizer,
        vm=vm,
        backtest_engine=backtest,
        bos_id=model.bos_id,
        token_arity=token_arity.to(ModelConfig.DEVICE),
        token_delta=token_delta.to(ModelConfig.DEVICE),
        device=ModelConfig.DEVICE,
        use_lord=use_lord,
        lord_opt=lord_opt,
        rank_monitor=rank_monitor,
        train_steps=ModelConfig.TRAIN_STEPS,
        batch_size=ModelConfig.BATCH_SIZE,
        max_formula_len=ModelConfig.MAX_FORMULA_LEN,
        ppo_epochs=ModelConfig.PPO_EPOCHS,
        ppo_clip_eps=ModelConfig.PPO_CLIP_EPS,
        ppo_value_coef=ModelConfig.PPO_VALUE_COEF,
        ppo_entropy_coef=ModelConfig.PPO_ENTROPY_COEF,
        ppo_max_grad_norm=ModelConfig.PPO_MAX_GRAD_NORM,
        rank_every=100,
    )


def create_training_workflow_service(
    *,
    use_lord_regularization: bool = True,
    lord_decay_rate: float = 1e-3,
    lord_num_iterations: int = 5,
    data_kwargs: Optional[dict[str, Any]] = None,
) -> TrainingWorkflowService:
    """Build the application-level training workflow from the composition root."""

    return create_training_workflow_service_from_components(
        use_lord_regularization=use_lord_regularization,
        lord_decay_rate=lord_decay_rate,
        lord_num_iterations=lord_num_iterations,
        data_kwargs=data_kwargs,
    )
```

model_core/code_alias.py
```python
from __future__ import annotations

import csv
from pathlib import Path


def load_code_alias_map(path: Path) -> dict[str, str]:
    """Load code alias mapping from CSV: old_code,new_code."""
    if not path.exists():
        return {}

    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return {}
        field_map = {name.strip().lower(): name for name in reader.fieldnames}
        old_col = field_map.get("old_code")
        new_col = field_map.get("new_code")
        if not old_col or not new_col:
            return {}

        for row in reader:
            old_code = (row.get(old_col) or "").strip()
            new_code = (row.get(new_col) or "").strip()
            if not old_code or not new_code:
                continue
            if old_code == new_code:
                continue
            mapping[old_code] = new_code
    return mapping
```

model_core/config.py
```python
import os
import torch

class ModelConfig:
    """Configuration for A-share minute backtest/training."""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training defaults (A-share)
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1024"))
    TRAIN_STEPS = int(os.getenv("TRAIN_STEPS", "400"))
    MAX_FORMULA_LEN = int(os.getenv("MAX_FORMULA_LEN", "8"))
    PPO_EPOCHS = int(os.getenv("PPO_EPOCHS", "4"))
    PPO_CLIP_EPS = float(os.getenv("PPO_CLIP_EPS", "0.2"))
    PPO_VALUE_COEF = float(os.getenv("PPO_VALUE_COEF", "0.5"))
    PPO_ENTROPY_COEF = float(os.getenv("PPO_ENTROPY_COEF", "0.01"))
    PPO_MAX_GRAD_NORM = float(os.getenv("PPO_MAX_GRAD_NORM", "1.0"))

    # China market settings
    COST_RATE = float(os.getenv("COST_RATE", "0.0005"))
    SLIPPAGE_RATE = float(os.getenv("SLIPPAGE_RATE", "0.0001"))
    SLIPPAGE_IMPACT = float(os.getenv("SLIPPAGE_IMPACT", "0.0"))
    ALLOW_SHORT = os.getenv("ALLOW_SHORT", "0") == "1"
    SIGNAL_LAG = int(os.getenv("CN_SIGNAL_LAG", "1"))
    ANNUALIZATION_FACTOR = int(os.getenv("ANNUALIZATION_FACTOR", "252"))
    STRATEGY_FILE = os.getenv("STRATEGY_FILE", "best_cn_strategy.json")
    CN_MINUTE_DATA_ROOT = os.getenv("CN_MINUTE_DATA_ROOT", "data")
    CN_USE_ADJ_FACTOR = os.getenv("CN_USE_ADJ_FACTOR", "1") == "1"
    CN_ADJ_FACTOR_DIR = os.getenv("CN_ADJ_FACTOR_DIR", "复权因子")
    CN_CODE_ALIAS_FILE = os.getenv("CN_CODE_ALIAS_FILE", "code_alias_map.csv")
    CN_MINUTE_START_DATE = os.getenv("CN_MINUTE_START_DATE", "")
    CN_MINUTE_END_DATE = os.getenv("CN_MINUTE_END_DATE", "")
    CN_SIGNAL_TIME = os.getenv("CN_SIGNAL_TIME", "10:00")
    CN_EXIT_TIME = os.getenv("CN_EXIT_TIME", "15:00")
    CN_MAX_CODES = int(os.getenv("CN_MAX_CODES", "50"))
    CN_MINUTE_DAYS = int(os.getenv("CN_MINUTE_DAYS", "120"))
    CN_TRAIN_RATIO = float(os.getenv("CN_TRAIN_RATIO", "0.7"))
    CN_VAL_RATIO = float(os.getenv("CN_VAL_RATIO", "0.0"))
    CN_TRAIN_DAYS = int(os.getenv("CN_TRAIN_DAYS", "0"))
    CN_VAL_DAYS = int(os.getenv("CN_VAL_DAYS", "0"))
    CN_TEST_DAYS = int(os.getenv("CN_TEST_DAYS", "0"))
    CN_WALK_FORWARD = os.getenv("CN_WALK_FORWARD", "0") == "1"
    CN_WFO_TRAIN_DAYS = int(os.getenv("CN_WFO_TRAIN_DAYS", "60"))
    CN_WFO_VAL_DAYS = int(os.getenv("CN_WFO_VAL_DAYS", "20"))
    CN_WFO_TEST_DAYS = int(os.getenv("CN_WFO_TEST_DAYS", "20"))
    CN_WFO_STEP_DAYS = int(os.getenv("CN_WFO_STEP_DAYS", "20"))
    CN_FEATURE_NORM = os.getenv("CN_FEATURE_NORM", "train").strip().lower()
    CN_FEATURE_CLIP = float(os.getenv("CN_FEATURE_CLIP", "5.0"))
    CN_REWARD_MODE = os.getenv("CN_REWARD_MODE", "train").strip().lower()
    CN_STRICT_FEATURE_INDICATORS = os.getenv("CN_STRICT_FEATURE_INDICATORS", "1") == "1"
    CN_FEATURE_NEAR_ZERO_STD_TOL = float(os.getenv("CN_FEATURE_NEAR_ZERO_STD_TOL", "1e-6"))
    _CN_CODES_RAW = os.getenv("CN_CODES", "")
    CN_CODES = [c.strip() for c in _CN_CODES_RAW.split(",") if c.strip()]
    _CN_MINUTE_YEARS_RAW = os.getenv("CN_MINUTE_YEARS", "")
    CN_MINUTE_YEARS = [
        int(y.strip()) for y in _CN_MINUTE_YEARS_RAW.split(",") if y.strip().isdigit()
    ]
```

model_core/data_loader.py
```python
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from .config import ModelConfig
from .code_alias import load_code_alias_map
from .factors import FeatureEngineer


@dataclass
class DataSlice:
    feat_tensor: torch.Tensor
    raw_data_cache: dict[str, torch.Tensor]
    target_ret: torch.Tensor
    dates: pd.DatetimeIndex
    symbols: list[str]
    start_idx: int
    end_idx: int


@dataclass
class WalkForwardFold:
    train: DataSlice
    val: DataSlice
    test: DataSlice


class ChinaMinuteDataLoader:
    """
    Minute-level data loader for China A-share/ETF.
    Builds daily decision tensors from minute bars, then holds to exit minute.
    """

    def __init__(self, data_root: Optional[str] = None):
        self.data_root = Path(data_root or ModelConfig.CN_MINUTE_DATA_ROOT)
        self.feat_tensor: Optional[torch.Tensor] = None
        self.raw_data_cache: Optional[dict[str, torch.Tensor]] = None
        self.target_ret: Optional[torch.Tensor] = None
        self.dates: Optional[pd.DatetimeIndex] = None
        self.symbols: Optional[list[str]] = None
        self.feature_norm_info: dict[str, int | str | float] = {}
        alias_path = Path(ModelConfig.CN_CODE_ALIAS_FILE)
        if not alias_path.is_absolute():
            alias_path = self.data_root.parent / alias_path
        self.code_alias_map = load_code_alias_map(alias_path)
        self._warned_missing_adj: set[str] = set()
        self._warned_alias_adj: set[str] = set()

    def _parse_time(self, value: str) -> Optional[time]:
        if not value:
            return None
        try:
            return datetime.strptime(value, "%H:%M").time()
        except ValueError:
            try:
                return datetime.strptime(value, "%H:%M:%S").time()
            except ValueError:
                return None

    def _is_date_only_literal(self, value: str) -> bool:
        """Return True when value is a date-only literal without clock time."""
        value = value.strip()
        if not value:
            return False
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
            try:
                datetime.strptime(value, fmt)
                return True
            except ValueError:
                continue
        return False

    def _resolve_years(self, years: Optional[list[int]]) -> list[int]:
        if years:
            return years
        if ModelConfig.CN_MINUTE_YEARS:
            return ModelConfig.CN_MINUTE_YEARS
        if not self.data_root.exists():
            return []
        available = [int(p.name) for p in self.data_root.iterdir() if p.is_dir() and p.name.isdigit()]
        available.sort()
        return available[-2:] if len(available) >= 2 else available

    def _resolve_codes(self, codes: Optional[list[str]], years: list[int], limit_codes: int) -> list[str]:
        if codes:
            return codes
        if ModelConfig.CN_CODES:
            return ModelConfig.CN_CODES[:limit_codes] if limit_codes else ModelConfig.CN_CODES
        candidates = []
        for year in years:
            year_dir = self.data_root / str(year)
            if not year_dir.exists():
                continue
            candidates.extend([p.stem for p in year_dir.glob("*.csv")])
            if candidates:
                break
        candidates = sorted(set(candidates))
        return candidates[:limit_codes] if limit_codes else candidates

    def _read_adj_factor_csv(self, path: Path) -> pd.DataFrame:
        encodings = ("utf-8", "utf-8-sig", "gbk", "gb18030")
        last_err: Optional[Exception] = None
        for enc in encodings:
            try:
                return pd.read_csv(
                    path,
                    usecols=lambda c: c in {"date", "adj_factor", "code", "证券代码"},
                    dtype={"date": "string"},
                    encoding=enc,
                )
            except Exception as exc:
                last_err = exc
        if last_err:
            raise last_err
        return pd.DataFrame()

    def _load_adj_factors(self, code: str) -> Optional[pd.DataFrame]:
        if not ModelConfig.CN_USE_ADJ_FACTOR:
            return None
        alias_code = self.code_alias_map.get(code)
        candidates = [code]
        if alias_code and alias_code != code:
            candidates.append(alias_code)

        for candidate in candidates:
            path = self.data_root / ModelConfig.CN_ADJ_FACTOR_DIR / f"{candidate}.csv"
            if not path.exists():
                continue
            try:
                df = self._read_adj_factor_csv(path)
            except Exception:
                continue
            if df.empty or "adj_factor" not in df.columns or "date" not in df.columns:
                continue
            df = df.loc[:, ["date", "adj_factor"]].copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            df["date"] = df["date"].dt.normalize()
            df["adj_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce")
            df = df.dropna(subset=["adj_factor"])
            if df.empty:
                continue
            df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date")
            if candidate != code and code not in self._warned_alias_adj:
                print(f"[adj] alias applied: {code} -> {candidate}")
                self._warned_alias_adj.add(code)
            return df

        if code not in self._warned_missing_adj:
            print(f"[adj] missing adj_factor for {code}; fallback to 1.0")
            self._warned_missing_adj.add(code)
        return None

    def _apply_adj_factors(self, code: str, frame: pd.DataFrame) -> pd.DataFrame:
        adj = self._load_adj_factors(code)
        if adj is None or adj.empty:
            frame["adj_factor"] = 1.0
            return frame
        merged = frame.merge(adj, on="date", how="left").sort_values("date")
        merged["adj_factor"] = merged["adj_factor"].ffill().fillna(1.0)
        for col in ("open", "high", "low", "close"):
            merged[col] = merged[col].astype("float64") * merged["adj_factor"]
        return merged

    def _resolve_split_sizes(self, total_len: int) -> tuple[int, int, int]:
        if total_len <= 0:
            return 0, 0, 0
        if ModelConfig.CN_TRAIN_DAYS or ModelConfig.CN_VAL_DAYS or ModelConfig.CN_TEST_DAYS:
            train = max(0, ModelConfig.CN_TRAIN_DAYS)
            val = max(0, ModelConfig.CN_VAL_DAYS)
            test = max(0, ModelConfig.CN_TEST_DAYS)
            if train + val + test == 0:
                train = total_len
            allocated = train + val + test
            if allocated < total_len:
                test += total_len - allocated

            overflow = train + val + test - total_len
            if overflow > 0:
                trim = min(test, overflow)
                test -= trim
                overflow -= trim
            if overflow > 0:
                trim = min(val, overflow)
                val -= trim
                overflow -= trim
            if overflow > 0:
                train = max(0, train - overflow)
            return train, val, test
        train = int(total_len * ModelConfig.CN_TRAIN_RATIO)
        val = int(total_len * ModelConfig.CN_VAL_RATIO)
        if train <= 0:
            train = max(1, total_len - val)
        if train + val > total_len:
            val = max(0, total_len - train)
        test = max(0, total_len - train - val)
        return train, val, test

    def _validate_split_order(
        self,
        dates: pd.DatetimeIndex,
        train_len: int,
        val_len: int,
        test_len: int,
    ) -> None:
        if len(dates) == 0:
            return
        if not dates.is_monotonic_increasing:
            raise ValueError("Date index is not sorted ascending.")
        if train_len + val_len + test_len > len(dates):
            raise ValueError("Split sizes exceed available data length.")

        # Hard check: train < val < test chronological boundary.
        if train_len > 0 and val_len > 0:
            if dates[train_len - 1] >= dates[train_len]:
                raise ValueError("Split order invalid: train end must be earlier than val start.")
        if val_len > 0 and test_len > 0:
            boundary = train_len + val_len
            if dates[boundary - 1] >= dates[boundary]:
                raise ValueError("Split order invalid: val end must be earlier than test start.")
        if val_len == 0 and train_len > 0 and test_len > 0:
            boundary = train_len
            if dates[boundary - 1] >= dates[boundary]:
                raise ValueError("Split order invalid: train end must be earlier than test start.")

    def _validate_target_ret_mask(self, target_df: pd.DataFrame, target_tensor: torch.Tensor) -> None:
        expected_nan = torch.tensor(
            target_df.isna().to_numpy().T,
            dtype=torch.bool,
            device=target_tensor.device,
        )
        actual_nan = torch.isnan(target_tensor)
        if not torch.equal(expected_nan, actual_nan):
            raise ValueError("target_ret NaN mask changed after tensor conversion.")

        # Hard check: missing target entries must not be copied from previous day.
        for col_idx, col in enumerate(target_df.columns):
            series = target_df[col]
            missing_idx = series.index[series.isna()]
            if len(missing_idx) == 0:
                continue
            pos = int(target_df.index.get_loc(missing_idx[0]))
            if pos == 0:
                continue
            prev = series.iloc[pos - 1]
            if pd.isna(prev):
                continue
            cur = target_tensor[col_idx, pos]
            if torch.isfinite(cur):
                if abs(float(cur.item()) - float(prev)) < 1e-12:
                    raise ValueError(f"target_ret forward-fill detected for {col}.")

    def _normalize_features(self, raw_feat: torch.Tensor, train_len: int) -> torch.Tensor:
        mode = ModelConfig.CN_FEATURE_NORM
        clip = ModelConfig.CN_FEATURE_CLIP
        total_len = raw_feat.shape[2]

        if mode == "none":
            self.feature_norm_info = {"mode": "none", "fit_len": 0, "clip": clip}
            return raw_feat
        if mode != "train":
            raise ValueError(f"Unsupported CN_FEATURE_NORM={mode}; expected 'none' or 'train'.")
        if train_len <= 0:
            raise ValueError("CN_FEATURE_NORM=train requires a non-empty train split.")
        if train_len > total_len:
            raise ValueError("Train split exceeds feature timeline length.")

        train_feat = raw_feat[:, :, :train_len]
        norm_stats = FeatureEngineer.fit_robust_stats(train_feat)
        self.feature_norm_info = {"mode": "train", "fit_len": train_len, "clip": clip}
        return FeatureEngineer.apply_robust_norm(raw_feat, norm_stats=norm_stats, clip=clip)

    def _slice_raw_data(self, start: int, end: int) -> dict[str, torch.Tensor]:
        if self.raw_data_cache is None:
            raise ValueError("raw_data_cache is empty. Call load_data() first.")
        return {k: v[:, start:end] for k, v in self.raw_data_cache.items()}

    def get_slice(self, start: int, end: int) -> DataSlice:
        if self.feat_tensor is None or self.target_ret is None or self.dates is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        start = max(0, start)
        end = max(start, min(end, self.feat_tensor.shape[2]))
        return DataSlice(
            feat_tensor=self.feat_tensor[:, :, start:end],
            raw_data_cache=self._slice_raw_data(start, end),
            target_ret=self.target_ret[:, start:end],
            dates=self.dates[start:end],
            symbols=self.symbols or [],
            start_idx=start,
            end_idx=end,
        )

    def train_val_test_split(self) -> dict[str, DataSlice]:
        if self.dates is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        total_len = len(self.dates)
        train_len, val_len, test_len = self._resolve_split_sizes(total_len)
        splits: dict[str, DataSlice] = {}
        cursor = 0
        if train_len > 0:
            splits["train"] = self.get_slice(cursor, cursor + train_len)
            cursor += train_len
        if val_len > 0:
            splits["val"] = self.get_slice(cursor, cursor + val_len)
            cursor += val_len
        if test_len > 0:
            splits["test"] = self.get_slice(cursor, cursor + test_len)
        return splits

    def walk_forward_splits(self) -> list[WalkForwardFold]:
        if self.dates is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        total_len = len(self.dates)
        train_len = max(0, ModelConfig.CN_WFO_TRAIN_DAYS)
        val_len = max(0, ModelConfig.CN_WFO_VAL_DAYS)
        test_len = max(0, ModelConfig.CN_WFO_TEST_DAYS)
        step_len = max(1, ModelConfig.CN_WFO_STEP_DAYS)
        window = train_len + val_len + test_len
        if window <= 0 or total_len < window:
            return []
        folds: list[WalkForwardFold] = []
        start = 0
        while start + window <= total_len:
            train_slice = self.get_slice(start, start + train_len)
            val_slice = self.get_slice(start + train_len, start + train_len + val_len)
            test_slice = self.get_slice(start + train_len + val_len, start + window)
            folds.append(WalkForwardFold(train=train_slice, val=val_slice, test=test_slice))
            start += step_len
        return folds

    def _resolve_time_bounds(
        self,
        start_date: str,
        end_date: str,
    ) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        start_raw = start_date or ModelConfig.CN_MINUTE_START_DATE
        end_raw = end_date or ModelConfig.CN_MINUTE_END_DATE
        start_dt = pd.to_datetime(start_raw) if start_raw else None
        end_dt = pd.to_datetime(end_raw) if end_raw else None
        end_dt_exclusive: Optional[pd.Timestamp] = None
        if end_dt is not None and self._is_date_only_literal(end_raw):
            end_dt_exclusive = end_dt.normalize() + pd.Timedelta(days=1)
        return start_dt, end_dt, end_dt_exclusive

    def _load_daily_records_for_code(
        self,
        *,
        code: str,
        years: list[int],
        start_dt: Optional[pd.Timestamp],
        end_dt: Optional[pd.Timestamp],
        end_dt_exclusive: Optional[pd.Timestamp],
        sig_time: time,
        exit_t: Optional[time],
    ) -> list[dict[str, float | pd.Timestamp]]:
        records: list[dict[str, float | pd.Timestamp]] = []
        for year in years:
            path = self.data_root / str(year) / f"{code}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(
                path,
                usecols=["trade_time", "open", "high", "low", "close", "vol", "amount"],
                dtype={"trade_time": "string"},
            )
            if df.empty:
                continue
            df["trade_time"] = pd.to_datetime(df["trade_time"], errors="coerce")
            df = df.dropna(subset=["trade_time"])
            if start_dt is not None:
                df = df[df["trade_time"] >= start_dt]
            if end_dt_exclusive is not None:
                # Date-only end bounds are inclusive of the whole end day.
                df = df[df["trade_time"] < end_dt_exclusive]
            elif end_dt is not None:
                df = df[df["trade_time"] <= end_dt]
            if df.empty:
                continue
            df["date"] = df["trade_time"].dt.normalize()

            for date, day_frame in df.groupby("date"):
                day_frame = day_frame.sort_values("trade_time")
                time_series = day_frame["trade_time"].dt.time
                entry_candidates = day_frame[time_series >= sig_time]
                entry_row = entry_candidates.iloc[0] if not entry_candidates.empty else day_frame.iloc[0]

                if exit_t:
                    exit_candidates = day_frame[time_series >= exit_t]
                    exit_row = exit_candidates.iloc[0] if not exit_candidates.empty else day_frame.iloc[-1]
                else:
                    exit_row = day_frame.iloc[-1]

                entry_open = float(entry_row["open"])
                exit_close = float(exit_row["close"])
                if entry_open == 0:
                    continue

                records.append(
                    {
                        "date": date,
                        "open": float(entry_row["open"]),
                        "high": float(entry_row["high"]),
                        "low": float(entry_row["low"]),
                        "close": float(entry_row["close"]),
                        "volume": float(entry_row["vol"]),
                        "amount": float(entry_row["amount"]),
                        "target_ret": (exit_close / entry_open) - 1.0,
                    }
                )
        return records

    def _build_per_code_frames(
        self,
        *,
        codes: list[str],
        years: list[int],
        start_dt: Optional[pd.Timestamp],
        end_dt: Optional[pd.Timestamp],
        end_dt_exclusive: Optional[pd.Timestamp],
        sig_time: time,
        exit_t: Optional[time],
    ) -> dict[str, pd.DataFrame]:
        per_code_frames: dict[str, pd.DataFrame] = {}
        for code in codes:
            records = self._load_daily_records_for_code(
                code=code,
                years=years,
                start_dt=start_dt,
                end_dt=end_dt,
                end_dt_exclusive=end_dt_exclusive,
                sig_time=sig_time,
                exit_t=exit_t,
            )
            if not records:
                continue
            frame = pd.DataFrame(records)
            frame = frame.drop_duplicates(subset=["date"], keep="last").sort_values("date")
            if ModelConfig.CN_USE_ADJ_FACTOR:
                frame = self._apply_adj_factors(code, frame)
            per_code_frames[code] = frame
        return per_code_frames

    def _apply_recent_day_cutoff(
        self,
        per_code_frames: dict[str, pd.DataFrame],
        *,
        end_dt: Optional[pd.Timestamp],
    ) -> None:
        if end_dt is not None or not ModelConfig.CN_MINUTE_DAYS:
            return
        cutoff_days = ModelConfig.CN_MINUTE_DAYS
        for code, frame in list(per_code_frames.items()):
            frame = frame.sort_values("date")
            if len(frame) > cutoff_days:
                frame = frame.iloc[-cutoff_days:]
            per_code_frames[code] = frame

    def _build_pivots(self, per_code_frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        def build_pivot(
            field: str,
            *,
            ffill: bool = True,
            fill_value: Optional[float] = 0.0,
        ) -> pd.DataFrame:
            series_list = []
            for code, frame in per_code_frames.items():
                s = frame.set_index("date")[field].rename(code)
                series_list.append(s)
            pivot = pd.concat(series_list, axis=1).sort_index()
            if ffill:
                pivot = pivot.ffill()
            if fill_value is not None:
                pivot = pivot.fillna(fill_value)
            return pivot

        pivot_specs: dict[str, tuple[bool, Optional[float]]] = {
            "open": (True, None),
            "high": (True, None),
            "low": (True, None),
            "close": (True, None),
            "volume": (False, 0.0),
            "amount": (False, 0.0),
            "target_ret": (False, None),
        }
        pivots = {
            field: build_pivot(field, ffill=ffill, fill_value=fill_value)
            for field, (ffill, fill_value) in pivot_specs.items()
        }
        if ModelConfig.CN_USE_ADJ_FACTOR:
            pivots["adj_factor"] = build_pivot("adj_factor", ffill=True, fill_value=1.0)
        return pivots

    @staticmethod
    def _pivot_to_tensor(
        pivot: pd.DataFrame,
        *,
        index: pd.DatetimeIndex,
        columns: pd.Index,
    ) -> torch.Tensor:
        aligned = pivot.reindex(index=index, columns=columns)
        return torch.tensor(aligned.values.T, dtype=torch.float32, device=ModelConfig.DEVICE)

    def _build_tensors_from_pivots(
        self,
        pivots: dict[str, pd.DataFrame],
        *,
        index: pd.DatetimeIndex,
        columns: pd.Index,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        target_tensor = self._pivot_to_tensor(pivots["target_ret"], index=index, columns=columns)
        self._validate_target_ret_mask(pivots["target_ret"], target_tensor)

        raw_data_cache = {
            "open": self._pivot_to_tensor(pivots["open"], index=index, columns=columns),
            "high": self._pivot_to_tensor(pivots["high"], index=index, columns=columns),
            "low": self._pivot_to_tensor(pivots["low"], index=index, columns=columns),
            "close": self._pivot_to_tensor(pivots["close"], index=index, columns=columns),
            "volume": self._pivot_to_tensor(pivots["volume"], index=index, columns=columns),
            "amount": self._pivot_to_tensor(pivots["amount"], index=index, columns=columns),
            "liquidity": self._pivot_to_tensor(pivots["amount"], index=index, columns=columns),
            "fdv": self._pivot_to_tensor(pivots["amount"], index=index, columns=columns),
        }
        if "adj_factor" in pivots:
            raw_data_cache["adj_factor"] = self._pivot_to_tensor(
                pivots["adj_factor"],
                index=index,
                columns=columns,
            )
        return raw_data_cache, target_tensor

    def load_data(
        self,
        codes: Optional[list[str]] = None,
        years: Optional[list[int]] = None,
        start_date: str = "",
        end_date: str = "",
        signal_time: str = "",
        exit_time: str = "",
        limit_codes: int = 50,
    ) -> None:
        years = self._resolve_years(years)
        if not years:
            raise ValueError("No available year folders for minute data.")

        codes = self._resolve_codes(codes, years, limit_codes)
        if not codes:
            raise ValueError("No codes resolved for minute data.")

        sig_time = self._parse_time(signal_time or ModelConfig.CN_SIGNAL_TIME) or time(10, 0)
        exit_t = self._parse_time(exit_time or ModelConfig.CN_EXIT_TIME)
        start_dt, end_dt, end_dt_exclusive = self._resolve_time_bounds(start_date, end_date)
        per_code_frames = self._build_per_code_frames(
            codes=codes,
            years=years,
            start_dt=start_dt,
            end_dt=end_dt,
            end_dt_exclusive=end_dt_exclusive,
            sig_time=sig_time,
            exit_t=exit_t,
        )

        if not per_code_frames:
            raise ValueError("No minute data loaded. Check codes/years/date filters.")

        self._apply_recent_day_cutoff(per_code_frames, end_dt=end_dt)
        pivots = self._build_pivots(per_code_frames)

        index = pivots["close"].index
        columns = pivots["close"].columns
        train_len, val_len, test_len = self._resolve_split_sizes(len(index))
        self._validate_split_order(index, train_len, val_len, test_len)
        self.raw_data_cache, target_tensor = self._build_tensors_from_pivots(
            pivots,
            index=index,
            columns=columns,
        )

        raw_feat = FeatureEngineer.compute_features(
            self.raw_data_cache,
            normalize=False,
            strict_indicator_mapping=ModelConfig.CN_STRICT_FEATURE_INDICATORS,
            near_zero_std_tol=ModelConfig.CN_FEATURE_NEAR_ZERO_STD_TOL,
        )
        self.feat_tensor = self._normalize_features(raw_feat, train_len=train_len)
        self.target_ret = target_tensor
        self.dates = index
        self.symbols = list(columns)

        print(f"CN Minute Data Ready. Shape: {self.feat_tensor.shape}")
        if self.feature_norm_info:
            print(
                "[norm] mode={mode} fit_len={fit_len} clip={clip}".format(
                    mode=self.feature_norm_info.get("mode"),
                    fit_len=self.feature_norm_info.get("fit_len"),
                    clip=self.feature_norm_info.get("clip"),
                )
            )
```

model_core/domain/__init__.py
```python
"""Domain models shared across application and infrastructure layers."""

from .models import (
    BacktestEvaluation,
    DataBundle,
    DatasetSlice,
    Formula,
    TrainingArtifact,
    WalkForwardBundle,
)

__all__ = [
    "BacktestEvaluation",
    "DataBundle",
    "DatasetSlice",
    "Formula",
    "TrainingArtifact",
    "WalkForwardBundle",
]
```

model_core/domain/models.py
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import torch

Formula = list[int]


@dataclass
class DataBundle:
    """Canonical in-memory market data payload."""

    feat_tensor: torch.Tensor
    raw_data_cache: dict[str, torch.Tensor]
    target_ret: torch.Tensor
    dates: pd.DatetimeIndex
    symbols: list[str]


@dataclass
class DatasetSlice:
    """Windowed view over the canonical market payload."""

    feat_tensor: torch.Tensor
    raw_data_cache: dict[str, torch.Tensor]
    target_ret: torch.Tensor
    dates: pd.DatetimeIndex
    symbols: list[str]
    start_idx: int
    end_idx: int


@dataclass
class WalkForwardBundle:
    """One walk-forward fold containing train/val/test windows."""

    train: DatasetSlice
    val: DatasetSlice
    test: DatasetSlice


@dataclass
class TrainingArtifact:
    """Minimal training output used by application use-cases."""

    best_formula: Optional[Formula]
    best_score: float
    strategy_path: Optional[str] = None


@dataclass
class BacktestEvaluation:
    """Serializable backtest result returned by application ports."""

    score: float
    mean_return: float
    metrics: Optional[dict[str, float]] = None
    equity_curve: Optional[list[float]] = None
    portfolio_returns: Optional[list[float]] = None
```

model_core/engine.py
```python
from __future__ import annotations

from typing import Any, Optional
from warnings import warn

from tqdm import tqdm

from .application.services import TrainingWorkflowService
from .bootstrap.factories import create_training_workflow_service_from_components
from .config import ModelConfig


class AlphaEngine:
    """
    Compatibility facade over application-level training workflow.

    Deprecated path: prefer wiring through `TrainAlphaUseCase`.
    """

    def __init__(
        self,
        use_lord_regularization: bool = True,
        lord_decay_rate: float = 1e-3,
        lord_num_iterations: int = 5,
        *,
        workflow: Optional[TrainingWorkflowService] = None,
        loader=None,
        model=None,
        optimizer=None,
        vm=None,
        backtest=None,
        auto_load_data: bool = True,
        data_kwargs: Optional[dict[str, Any]] = None,
    ):
        warn(
            "AlphaEngine is a compatibility wrapper; prefer application/use_cases APIs.",
            DeprecationWarning,
            stacklevel=2,
        )

        if workflow is not None:
            self.workflow = workflow
        else:
            self.workflow = create_training_workflow_service_from_components(
                use_lord_regularization=use_lord_regularization,
                lord_decay_rate=lord_decay_rate,
                lord_num_iterations=lord_num_iterations,
                loader=loader,
                model=model,
                optimizer=optimizer,
                vm=vm,
                backtest=backtest,
                auto_load_data=auto_load_data,
                data_kwargs=data_kwargs,
            )

        # Compatibility attributes exposed for older callers.
        self.loader = self.workflow.loader
        self.model = self.workflow.model
        self.opt = self.workflow.optimizer
        self.vm = self.workflow.vm
        self.bt = self.workflow.backtest_engine
        self.token_arity = self.workflow.token_arity
        self.token_delta = self.workflow.token_delta
        self.lord_opt = self.workflow.lord_opt
        self.rank_monitor = self.workflow.rank_monitor
        self.use_lord = self.workflow.use_lord

        self.best_score: float = -float("inf")
        self.best_formula: Optional[list[int]] = None
        self.training_history: dict[str, list[Any]] = {
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

    def train(self) -> None:
        print(
            "🚀 Starting Alpha Mining with PPO + LoRD..."
            if self.use_lord
            else "🚀 Starting Alpha Mining with PPO..."
        )
        if self.use_lord:
            print("   LoRD Regularization enabled")
            print("   Target keywords: ['q_proj', 'k_proj', 'attention']")
        for line in self.workflow.train_window_descriptions():
            print(line)

        def _on_new_best(score: float, mean_return: float, formula: list[int]) -> None:
            tqdm.write(f"[!] New King: Score {score:.2f} | Ret {mean_return:.2%} | Formula {formula}")

        result = self.workflow.run(
            strategy_path=ModelConfig.STRATEGY_FILE,
            history_path="training_history.json",
            on_new_best=_on_new_best,
        )

        self.best_score = result.best_score
        self.best_formula = result.best_formula
        self.training_history = result.history

        print("\n✓ Training completed!")
        print(f"  Best score: {self.best_score:.4f}")
        print(f"  Best formula: {self.best_formula}")
        for snapshot in result.evaluations:
            print(
                f"  {snapshot.label}: Score {snapshot.score:.4f} | "
                f"MeanRet {snapshot.mean_return:.2%} | Sharpe {snapshot.sharpe:.2f} | "
                f"MaxDD {snapshot.max_drawdown:.2%}"
            )
```

model_core/entrypoints.py
```python
from __future__ import annotations

from model_core.bootstrap.container import LegacyContainer
from model_core.bootstrap import create_legacy_container
from model_core.application.use_cases import BacktestFormulaUseCase, TrainAlphaUseCase


def create_app_container(
    *,
    use_lord_regularization: bool = True,
    lord_decay_rate: float = 1e-3,
    lord_num_iterations: int = 5,
) -> LegacyContainer:
    """Composition-root helper for new use-case based integration."""

    return create_legacy_container(
        use_lord_regularization=use_lord_regularization,
        lord_decay_rate=lord_decay_rate,
        lord_num_iterations=lord_num_iterations,
    )


def create_train_use_case(
    *,
    use_lord_regularization: bool = True,
    lord_decay_rate: float = 1e-3,
    lord_num_iterations: int = 5,
) -> tuple[TrainAlphaUseCase, LegacyContainer]:
    """Create training use-case with wired dependencies."""

    container = create_app_container(
        use_lord_regularization=use_lord_regularization,
        lord_decay_rate=lord_decay_rate,
        lord_num_iterations=lord_num_iterations,
    )
    return container.train_use_case(), container


def create_backtest_use_case() -> tuple[BacktestFormulaUseCase, LegacyContainer]:
    """Create backtest use-case with wired dependencies."""

    container = create_app_container()
    return container.backtest_use_case(), container
```

model_core/factors.py
```python
import torch
import pandas as pd
from typing import Optional

try:
    import pandas_ta_classic as ta
except ImportError:
    try:
        import pandas_ta as ta
        if not hasattr(ta, "Strategy"):
            raise ImportError("pandas_ta package does not provide Strategy API")
    except ImportError as exc:
        raise ImportError(
            "pandas_ta Strategy API is required for feature generation. "
            "Install with `pip install pandas-ta-classic`."
        ) from exc

class FeatureEngineer:
    """Feature engineer for China A-share/ETF data using pandas_ta."""
    
    # 61 Features
    FEATURES = [
        # Price Transform
        'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'AMOUNT',
        'AVGPRICE', 'MEDPRICE', 'TYPPRICE', 'WCLOSE',
        
        # Returns & Volatility
        'RET', 'RET5', 'RET10', 'RET20',
        'LOG_RET',
        'TR', 'ATR14', 'NATR14',
        
        # Momentum
        'RSI14', 'RSI24',
        'MACD', 'MACDh', 'MACDs',
        'BOP', 'CCI14', 'CMO14',
        'KDJ_K', 'KDJ_D', 'KDJ_J',
        'MOM10', 'ROC10', 'PPO', 'PPOh', 'PPOs',
        'TSI', 'UO', 'WILLR',
        
        # Overlap / Trend
        'SMA5', 'SMA10', 'SMA20', 'SMA60',
        'EMA5', 'EMA10', 'EMA20', 'EMA60',
        'TEMA10', 
        'BB_UPPER', 'BB_MID', 'BB_LOWER', 'BB_WIDTH',
        'MIDPOINT', 'MIDPRICE',
        'SAR',
        
        # Volume
        'OBV', 'AD', 'ADOSC', 'CMF', 'MFI14', 
        'V_RET', 'VOL_MA5', 'VOL_MA20'
    ]
    
    INPUT_DIM = len(FEATURES)

    @staticmethod
    def fit_robust_stats(t: torch.Tensor) -> dict[str, torch.Tensor]:
        """Fit robust normalization stats along time axis."""
        median = torch.nanmedian(t, dim=-1, keepdim=True)[0]
        mad = torch.nanmedian(torch.abs(t - median), dim=-1, keepdim=True)[0] + 1e-6
        return {"median": median, "mad": mad}

    @staticmethod
    def apply_robust_norm(
        t: torch.Tensor,
        norm_stats: Optional[dict[str, torch.Tensor]] = None,
        clip: float = 5.0,
    ) -> torch.Tensor:
        """Apply robust z-score using provided stats or self-fitted stats."""
        stats = norm_stats or FeatureEngineer.fit_robust_stats(t)
        median = stats["median"]
        mad = stats["mad"]
        norm = (t - median) / mad
        return torch.clamp(norm, -clip, clip)

    @staticmethod
    def compute_features(
        raw_dict: dict[str, torch.Tensor],
        *,
        normalize: bool = True,
        norm_stats: Optional[dict[str, torch.Tensor]] = None,
        clip: float = 5.0,
        strict_indicator_mapping: bool = True,
        near_zero_std_tol: float = 1e-6,
    ) -> torch.Tensor:
        """
        Compute features using pandas_ta.
        
        Args:
            raw_dict: Dictionary of raw tensors [Batch, Time].
                      Keys: open, high, low, close, volume, amount
        
        Returns:
            features: [Batch, N_Features, Time]
        """
        device = raw_dict['close'].device
        dtype = raw_dict['close'].dtype

        # We must process each asset (column) individually or use pandas_ta machinery?
        # AShareGPT handles batch of assets (N Symbols).
        # pandas_ta is designed for single DataFrame (Time, OHLCV).
        # To be efficient, we iterate over pandas_ta functions, not assets.
        # But pandas_ta functions usually take Series. We can apply them to the whole DataFrame if structure permits,
        # but mostly they expect single Series.
        # Given "Batch" dimension is usually Symbols (e.g. 50), looping 50 times is fast enough.
        
        # Convert raw tensors to numpy for pandas processing.
        # Structure: [Batch, Time] -> [Time, Batch] for DataFrame.
        
        opens = raw_dict['open'].detach().cpu().float().numpy().T
        highs = raw_dict['high'].detach().cpu().float().numpy().T
        lows = raw_dict['low'].detach().cpu().float().numpy().T
        closes = raw_dict['close'].detach().cpu().numpy().T
        volumes = raw_dict['volume'].detach().cpu().float().numpy().T
        amounts = raw_dict['amount'].detach().cpu().float().numpy().T
        
        n_time, n_assets = closes.shape
        n_features = len(FeatureEngineer.FEATURES)
        
        # Pre-allocate output feature array [n_assets, n_features, n_time]
        feat_out = torch.zeros((n_assets, n_features, n_time), dtype=dtype, device=device)
        
        # We prefer to use pandas_ta Strategy for speed if possible, but 
        # looping over assets is safer for correctness with pandas_ta's structure.
        
        CustomStrategy = ta.Strategy(
            name="AlphaGPT Strategy",
            ta=[
                # Returns
                {"kind": "log_return", "cumulative": False},
                {"kind": "percent_return", "length": 1},
                {"kind": "percent_return", "length": 5},
                {"kind": "percent_return", "length": 10},
                {"kind": "percent_return", "length": 20},
                
                # Volatility
                {"kind": "true_range"},
                {"kind": "atr", "length": 14},
                {"kind": "natr", "length": 14},
                
                # Momentum
                {"kind": "rsi", "length": 14},
                {"kind": "rsi", "length": 24},
                {"kind": "macd"},
                {"kind": "bop"},
                {"kind": "cci", "length": 14},
                {"kind": "cmo", "length": 14},
                {"kind": "kdj"},
                {"kind": "mom", "length": 10},
                {"kind": "roc", "length": 10},
                {"kind": "ppo"},
                {"kind": "tsi"},
                {"kind": "uo"},
                {"kind": "willr"},
                
                # Overlap
                {"kind": "sma", "length": 5},
                {"kind": "sma", "length": 10},
                {"kind": "sma", "length": 20},
                {"kind": "sma", "length": 60},
                {"kind": "ema", "length": 5},
                {"kind": "ema", "length": 10},
                {"kind": "ema", "length": 20},
                {"kind": "ema", "length": 60},
                {"kind": "tema", "length": 10},
                {"kind": "bbands", "length": 20},
                {"kind": "midpoint"},
                {"kind": "midprice"},
                {"kind": "psar"},
                
                # Volume
                {"kind": "obv"},
                {"kind": "ad"},
                {"kind": "adosc"},
                {"kind": "cmf"},
            ]
        )
        
        missing_indicator_patterns: set[str] = set()

        # Iterate over each asset
        for i in range(n_assets):
            df = pd.DataFrame({
                'open': opens[:, i],
                'high': highs[:, i],
                'low': lows[:, i],
                'close': closes[:, i],
                'volume': volumes[:, i],
            })
            # Handle zeros in volume to avoid div by zero in some indicators
            df['volume'] = df['volume'].replace(0, 1e-4)
            
            # Run Strategy
            ta_accessor = df.ta
            if hasattr(ta_accessor, "cores"):
                # Avoid multiprocessing path instability across pandas-ta variants.
                ta_accessor.cores = 0
            ta_accessor.strategy(CustomStrategy)
            
            # --- Map implementation output columns to FEATURES list ---
            # pandas_ta auto-names columns like "RSI_14", "MACD_12_26_9", etc.
            # We need to map them rigorously.
            
            # Helper to safely get col
            def get_col(name_patterns):
                # patterns: list of possible names, e.g. ["RSI_14", "RSI"]
                for pat in name_patterns:
                    if pat in df.columns:
                        return df[pat].values
                # Prefix search
                for col in df.columns:
                    for pat in name_patterns:
                        if col.startswith(pat):
                            return df[col].values
                pattern_desc = " | ".join(name_patterns)
                if strict_indicator_mapping:
                    raise ValueError(
                        f"Missing indicator mapping [{pattern_desc}] for asset index {i}. "
                        "Set CN_STRICT_FEATURE_INDICATORS=0 to downgrade to zero-fallback."
                    )
                missing_indicator_patterns.add(pattern_desc)
                return df['close'].values * 0 # Fallback
            
            # 1. Base Prices
            feat_dict = {}
            feat_dict['OPEN'] = df['open'].values
            feat_dict['HIGH'] = df['high'].values
            feat_dict['LOW'] = df['low'].values
            feat_dict['CLOSE'] = df['close'].values
            feat_dict['VOLUME'] = df['volume'].values
            feat_dict['AMOUNT'] = amounts[:, i] # Raw passed through
            
            # 2. Transformed Prices
            feat_dict['AVGPRICE'] = (df['open'].values + df['high'].values + df['low'].values + df['close'].values) / 4.0
            feat_dict['MEDPRICE'] = (df['high'].values + df['low'].values) / 2.0
            feat_dict['TYPPRICE'] = (df['high'].values + df['low'].values + df['close'].values) / 3.0
            feat_dict['WCLOSE'] = (df['high'].values + df['low'].values + 2.0 * df['close'].values) / 4.0
            
            # 3. Returns
            feat_dict['RET'] = get_col(["PCTRET_1"])
            feat_dict['RET5'] = get_col(["PCTRET_5"])
            feat_dict['RET10'] = get_col(["PCTRET_10"])
            feat_dict['RET20'] = get_col(["PCTRET_20"])
            feat_dict['LOG_RET'] = get_col(["LOGRET_1"])
            
            # 4. Volatility
            feat_dict['TR'] = get_col(["TR", "TRUERANGE"])
            feat_dict['ATR14'] = get_col(["ATR_14", "ATRr_14"])
            feat_dict['NATR14'] = get_col(["NATR_14"])
            
            # 5. Momentum
            feat_dict['RSI14'] = get_col(["RSI_14"])
            feat_dict['RSI24'] = get_col(["RSI_24"])
            feat_dict['MACD'] = get_col(["MACD_12_26_9"])
            feat_dict['MACDh'] = get_col(["MACDh_12_26_9"])
            feat_dict['MACDs'] = get_col(["MACDs_12_26_9"])
            feat_dict['BOP'] = get_col(["BOP"])
            feat_dict['CCI14'] = get_col(["CCI_14_0.015"])
            feat_dict['CMO14'] = get_col(["CMO_14"])
            feat_dict['KDJ_K'] = get_col(["K_9_3"])
            feat_dict['KDJ_D'] = get_col(["D_9_3"])
            feat_dict['KDJ_J'] = get_col(["J_9_3"])
            feat_dict['MOM10'] = get_col(["MOM_10"])
            feat_dict['ROC10'] = get_col(["ROC_10"])
            feat_dict['PPO'] = get_col(["PPO_12_26_9"])
            feat_dict['PPOh'] = get_col(["PPOh_12_26_9"])
            feat_dict['PPOs'] = get_col(["PPOs_12_26_9"])
            feat_dict['TSI'] = get_col(["TSI_13_25_13"])
            feat_dict['UO'] = get_col(["UO_7_14_28"])
            feat_dict['WILLR'] = get_col(["WILLR_14"])
            
            # 6. Overlap
            feat_dict['SMA5'] = get_col(["SMA_5"])
            feat_dict['SMA10'] = get_col(["SMA_10"])
            feat_dict['SMA20'] = get_col(["SMA_20"])
            feat_dict['SMA60'] = get_col(["SMA_60"])
            feat_dict['EMA5'] = get_col(["EMA_5"])
            feat_dict['EMA10'] = get_col(["EMA_10"])
            feat_dict['EMA20'] = get_col(["EMA_20"])
            feat_dict['EMA60'] = get_col(["EMA_60"])
            feat_dict['TEMA10'] = get_col(["TEMA_10"])
            
            feat_dict['BB_UPPER'] = get_col(["BBU_5_2.0", "BBU_20_2.0"])
            feat_dict['BB_MID'] = get_col(["BBM_5_2.0", "BBM_20_2.0"])
            feat_dict['BB_LOWER'] = get_col(["BBL_5_2.0", "BBL_20_2.0"])
            feat_dict['BB_WIDTH'] = get_col(["BBB_5_2.0", "BBB_20_2.0"])
            
            feat_dict['MIDPOINT'] = get_col(["MIDPOINT_2"])
            feat_dict['MIDPRICE'] = get_col(["MIDPRICE_2"])
            if "PSARl_0.02_0.2" in df.columns and "PSARs_0.02_0.2" in df.columns:
                sar_series = df["PSARl_0.02_0.2"].combine_first(df["PSARs_0.02_0.2"]).fillna(0.0)
                feat_dict['SAR'] = sar_series.values
            elif "PSARl_0.02_0.2" in df.columns:
                feat_dict['SAR'] = df["PSARl_0.02_0.2"].fillna(0.0).values
            elif "PSARs_0.02_0.2" in df.columns:
                feat_dict['SAR'] = df["PSARs_0.02_0.2"].fillna(0.0).values
            else:
                feat_dict['SAR'] = get_col(["SAR"])
            
            # 7. Volume
            feat_dict['OBV'] = get_col(["OBV"])
            feat_dict['AD'] = get_col(["AD"])
            feat_dict['ADOSC'] = get_col(["ADOSC_3_10"])
            feat_dict['CMF'] = get_col(["CMF_20"])
            typ_price = (df["high"] + df["low"] + df["close"]) / 3.0
            raw_money = typ_price * df["volume"]
            price_delta = typ_price.diff()
            pos_mf = raw_money.where(price_delta > 0, 0.0)
            neg_mf = raw_money.where(price_delta < 0, 0.0).abs()
            pos_sum = pos_mf.rolling(14).sum()
            neg_sum = neg_mf.rolling(14).sum()
            money_ratio = pos_sum / (neg_sum + 1e-6)
            mfi14 = 100.0 - (100.0 / (1.0 + money_ratio))
            feat_dict['MFI14'] = mfi14.fillna(50.0).values
            
            # custom volume features not in pandas_ta strategy explicitly or need custom calc
            # V_RET
            v_curr = df['volume'].values
            v_prev = pd.Series(v_curr).shift(1).bfill().values
            feat_dict['V_RET'] = (v_curr / (v_prev + 1e-6)) - 1.0
            
            # VOL MA
            feat_dict['VOL_MA5'] = pd.Series(df['volume']).rolling(5).mean().fillna(0).values
            feat_dict['VOL_MA20'] = pd.Series(df['volume']).rolling(20).mean().fillna(0).values
            
            # Fill tensor
            for f_idx, name in enumerate(FeatureEngineer.FEATURES):
                val = feat_dict.get(name, None)
                if val is None:
                    # Fallback for missing mapping
                    pass 
                else:
                    # Fill
                    val_np = val.copy() if hasattr(val, "copy") else val
                    feat_out[i, f_idx, :] = torch.as_tensor(val_np, dtype=dtype, device=device)

        # Post-process: NaN handling & lightweight quality checks.
        feat_out = torch.nan_to_num(feat_out, nan=0.0, posinf=0.0, neginf=0.0)

        if missing_indicator_patterns:
            samples = sorted(missing_indicator_patterns)[:8]
            print(
                f"[feature-check] indicator mapping fallback-to-zero count={len(missing_indicator_patterns)} "
                f"samples={samples}"
            )

        if near_zero_std_tol > 0 and feat_out.numel() > 0:
            # [Asset, Feature, Time] -> [Feature, Asset*Time]
            per_feature = feat_out.permute(1, 0, 2).reshape(n_features, -1)
            std = per_feature.std(dim=1, unbiased=False)
            near_zero_mask = std <= near_zero_std_tol
            near_zero_count = int(near_zero_mask.sum().item())
            if near_zero_count > 0:
                near_zero_names = [
                    FeatureEngineer.FEATURES[idx]
                    for idx in torch.nonzero(near_zero_mask, as_tuple=False).flatten().tolist()
                ]
                print(
                    f"[feature-check] near_zero_std={near_zero_count}/{n_features} "
                    f"tol={near_zero_std_tol:g} samples={near_zero_names[:8]}"
                )
        
        if not normalize:
            return feat_out

        # Keep normalization optional so caller can avoid train/val leakage.
        return FeatureEngineer.apply_robust_norm(feat_out, norm_stats=norm_stats, clip=clip)
```

model_core/infrastructure/__init__.py
```python
"""Infrastructure implementations."""

from .adapters import (
    ChinaBacktestEngineAdapter,
    ChinaDataGatewayAdapter,
    LegacyAlphaTrainer,
    StackVmFormulaExecutorAdapter,
)

__all__ = [
    "ChinaBacktestEngineAdapter",
    "ChinaDataGatewayAdapter",
    "LegacyAlphaTrainer",
    "StackVmFormulaExecutorAdapter",
]
```

model_core/infrastructure/adapters/__init__.py
```python
"""Primary infrastructure adapters for application ports."""

from .backtest_engine import ChinaBacktestEngineAdapter
from .data_gateway import ChinaDataGatewayAdapter
from .formula_executor import StackVmFormulaExecutorAdapter
from .trainer import LegacyAlphaTrainer

__all__ = [
    "ChinaBacktestEngineAdapter",
    "ChinaDataGatewayAdapter",
    "LegacyAlphaTrainer",
    "StackVmFormulaExecutorAdapter",
]
```

model_core/infrastructure/adapters/backtest_engine.py
```python
from __future__ import annotations

from typing import Optional

import torch

from model_core.backtest import ChinaBacktest
from model_core.domain.models import BacktestEvaluation


class ChinaBacktestEngineAdapter:
    """Adapter from `ChinaBacktest` to `BacktestEnginePort`."""

    def __init__(self, backtest: Optional[ChinaBacktest] = None):
        self._backtest = backtest or ChinaBacktest()

    @property
    def backtest(self) -> ChinaBacktest:
        return self._backtest

    def evaluate(
        self,
        factors: torch.Tensor,
        raw_data: dict[str, torch.Tensor],
        target_ret: torch.Tensor,
        *,
        return_details: bool = False,
    ) -> BacktestEvaluation:
        raw = self._backtest.evaluate(
            factors,
            raw_data,
            target_ret,
            return_details=return_details,
        )
        return BacktestEvaluation(
            score=float(raw.score.item()),
            mean_return=float(raw.mean_return),
            metrics=dict(raw.metrics or {}),
            equity_curve=raw.equity_curve.tolist() if raw.equity_curve is not None else None,
            portfolio_returns=raw.portfolio_returns.tolist() if raw.portfolio_returns is not None else None,
        )
```

model_core/infrastructure/adapters/data_gateway.py
```python
from __future__ import annotations

from typing import Optional

from model_core.data_loader import (
    ChinaMinuteDataLoader,
    DataSlice as LegacyDataSlice,
    WalkForwardFold as LegacyWalkForwardFold,
)
from model_core.domain.models import DataBundle, DatasetSlice, WalkForwardBundle


class ChinaDataGatewayAdapter:
    """Adapter from `ChinaMinuteDataLoader` to `DataGatewayPort`."""

    def __init__(self, loader: Optional[ChinaMinuteDataLoader] = None):
        self._loader = loader or ChinaMinuteDataLoader()

    @property
    def loader(self) -> ChinaMinuteDataLoader:
        return self._loader

    def load(
        self,
        *,
        codes: Optional[list[str]] = None,
        years: Optional[list[int]] = None,
        start_date: str = "",
        end_date: str = "",
        signal_time: str = "",
        exit_time: str = "",
        limit_codes: int = 50,
    ) -> None:
        self._loader.load_data(
            codes=codes,
            years=years,
            start_date=start_date,
            end_date=end_date,
            signal_time=signal_time,
            exit_time=exit_time,
            limit_codes=limit_codes,
        )

    def bundle(self) -> DataBundle:
        if (
            self._loader.feat_tensor is None
            or self._loader.raw_data_cache is None
            or self._loader.target_ret is None
            or self._loader.dates is None
        ):
            raise ValueError("Data not loaded. Call load() first.")
        return DataBundle(
            feat_tensor=self._loader.feat_tensor,
            raw_data_cache=self._loader.raw_data_cache,
            target_ret=self._loader.target_ret,
            dates=self._loader.dates,
            symbols=self._loader.symbols or [],
        )

    def train_val_test_split(self) -> dict[str, DatasetSlice]:
        return {
            name: self._convert_slice(data_slice)
            for name, data_slice in self._loader.train_val_test_split().items()
        }

    def walk_forward_splits(self) -> list[WalkForwardBundle]:
        return [self._convert_fold(fold) for fold in self._loader.walk_forward_splits()]

    def _convert_slice(self, data_slice: LegacyDataSlice) -> DatasetSlice:
        return DatasetSlice(
            feat_tensor=data_slice.feat_tensor,
            raw_data_cache=data_slice.raw_data_cache,
            target_ret=data_slice.target_ret,
            dates=data_slice.dates,
            symbols=data_slice.symbols,
            start_idx=data_slice.start_idx,
            end_idx=data_slice.end_idx,
        )

    def _convert_fold(self, fold: LegacyWalkForwardFold) -> WalkForwardBundle:
        return WalkForwardBundle(
            train=self._convert_slice(fold.train),
            val=self._convert_slice(fold.val),
            test=self._convert_slice(fold.test),
        )
```

model_core/infrastructure/adapters/formula_executor.py
```python
from __future__ import annotations

from typing import Optional

import torch

from model_core.domain.models import Formula
from model_core.vm import StackVM


class StackVmFormulaExecutorAdapter:
    """Adapter from `StackVM` to `FormulaExecutorPort`."""

    def __init__(self, vm: Optional[StackVM] = None):
        self._vm = vm or StackVM()

    @property
    def vm(self) -> StackVM:
        return self._vm

    def execute(self, formula: Formula, feat_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        return self._vm.execute(formula, feat_tensor)
```

model_core/infrastructure/adapters/trainer.py
```python
from __future__ import annotations

from typing import Callable, Optional

from tqdm import tqdm

from model_core.application.services import TrainingWorkflowService
from model_core.config import ModelConfig
from model_core.domain.models import TrainingArtifact


class LegacyAlphaTrainer:
    """Compatibility trainer adapter backed by the workflow service."""

    def __init__(
        self,
        *,
        use_lord_regularization: bool = True,
        lord_decay_rate: float = 1e-3,
        lord_num_iterations: int = 5,
        workflow_factory: Optional[Callable[..., TrainingWorkflowService]] = None,
    ):
        if workflow_factory is None:
            raise ValueError("workflow_factory is required for LegacyAlphaTrainer.")
        self._workflow_factory = workflow_factory
        self._workflow_kwargs = {
            "use_lord_regularization": use_lord_regularization,
            "lord_decay_rate": lord_decay_rate,
            "lord_num_iterations": lord_num_iterations,
        }
        self._workflow: Optional[TrainingWorkflowService] = None

    @property
    def workflow(self) -> Optional[TrainingWorkflowService]:
        return self._workflow

    def train(self) -> TrainingArtifact:
        self._workflow = self._workflow_factory(**self._workflow_kwargs)

        print(
            "🚀 Starting Alpha Mining with PPO + LoRD..."
            if self._workflow.use_lord
            else "🚀 Starting Alpha Mining with PPO..."
        )
        if self._workflow.use_lord:
            print("   LoRD Regularization enabled")
            print("   Target keywords: ['q_proj', 'k_proj', 'attention']")
        for line in self._workflow.train_window_descriptions():
            print(line)

        def _on_new_best(score: float, mean_return: float, formula: list[int]) -> None:
            tqdm.write(f"[!] New King: Score {score:.2f} | Ret {mean_return:.2%} | Formula {formula}")

        result = self._workflow.run(
            strategy_path=ModelConfig.STRATEGY_FILE,
            history_path="training_history.json",
            on_new_best=_on_new_best,
        )

        print("\n✓ Training completed!")
        print(f"  Best score: {result.best_score:.4f}")
        print(f"  Best formula: {result.best_formula}")
        for snapshot in result.evaluations:
            print(
                f"  {snapshot.label}: Score {snapshot.score:.4f} | "
                f"MeanRet {snapshot.mean_return:.2%} | Sharpe {snapshot.sharpe:.2f} | "
                f"MaxDD {snapshot.max_drawdown:.2%}"
            )

        return TrainingArtifact(
            best_formula=result.best_formula,
            best_score=float(result.best_score),
            strategy_path=ModelConfig.STRATEGY_FILE if result.best_formula is not None else None,
        )
```

model_core/ops.py
```python

from typing import Callable
import torch

@torch.jit.script
def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    if d <= 0:
        return x
    t = x.shape[1]
    if d >= t:
        return torch.zeros_like(x)
    pad = torch.zeros((x.shape[0], d), device=x.device, dtype=x.dtype)
    return torch.cat([pad, x[:, : t - d]], dim=1)

@torch.jit.script
def _ts_delta(x: torch.Tensor, d: int) -> torch.Tensor:
    return x - _ts_delay(x, d)

@torch.jit.script
def _ts_zscore(x: torch.Tensor, d: int) -> torch.Tensor:
    if d <= 1:
        return torch.zeros_like(x)
    pad = torch.zeros((x.shape[0], d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    mean = windows.mean(dim=-1)
    std = windows.std(dim=-1) + 1e-6
    return (x - mean) / std

@torch.jit.script
def _ts_decay_linear(x: torch.Tensor, d: int) -> torch.Tensor:
    if d <= 1:
        return x
    pad = torch.zeros((x.shape[0], d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    w = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)
    w = w / w.sum()
    return (windows * w).sum(dim=-1)

@torch.jit.script
def _ts_std(x: torch.Tensor, d: int) -> torch.Tensor:
    if d <= 1:
        return torch.zeros_like(x)
    pad = torch.zeros((x.shape[0], d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    return windows.std(dim=-1, unbiased=False)

@torch.jit.script
def _ts_rank(x: torch.Tensor, d: int) -> torch.Tensor:
    if d <= 1:
        return torch.zeros_like(x)
    pad = torch.zeros((x.shape[0], d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    last = windows[:, :, -1].unsqueeze(-1)
    less_equal = (windows <= last).to(x.dtype).sum(dim=-1)
    return (less_equal - 1.0) / float(d - 1)

@torch.jit.script
def _safe_div(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Keep denominator away from zero while preserving sign.
    denom = torch.where(y >= 0, y + eps, y - eps)
    return x / denom

OPS_CONFIG: list[tuple[str, Callable[..., torch.Tensor], int]] = [
    ('ADD', lambda x, y: x + y, 2),
    ('SUB', lambda x, y: x - y, 2),
    ('MUL', lambda x, y: x * y, 2),
    ('DIV', _safe_div, 2),
    ('NEG', lambda x: -x, 1),
    ('ABS', torch.abs, 1),
    ('SIGN', torch.sign, 1),
    ('DELTA5', lambda x: _ts_delta(x, 5), 1),
    ('MA20', lambda x: _ts_decay_linear(x, 20), 1),
    ('STD20', lambda x: _ts_std(x, 20), 1),
    ('TS_RANK20', lambda x: _ts_rank(x, 20), 1),
]
```

model_core/ports/__init__.py
```python
"""Port interfaces for dependency inversion."""

from .interfaces import (
    BacktestEnginePort,
    DataGatewayPort,
    FormulaExecutorPort,
    TrainerPort,
)

__all__ = [
    "BacktestEnginePort",
    "DataGatewayPort",
    "FormulaExecutorPort",
    "TrainerPort",
]
```

model_core/ports/interfaces.py
```python
from __future__ import annotations

from typing import Optional, Protocol

import torch

from model_core.domain.models import (
    BacktestEvaluation,
    DataBundle,
    DatasetSlice,
    Formula,
    TrainingArtifact,
    WalkForwardBundle,
)


class DataGatewayPort(Protocol):
    """Data access contract for loading and slicing market data."""

    def load(
        self,
        *,
        codes: Optional[list[str]] = None,
        years: Optional[list[int]] = None,
        start_date: str = "",
        end_date: str = "",
        signal_time: str = "",
        exit_time: str = "",
        limit_codes: int = 50,
    ) -> None:
        ...

    def bundle(self) -> DataBundle:
        ...

    def train_val_test_split(self) -> dict[str, DatasetSlice]:
        ...

    def walk_forward_splits(self) -> list[WalkForwardBundle]:
        ...


class FormulaExecutorPort(Protocol):
    """Formula execution contract."""

    def execute(self, formula: Formula, feat_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        ...


class BacktestEnginePort(Protocol):
    """Backtest evaluation contract."""

    def evaluate(
        self,
        factors: torch.Tensor,
        raw_data: dict[str, torch.Tensor],
        target_ret: torch.Tensor,
        *,
        return_details: bool = False,
    ) -> BacktestEvaluation:
        ...


class TrainerPort(Protocol):
    """Training orchestration contract."""

    def train(self) -> TrainingArtifact:
        ...
```

model_core/vm.py
```python
from typing import Optional, Callable
import torch
from .ops import OPS_CONFIG
from .factors import FeatureEngineer


class StackVM:
    """
    Stack-based Virtual Machine for executing symbolic regression formulas.
    
    Interprets a sequence of tokens (integers) as a Reverse Polish Notation (RPN) expression.
    - Feature Tokens (0 to N-1): Push feature tensor onto stack.
    - Operator Tokens (N to M): Pop arguments from stack, apply function, push result.
    """
    
    def __init__(self):
        # Feature indices are [0, INPUT_DIM - 1]
        self.feat_offset = FeatureEngineer.INPUT_DIM
        
        # Build lookup tables for operators
        # Token ID = Offset + Index in OPS_CONFIG
        self.op_map: dict[int, Callable[..., torch.Tensor]] = {
            i + self.feat_offset: cfg[1] for i, cfg in enumerate(OPS_CONFIG)
        }
        self.arity_map: dict[int, int] = {
            i + self.feat_offset: cfg[2] for i, cfg in enumerate(OPS_CONFIG)
        }

    def execute(self, formula_tokens: list[int], feat_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Execute a formula (token sequence) on the given feature tensor.
        
        Args:
            formula_tokens: list of integer tokens representing the RPN formula.
            feat_tensor: Input features [Batch, Channels, Time].
            
        Returns:
            Result tensor [Batch, Time] (Signal) if successful, else None.
        """
        stack: list[torch.Tensor] = []
        try:
            for token in formula_tokens:
                token = int(token)
                
                # Case 1: Feature Token (Leaf Node)
                if token < self.feat_offset:
                    # Push feature channel onto stack
                    # feat_tensor: [Batch, Channels, Time] -> [Batch, Time]
                    stack.append(feat_tensor[:, token, :])
                    
                # Case 2: Operator Token (Internal Node)
                elif token in self.op_map:
                    arity = self.arity_map[token]
                    
                    # Stack Underflow Check
                    if len(stack) < arity:
                        return None
                    
                    # Pop arguments
                    args = []
                    for _ in range(arity):
                        args.append(stack.pop())
                    # RPN arguments are popped in reverse order (Right operand first)
                    args.reverse()
                    
                    # Apply Operator
                    func = self.op_map[token]
                    res = func(*args)
                    
                    # NaN/Inf Protection (Robustness)
                    if torch.isnan(res).any() or torch.isinf(res).any():
                        res = torch.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)
                        
                    stack.append(res)
                else:
                    # Unknown Token
                    return None
            
            # Valid formula must result in exactly one value on the stack.
            return stack[0] if len(stack) == 1 else None
                
        except Exception:
            return None # Runtime error protection
```

run_cn_backtest.py
```python
#!/usr/bin/env python
"""
A-share Strategy Backtest Script (Minute CSV only)

Usage:
    python run_cn_backtest.py --strategy best_cn_strategy.json
    python run_cn_backtest.py --symbols 000001.SZ,600519.SH
"""

import os
import sys
import json
import argparse
from pathlib import Path

import pandas as pd

from model_core.config import ModelConfig
from model_core.entrypoints import create_backtest_use_case


def load_strategy(filepath: str) -> list:
    with open(filepath, 'r') as f:
        return json.load(f)


def _format_pct(value: float) -> str:
    return f"{value:.2%}"


def print_metrics(title: str, result) -> None:
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)
    print(f"Sortino Score: {result.score:.4f}")
    print(f"Mean Return: {result.mean_return:.4%}")
    if not result.metrics:
        return
    m = result.metrics
    print(f"CAGR: {_format_pct(m['cagr'])} | Annual Vol: {_format_pct(m['annual_vol'])} | Sharpe: {m['sharpe']:.2f}")
    print(f"Max Drawdown: {_format_pct(m['max_drawdown'])} | Calmar: {m['calmar']:.2f} | Win Rate: {_format_pct(m['win_rate'])}")
    print(f"Profit Factor: {m['profit_factor']:.2f} | Expectancy: {_format_pct(m['expectancy'])}")
    print(f"Avg Turnover: {_format_pct(m['avg_turnover'])} | Long: {_format_pct(m['long_ratio'])} | Short: {_format_pct(m['short_ratio'])} | Flat: {_format_pct(m['flat_ratio'])}")


def save_equity_curve(path: str, dates, result) -> None:
    if result.equity_curve is None or result.portfolio_returns is None:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "date": dates.astype("datetime64[ns]"),
            "equity": result.equity_curve,
            "return": result.portfolio_returns,
        }
    )
    df.to_csv(out_path, index=False)
    print(f"📈 Equity curve saved: {out_path}")


def run_backtest(
    formula: list,
    symbols: list | None = None,
    split: bool = False,
    walk_forward: bool = False,
    curve_out: str | None = None,
):
    print("Loading minute CSV data...")
    mode = "walk_forward" if walk_forward else ("split" if split else "full")
    backtest_use_case, _ = create_backtest_use_case()
    use_case_result = backtest_use_case.run(
        formula=formula,
        mode=mode,
        symbols=symbols,
        limit_codes=ModelConfig.CN_MAX_CODES,
        return_details=True,
    )
    if not use_case_result.ok:
        print(f"❌ {use_case_result.message}")
        return None

    payload = use_case_result.payload or {}
    dates = payload.get("dates")
    symbols_loaded = payload.get("symbols") or []
    feat_shape = payload.get("feat_shape")
    warnings = payload.get("warnings") or []

    if dates is None:
        print("❌ Backtest use-case returned incomplete payload")
        return None

    print(f"Data shape: {feat_shape}")
    print(f"Symbols: {len(symbols_loaded) if symbols_loaded else 'N/A'}")
    print(f"Date range: {dates.min()} to {dates.max()}")
    print("\nExecuting strategy formula...")
    for warning in warnings:
        print(f"⚠️  Warning: {warning}")
    print("\nRunning backtest...")

    if mode == "walk_forward":
        folds = payload.get("folds") or []
        if not folds:
            print(f"⚠️  {use_case_result.message}")
            return None
        for fold in folds:
            idx = fold.get("index")
            val_result = fold.get("val")
            test_result = fold.get("test")
            if val_result is not None:
                print_metrics(f"Fold {idx} - Validation", val_result)
            if test_result is not None:
                print_metrics(f"Fold {idx} - Test", test_result)
        avg_val_score = payload.get("avg_val_score")
        avg_test_score = payload.get("avg_test_score")
        if avg_val_score is not None:
            print(f"\nWalk-forward Avg Val Score: {avg_val_score:.4f}")
        if avg_test_score is not None:
            print(f"Walk-forward Avg Test Score: {avg_test_score:.4f}")
        return None

    if mode == "split":
        split_results = payload.get("splits") or {}
        for name in ("train", "val", "test"):
            out = split_results.get(name)
            if not out:
                continue
            result = out.get("result")
            split_dates = out.get("dates")
            if result is None or split_dates is None:
                continue
            print_metrics(f"{name.capitalize()} Results", result)
            if curve_out:
                suffix = f"_{name}"
                out_path = Path(curve_out)
                out_file = out_path.with_name(f"{out_path.stem}{suffix}{out_path.suffix or '.csv'}")
                save_equity_curve(str(out_file), split_dates, result)
        return None

    result = payload.get("result")
    if result is None:
        print("❌ Backtest use-case returned no full-sample result")
        return None

    print("\n" + "=" * 60)
    print("📊 Backtest Results")
    print("=" * 60)
    print_metrics("Full Sample", result)

    if curve_out:
        save_equity_curve(curve_out, dates, result)

    return {
        'score': result.score,
        'mean_return': result.mean_return,
        'avg_turnover': result.metrics["avg_turnover"] if result.metrics else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Run A-share minute backtest")
    parser.add_argument("--strategy", type=str, default=None, help="Path to strategy JSON file")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated list of symbols")
    parser.add_argument("--formula", type=str, default=None, help="Formula as JSON string")
    parser.add_argument("--split", action="store_true", help="Report train/val/test metrics")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward validation")
    parser.add_argument("--curve-out", type=str, default=None, help="Save equity curve CSV")
    args = parser.parse_args()

    print("=" * 60)
    print("🇨🇳 AShareGPT Backtest (Minute CSV)")
    print("=" * 60)

    if args.formula:
        formula = json.loads(args.formula)
    elif args.strategy:
        formula = load_strategy(args.strategy)
    else:
        if os.path.exists(ModelConfig.STRATEGY_FILE):
            formula = load_strategy(ModelConfig.STRATEGY_FILE)
            print(f"Using strategy file: {ModelConfig.STRATEGY_FILE}")
        else:
            print(f"❌ No strategy file found at {ModelConfig.STRATEGY_FILE}")
            print("   Train a model first: python run_cn_train.py")
            sys.exit(1)

    print(f"Formula: {formula}")

    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
        print(f"Testing on symbols: {symbols}")

    try:
        run_backtest(
            formula,
            symbols,
            split=args.split,
            walk_forward=args.walk_forward,
            curve_out=args.curve_out,
        )
    except Exception as e:
        print(f"\n❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
```

run_cn_train.py
```python
#!/usr/bin/env python
"""
A-share Alpha Factor Mining Training Script (Minute CSV only)

Usage:
    python run_cn_train.py

Requirements:
    - Local minute CSVs under ./data/YYYY/<code>.csv
"""

import os
import sys

from model_core.entrypoints import create_train_use_case


def main():
    print("=" * 60)
    print("🇨🇳 AShareGPT Training (Minute CSV)")
    print("=" * 60)
    print(f"Strategy Output: {os.environ.get('STRATEGY_FILE', 'best_cn_strategy.json')}")
    print()

    try:
        train_use_case, _ = create_train_use_case(use_lord_regularization=True)
        artifact = train_use_case.run()

        print("\n" + "=" * 60)
        print("✅ Training Complete!")
        print("=" * 60)

        if artifact.best_formula:
            print(f"Best Score: {artifact.best_score:.4f}")
            print(f"Best Formula: {artifact.best_formula}")
        else:
            print("⚠️  No valid formula found. Try increasing TRAIN_STEPS.")

    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nPossible causes:")
        print("  1. Minute CSVs not found - check ./data/YYYY/")
        print("  2. Date filters exclude all data")
        print("  3. CN_CODES/CN_MINUTE_YEARS not set")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

scripts/backfill_adj_by_alias.py
```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model_core.code_alias import load_code_alias_map  # noqa: E402


ENCODINGS = ("utf-8", "utf-8-sig", "gbk", "gb18030")


def read_adj_csv(path: Path) -> pd.DataFrame:
    last_err: Optional[Exception] = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(
                path,
                usecols=lambda c: c in {"code", "date", "adj_factor", "证券代码"},
                dtype={"date": "string"},
                encoding=enc,
            )
        except Exception as exc:
            last_err = exc
    if last_err is not None:
        raise last_err
    return pd.DataFrame()


def collect_minute_codes(data_root: Path) -> set[str]:
    codes: set[str] = set()
    for year_dir in sorted(data_root.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        for path in year_dir.glob("*.csv"):
            codes.add(path.stem)
    return codes


def normalize_adj(df: pd.DataFrame, old_code: str) -> pd.DataFrame:
    if df.empty:
        return df
    if "code" not in df.columns and "证券代码" in df.columns:
        df = df.rename(columns={"证券代码": "code"})
    for col in ("date", "adj_factor"):
        if col not in df.columns:
            return pd.DataFrame()

    out = df.loc[:, ["date", "adj_factor"]].copy()
    out["date"] = pd.to_numeric(out["date"], errors="coerce").astype("Int64").astype("string")
    out["adj_factor"] = pd.to_numeric(out["adj_factor"], errors="coerce")
    out = out.dropna(subset=["date", "adj_factor"])
    out = out.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    if out.empty:
        return out
    out.insert(0, "code", old_code)
    return out


def write_adj(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill missing old-code adj files from mapped new-code adj files."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Data root containing year folders and adj factor folder.",
    )
    parser.add_argument(
        "--alias-file",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "code_alias_map.csv",
        help="CSV with old_code,new_code columns.",
    )
    parser.add_argument(
        "--adj-dir",
        type=str,
        default="复权因子",
        help="Adj factor directory name under data root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing old-code adj files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan only, do not write files.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional CSV report output path.",
    )
    args = parser.parse_args()

    data_root = args.data_root
    adj_root = data_root / args.adj_dir
    alias_file = args.alias_file

    alias_map = load_code_alias_map(alias_file)
    if not alias_map:
        raise SystemExit(f"No valid alias mapping found: {alias_file}")

    minute_codes = collect_minute_codes(data_root)
    stats = {
        "created": 0,
        "overwritten": 0,
        "skipped_exists": 0,
        "skipped_no_minute": 0,
        "missing_source": 0,
        "invalid_source": 0,
    }

    report_handle = None
    report_writer = None
    if args.report is not None:
        report_handle = args.report.open("w", encoding="utf-8", newline="")
        report_writer = csv.writer(report_handle, lineterminator="\n")
        report_writer.writerow(["old_code", "new_code", "status", "rows"])

    try:
        for old_code, new_code in sorted(alias_map.items()):
            status = ""
            rows = 0
            dst = adj_root / f"{old_code}.csv"
            src = adj_root / f"{new_code}.csv"
            dst_exists_before = dst.exists()

            if old_code not in minute_codes:
                status = "skipped_no_minute"
                stats[status] += 1
            elif dst_exists_before and not args.overwrite:
                status = "skipped_exists"
                stats[status] += 1
            elif not src.exists():
                status = "missing_source"
                stats[status] += 1
            else:
                try:
                    df = read_adj_csv(src)
                    out = normalize_adj(df, old_code)
                except Exception:
                    out = pd.DataFrame()
                if out.empty:
                    status = "invalid_source"
                    stats[status] += 1
                else:
                    rows = int(len(out))
                    if not args.dry_run:
                        write_adj(dst, out)
                    status = "overwritten" if dst_exists_before else "created"
                    stats[status] += 1

            print(f"{old_code} <- {new_code}: {status} rows={rows}")
            if report_writer is not None:
                report_writer.writerow([old_code, new_code, status, rows])
    finally:
        if report_handle is not None:
            report_handle.close()

    print("done")
    for key in ("created", "overwritten", "skipped_exists", "skipped_no_minute", "missing_source", "invalid_source"):
        print(f"{key}: {stats[key]}")
    if args.dry_run:
        print("dry_run: no files modified")


if __name__ == "__main__":
    main()
```

unify_data.py
```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def read_last_line(path: Path) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if size == 0:
            return ""
        block = 4096
        offset = min(size, block)
        while True:
            f.seek(size - offset)
            chunk = f.read(offset)
            if b"\n" in chunk or size == offset:
                lines = chunk.splitlines()
                return lines[-1].decode("utf-8", errors="ignore") if lines else ""
            offset = min(size, offset * 2)


def get_last_token(path: Path, expect_time: bool, time_col: str) -> Optional[str]:
    if not path.exists():
        return None
    line = read_last_line(path)
    if not line:
        return None
    col_idx = 0
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        header = next(csv.reader(handle), None)
        if header and time_col in header:
            col_idx = header.index(time_col)
    row = next(csv.reader([line]), [])
    if col_idx >= len(row):
        return None
    token = row[col_idx].strip()
    if expect_time:
        if len(token) >= 10 and token[:4].isdigit():
            return token
        return None
    if token.isdigit() and len(token) >= 8:
        return token
    return None


def read_csv_fallback(path: Path, *, dtype: dict, usecols) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(path, dtype=dtype, usecols=usecols, encoding=enc)
        except Exception as exc:
            last_err = exc
    raise last_err if last_err else RuntimeError(f"read_csv failed for {path}")


def append_group(
    output_path: Path,
    group: pd.DataFrame,
    time_col: str,
    cols: list[str],
    expect_time: bool,
    dry_run: bool,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    group = group.loc[:, cols].copy()
    group = group.dropna(subset=[time_col])
    group = group.drop_duplicates(subset=[time_col], keep="last")

    last_token = get_last_token(output_path, expect_time=expect_time, time_col=time_col)
    if last_token:
        group = group[group[time_col] > last_token]

    if group.empty:
        return 0

    group = group.sort_values(time_col)
    header = not output_path.exists()

    if not dry_run:
        group.to_csv(output_path, mode="a", header=header, index=False)

    return len(group)


def iter_csv_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.csv"))


def process_minute_data(minute_root: Path, data_root: Path, dry_run: bool, max_files: int) -> None:
    files = list(iter_csv_files(minute_root))
    if max_files:
        files = files[:max_files]

    total_files = 0
    total_rows = 0

    for path in files:
        total_files += 1
        rel = path.relative_to(minute_root)
        year = rel.parts[0] if rel.parts else path.stem.split("-")[0]

        df = read_csv_fallback(
            path,
            dtype={"证券代码": "string", "code": "string", "trade_time": "string"},
            usecols=lambda c: c in {"证券代码", "code", "trade_time", "open", "high", "low", "close", "vol", "amount"},
        )
        if "code" not in df.columns and "证券代码" not in df.columns:
            print(f"[minute] missing code column: {path}")
            continue
        if "code" not in df.columns:
            df = df.rename(columns={"证券代码": "code"})

        for code, group in df.groupby("code", sort=False):
            output_path = data_root / year / f"{code}.csv"
            rows = append_group(
                output_path,
                group,
                time_col="trade_time",
                cols=["trade_time", "open", "high", "low", "close", "vol", "amount"],
                expect_time=True,
                dry_run=dry_run,
            )
            total_rows += rows

        print(f"[minute] {path} -> year {year}")

    print(f"[minute] files={total_files} rows_appended={total_rows}")


def process_adj_factors(adj_root: Path, data_root: Path, dry_run: bool, max_files: int) -> None:
    files = list(iter_csv_files(adj_root))
    if max_files:
        files = files[:max_files]

    total_files = 0
    total_rows = 0

    for path in files:
        total_files += 1

        df = read_csv_fallback(
            path,
            dtype={"证券代码": "string", "code": "string", "date": "string"},
            usecols=lambda c: c in {"证券代码", "code", "date", "adj_factor"},
        )
        if "code" not in df.columns and "证券代码" not in df.columns:
            print(f"[adj] missing code column: {path}")
            continue
        if "code" not in df.columns:
            df = df.rename(columns={"证券代码": "code"})
        else:
            if "证券代码" in df.columns:
                df["code"] = df["code"].fillna(df["证券代码"])

        for code, group in df.groupby("code", sort=False):
            output_path = data_root / "复权因子" / f"{code}.csv"
            rows = append_group(
                output_path,
                group,
                time_col="date",
                cols=["code", "date", "adj_factor"],
                expect_time=False,
                dry_run=dry_run,
            )
            total_rows += rows

        print(f"[adj] {path}")

    print(f"[adj] files={total_files} rows_appended={total_rows}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unify raw downloads into standard per-code files.")
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parent / "data"))
    parser.add_argument("--raw-root", default=str(Path(__file__).resolve().parent / "data" / "raw_downloads"))
    parser.add_argument("--mode", choices=["all", "minute", "adj"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-files", type=int, default=0)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    raw_root = Path(args.raw_root)

    minute_root = raw_root / "a股1分钟"
    adj_root = raw_root / "复权因子"

    if args.mode in ("all", "minute"):
        process_minute_data(minute_root, data_root, args.dry_run, args.max_files)

    if args.mode in ("all", "adj"):
        process_adj_factors(adj_root, data_root, args.dry_run, args.max_files)


if __name__ == "__main__":
    main()
```

