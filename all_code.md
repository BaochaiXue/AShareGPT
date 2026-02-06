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
        self.target_keywords = target_keywords or ["qk_norm", "attention"]
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
        for name, W in self.params_to_decay:
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
            I = torch.eye(X.shape[-1], device=X.device, dtype=X.dtype)
            
            # Newton-Schulz iteration: Y_{k+1} = 0.5 * Y_k * (3*I - Y_k^T * Y_k)
            # This converges to an orthogonal matrix sharing the same singular vectors as X.
            # It essentially "whitens" the singular values, pushing them towards 1 or 0.
            for _ in range(self.num_iterations):
                A = Y.T @ Y
                Y = 0.5 * Y @ (3.0 * I - A)
            
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


class QKNorm(nn.Module):
    """
    Query-Key Normalization for Attention.
    Propounded to stabilize training of large Transformers by normalizing Q and K
    before the dot product. This prevents attention scores from growing too large.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1, 1, 1, d_model) * (d_model ** -0.5))
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Normalize Q and K independently along the head dimension
        # q, k: [Batch, Head, SeqLen, HeadDim]
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)
        return q_norm * self.scale, k_norm * self.scale


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
        # Unused parameter? Kept for legacy compatibility.
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
        task_probs = F.softmax(task_logits, dim=-1) # [Batch, NumTasks]
        
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
        
        # QK-Norm attention to stabilize training with high recurrence
        self.qk_norm = QKNorm(d_model // nhead)
        
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
            # Note: We should ideally plug QKNorm here if not using torch's native MHA, 
            # but torch's MHA encapsulates dot product. QKNorm requires custom attention impl.
            # Assuming MHA here for simplicity, but in full implementation we'd unpack MHA.
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
        B, T = idx.size()
        
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
            slip = slip + turnover * self.slippage_impact * hl_range
        return slip

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

        signal = torch.tanh(factors)

        if self.allow_short:
            position = torch.sign(signal)
        else:
            position = (signal > 0).float()

        if self.signal_lag > 0:
            position = torch.roll(position, self.signal_lag, dims=1)
            position[:, : self.signal_lag] = 0.0

        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0.0
        turnover = torch.abs(position - prev_pos)

        slippage = self._compute_slippage(turnover, raw_data)
        pnl = position * target_ret - turnover * self.cost_rate - slippage

        mu = pnl.mean(dim=1)
        std = pnl.std(dim=1) + 1e-6

        neg_mask = pnl < 0
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

        sortino = torch.clamp(sortino, -3.0, 5.0)
        final_fitness = torch.median(sortino)
        mean_return = pnl.mean(dim=1).mean().item()

        if not return_details:
            return BacktestResult(score=final_fitness, mean_return=mean_return)

        portfolio_ret = pnl.mean(dim=0)
        equity_curve = torch.cumprod(torch.clamp(1.0 + portfolio_ret, min=1e-6), dim=0)
        metrics = self._compute_risk_metrics(portfolio_ret, equity_curve, turnover, position)

        return BacktestResult(
            score=final_fitness,
            mean_return=mean_return,
            metrics=metrics,
            equity_curve=equity_curve.detach().cpu(),
            portfolio_returns=portfolio_ret.detach().cpu(),
        )
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
    INPUT_DIM = 58  # Updated for pandas_ta features
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
    CN_USE_MINUTE = os.getenv("CN_USE_MINUTE", "1") == "1"
    CN_MINUTE_DATA_ROOT = os.getenv("CN_MINUTE_DATA_ROOT", "data")
    CN_USE_ADJ_FACTOR = os.getenv("CN_USE_ADJ_FACTOR", "1") == "1"
    CN_ADJ_FACTOR_DIR = os.getenv("CN_ADJ_FACTOR_DIR", "复权因子")
    CN_MINUTE_START_DATE = os.getenv("CN_MINUTE_START_DATE", "")
    CN_MINUTE_END_DATE = os.getenv("CN_MINUTE_END_DATE", "")
    CN_SIGNAL_TIME = os.getenv("CN_SIGNAL_TIME", "10:00")
    CN_EXIT_TIME = os.getenv("CN_EXIT_TIME", "15:00")
    CN_MAX_CODES = int(os.getenv("CN_MAX_CODES", "50"))
    CN_MINUTE_DAYS = int(os.getenv("CN_MINUTE_DAYS", "7"))
    CN_TRAIN_RATIO = float(os.getenv("CN_TRAIN_RATIO", "0.7"))
    CN_VAL_RATIO = float(os.getenv("CN_VAL_RATIO", "0.15"))
    CN_TEST_RATIO = float(os.getenv("CN_TEST_RATIO", "0.15"))
    CN_TRAIN_DAYS = int(os.getenv("CN_TRAIN_DAYS", "0"))
    CN_VAL_DAYS = int(os.getenv("CN_VAL_DAYS", "0"))
    CN_TEST_DAYS = int(os.getenv("CN_TEST_DAYS", "0"))
    CN_WALK_FORWARD = os.getenv("CN_WALK_FORWARD", "0") == "1"
    CN_WFO_TRAIN_DAYS = int(os.getenv("CN_WFO_TRAIN_DAYS", "60"))
    CN_WFO_VAL_DAYS = int(os.getenv("CN_WFO_VAL_DAYS", "20"))
    CN_WFO_TEST_DAYS = int(os.getenv("CN_WFO_TEST_DAYS", "20"))
    CN_WFO_STEP_DAYS = int(os.getenv("CN_WFO_STEP_DAYS", "20"))
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
        path = self.data_root / ModelConfig.CN_ADJ_FACTOR_DIR / f"{code}.csv"
        if not path.exists():
            return None
        try:
            df = self._read_adj_factor_csv(path)
        except Exception:
            return None
        if df.empty or "adj_factor" not in df.columns or "date" not in df.columns:
            return None
        df = df.loc[:, ["date", "adj_factor"]].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["date"] = df["date"].dt.normalize()
        df["adj_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce")
        df = df.dropna(subset=["adj_factor"])
        if df.empty:
            return None
        df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        return df

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
            if train + val + test < total_len:
                test += total_len - (train + val + test)
            if train + val + test > total_len:
                overflow = train + val + test - total_len
                test = max(0, test - overflow)
            if train + val + test > total_len:
                overflow = train + val + test - total_len
                val = max(0, val - overflow)
            if train + val + test > total_len:
                overflow = train + val + test - total_len
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

        start_dt = pd.to_datetime(start_date or ModelConfig.CN_MINUTE_START_DATE) if (start_date or ModelConfig.CN_MINUTE_START_DATE) else None
        end_dt = pd.to_datetime(end_date or ModelConfig.CN_MINUTE_END_DATE) if (end_date or ModelConfig.CN_MINUTE_END_DATE) else None

        per_code_frames: dict[str, pd.DataFrame] = {}

        for code in codes:
            records = []
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
                if end_dt is not None:
                    df = df[df["trade_time"] <= end_dt]
                if df.empty:
                    continue
                df["date"] = df["trade_time"].dt.normalize()

                for date, g in df.groupby("date"):
                    g = g.sort_values("trade_time")
                    time_series = g["trade_time"].dt.time
                    entry_candidates = g[time_series >= sig_time]
                    entry_row = entry_candidates.iloc[0] if not entry_candidates.empty else g.iloc[0]

                    if exit_t:
                        exit_candidates = g[time_series >= exit_t]
                        exit_row = exit_candidates.iloc[0] if not exit_candidates.empty else g.iloc[-1]
                    else:
                        exit_row = g.iloc[-1]

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

            if records:
                frame = pd.DataFrame(records)
                frame = frame.drop_duplicates(subset=["date"], keep="last").sort_values("date")
                if ModelConfig.CN_USE_ADJ_FACTOR:
                    frame = self._apply_adj_factors(code, frame)
                per_code_frames[code] = frame

        if not per_code_frames:
            raise ValueError("No minute data loaded. Check codes/years/date filters.")

        if end_dt is None and ModelConfig.CN_MINUTE_DAYS:
            cutoff_days = ModelConfig.CN_MINUTE_DAYS
            for code, frame in list(per_code_frames.items()):
                frame = frame.sort_values("date")
                if len(frame) > cutoff_days:
                    frame = frame.iloc[-cutoff_days:]
                per_code_frames[code] = frame

        def build_pivot(field: str) -> pd.DataFrame:
            series_list = []
            for code, frame in per_code_frames.items():
                s = frame.set_index("date")[field].rename(code)
                series_list.append(s)
            pivot = pd.concat(series_list, axis=1).sort_index()
            pivot = pivot.ffill().fillna(0.0)
            return pivot

        open_df = build_pivot("open")
        high_df = build_pivot("high")
        low_df = build_pivot("low")
        close_df = build_pivot("close")
        volume_df = build_pivot("volume")
        amount_df = build_pivot("amount")
        target_df = build_pivot("target_ret")
        adj_df = build_pivot("adj_factor") if ModelConfig.CN_USE_ADJ_FACTOR else None

        index = close_df.index
        columns = close_df.columns

        def to_tensor(pivot: pd.DataFrame) -> torch.Tensor:
            pivot = pivot.reindex(index=index, columns=columns)
            return torch.tensor(pivot.values.T, dtype=torch.float32, device=ModelConfig.DEVICE)

        self.raw_data_cache = {
            "open": to_tensor(open_df),
            "high": to_tensor(high_df),
            "low": to_tensor(low_df),
            "close": to_tensor(close_df),
            "volume": to_tensor(volume_df),
            "amount": to_tensor(amount_df),
            "liquidity": to_tensor(amount_df),
            "fdv": to_tensor(amount_df),
        }
        if adj_df is not None:
            self.raw_data_cache["adj_factor"] = to_tensor(adj_df)

        self.feat_tensor = FeatureEngineer.compute_features(self.raw_data_cache)
        self.target_ret = to_tensor(target_df)
        self.dates = index
        self.symbols = list(columns)

        print(f"CN Minute Data Ready. Shape: {self.feat_tensor.shape}")
```

model_core/engine.py
```python
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
```

model_core/factors.py
```python
import torch
import pandas as pd
import pandas_ta as ta

class FeatureEngineer:
    """Feature engineer for China A-share/ETF data using pandas_ta."""
    
    # 58 Features
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
    def robust_norm(t: torch.Tensor, clip: float = 5.0) -> torch.Tensor:
        """Robust Z-Score Normalization (Cross-Sectional or Time-Series)."""
        # Normalize along the last axis (time axis).
        # Works for both [Batch, Time] and [Asset, Feature, Time].
        median = torch.nanmedian(t, dim=-1, keepdim=True)[0]
        mad = torch.nanmedian(torch.abs(t - median), dim=-1, keepdim=True)[0] + 1e-6
        norm = (t - median) / mad
        return torch.clamp(norm, -clip, clip)

    @staticmethod
    def compute_features(raw_dict: dict[str, torch.Tensor]) -> torch.Tensor:
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
        
        # Helper to convert Tensor to DataFrame (for pandas_ta)
        def to_df(key):
            t = raw_dict[key].detach().cpu().numpy()
            return pd.DataFrame(t.T) # [Time, Batch] as columns

        # We must process each asset (column) individually or use pandas_ta machinery?
        # AShareGPT handles batch of assets (N Symbols).
        # pandas_ta is designed for single DataFrame (Time, OHLCV).
        # To be efficient, we iterate over pandas_ta functions, not assets.
        # But pandas_ta functions usually take Series. We can apply them to the whole DataFrame if structure permits,
        # but mostly they expect single Series.
        # Given "Batch" dimension is usually Symbols (e.g. 50), looping 50 times is fast enough.
        
        # Convert raw tensors to numpy for pandas processing
        # Structure: [Batch, Time] -> [Time, Batch] for DataFrame
        eps = 1e-8
        
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
                # Price Transform
                {"kind": "avgprice"},
                {"kind": "medprice"},
                {"kind": "typprice"},
                {"kind": "wclose"}, # requires OHLC
                
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
                {"kind": "sar"},
                
                # Volume
                {"kind": "obv"},
                {"kind": "ad"},
                {"kind": "adosc"},
                {"kind": "cmf"},
                {"kind": "mfi", "length": 14},
            ]
        )
        
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
            df.ta.strategy(CustomStrategy)
            
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
            feat_dict['AVGPRICE'] = get_col(["AVGPRICE"])
            feat_dict['MEDPRICE'] = get_col(["MEDPRICE"])
            feat_dict['TYPPRICE'] = get_col(["TYPPRICE"])
            feat_dict['WCLOSE'] = get_col(["HLC3", "WCP"]) 
            
            # 3. Returns
            feat_dict['RET'] = get_col(["PCTRET_1"])
            feat_dict['RET5'] = get_col(["PCTRET_5"])
            feat_dict['RET10'] = get_col(["PCTRET_10"])
            feat_dict['RET20'] = get_col(["PCTRET_20"])
            feat_dict['LOG_RET'] = get_col(["LOGRET_1"])
            
            # 4. Volatility
            feat_dict['TR'] = get_col(["TR", "TRUERANGE"])
            feat_dict['ATR14'] = get_col(["ATR_14"])
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
            feat_dict['SAR'] = get_col(["SAR"])
            
            # 7. Volume
            feat_dict['OBV'] = get_col(["OBV"])
            feat_dict['AD'] = get_col(["AD"])
            feat_dict['ADOSC'] = get_col(["ADOSC_3_10"])
            feat_dict['CMF'] = get_col(["CMF_20"])
            feat_dict['MFI14'] = get_col(["MFI_14"])
            
            # custom volume features not in pandas_ta strategy explicitly or need custom calc
            # V_RET
            v_curr = df['volume'].values
            v_prev = pd.Series(v_curr).shift(1).fillna(method='bfill').values
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
                    feat_out[i, f_idx, :] = torch.from_numpy(val).to(device)

        # Post-process: NaN handling & Normalization
        feat_out = torch.nan_to_num(feat_out, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply Robust Normalization to everything?
        # Yes, AlphaGPT expects roughly standard normal inputs.
        # But we do this across Time per Asset (Time-Series Norm)
        normalized = FeatureEngineer.robust_norm(feat_out)
        
        return normalized
```

model_core/ops.py
```python

from typing import Callable
import torch

@torch.jit.script
def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    if d == 0:
        return x
    pad = torch.zeros((x.shape[0], d), device=x.device)
    return torch.cat([pad, x[:, :-d]], dim=1)

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

OPS_CONFIG: list[tuple[str, Callable[..., torch.Tensor], int]] = [
    ('ADD', lambda x, y: x + y, 2),
    ('SUB', lambda x, y: x - y, 2),
    ('MUL', lambda x, y: x * y, 2),
    ('DIV', lambda x, y: x / (y + 1e-6 * torch.sign(y)), 2),
    ('NEG', lambda x: -x, 1),
    ('ABS', torch.abs, 1),
    ('SIGN', torch.sign, 1),
    ('DELTA5', lambda x: _ts_delta(x, 5), 1),
    ('MA20', lambda x: _ts_decay_linear(x, 20), 1),
    ('STD20', lambda x: _ts_zscore(x, 20), 1),
    ('TS_RANK20', lambda x: _ts_zscore(x, 20), 1),
]
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
                    if len(stack) < arity: return None
                    
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
            
            # Valid formula must result in exactly one value on the stack
            if len(stack) == 1:
                return stack[0]
            else:
                return None # Stack not empty (incomplete formula) or empty
                
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
import torch

from model_core.config import ModelConfig
from model_core.data_loader import ChinaMinuteDataLoader
from model_core.backtest import ChinaBacktest
from model_core.vm import StackVM


def load_strategy(filepath: str) -> list:
    with open(filepath, 'r') as f:
        return json.load(f)


def _format_pct(value: float) -> str:
    return f"{value:.2%}"


def print_metrics(title: str, result) -> None:
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)
    print(f"Sortino Score: {result.score.item():.4f}")
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
            "equity": result.equity_curve.numpy(),
            "return": result.portfolio_returns.numpy(),
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
    loader = ChinaMinuteDataLoader()
    loader.load_data(
        codes=symbols or ModelConfig.CN_CODES,
        years=ModelConfig.CN_MINUTE_YEARS,
        start_date=ModelConfig.CN_MINUTE_START_DATE,
        end_date=ModelConfig.CN_MINUTE_END_DATE,
        signal_time=ModelConfig.CN_SIGNAL_TIME,
        exit_time=ModelConfig.CN_EXIT_TIME,
        limit_codes=ModelConfig.CN_MAX_CODES,
    )

    print(f"Data shape: {loader.feat_tensor.shape}")
    print(f"Symbols: {len(loader.symbols) if loader.symbols else 'N/A'}")
    print(f"Date range: {loader.dates.min()} to {loader.dates.max()}")

    print("\nExecuting strategy formula...")
    vm = StackVM()
    factors = vm.execute(formula, loader.feat_tensor)

    if factors is None:
        print("❌ Invalid formula - execution failed")
        return

    if factors.std() < 1e-4:
        print("⚠️  Warning: Factor has near-zero variance (trivial formula)")

    print("\nRunning backtest...")
    bt = ChinaBacktest()

    if walk_forward:
        folds = loader.walk_forward_splits()
        if not folds:
            print("⚠️  Walk-forward disabled: not enough data for configured windows.")
        else:
            val_scores = []
            test_scores = []
            for idx, fold in enumerate(folds, 1):
                if fold.val.end_idx > fold.val.start_idx:
                    res_val = factors[:, fold.val.start_idx:fold.val.end_idx]
                    val_result = bt.evaluate(
                        res_val,
                        fold.val.raw_data_cache,
                        fold.val.target_ret,
                        return_details=True,
                    )
                    print_metrics(f"Fold {idx} - Validation", val_result)
                    val_scores.append(val_result.score.item())
                if fold.test.end_idx > fold.test.start_idx:
                    res_test = factors[:, fold.test.start_idx:fold.test.end_idx]
                    test_result = bt.evaluate(
                        res_test,
                        fold.test.raw_data_cache,
                        fold.test.target_ret,
                        return_details=True,
                    )
                    print_metrics(f"Fold {idx} - Test", test_result)
                    test_scores.append(test_result.score.item())
            if val_scores:
                print(f"\nWalk-forward Avg Val Score: {sum(val_scores) / len(val_scores):.4f}")
            if test_scores:
                print(f"Walk-forward Avg Test Score: {sum(test_scores) / len(test_scores):.4f}")
        return None

    if split:
        splits = loader.train_val_test_split()
        for name in ("train", "val", "test"):
            if name not in splits:
                continue
            split_slice = splits[name]
            res_slice = factors[:, split_slice.start_idx:split_slice.end_idx]
            result = bt.evaluate(
                res_slice,
                split_slice.raw_data_cache,
                split_slice.target_ret,
                return_details=True,
            )
            print_metrics(f"{name.capitalize()} Results", result)
            if curve_out:
                suffix = f"_{name}"
                out_path = Path(curve_out)
                out_file = out_path.with_name(f"{out_path.stem}{suffix}{out_path.suffix or '.csv'}")
                save_equity_curve(str(out_file), split_slice.dates, result)
        return None

    result = bt.evaluate(
        factors,
        loader.raw_data_cache,
        loader.target_ret,
        return_details=True,
    )

    print("\n" + "=" * 60)
    print("📊 Backtest Results")
    print("=" * 60)
    print_metrics("Full Sample", result)

    if curve_out:
        save_equity_curve(curve_out, loader.dates, result)

    return {
        'score': result.score.item(),
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

from model_core.engine import AlphaEngine


def main():
    print("=" * 60)
    print("🇨🇳 AShareGPT Training (Minute CSV)")
    print("=" * 60)
    print(f"Strategy Output: {os.environ.get('STRATEGY_FILE', 'best_cn_strategy.json')}")
    print()

    try:
        engine = AlphaEngine(use_lord_regularization=True)
        engine.train()

        print("\n" + "=" * 60)
        print("✅ Training Complete!")
        print("=" * 60)

        if engine.best_formula:
            print(f"Best Score: {engine.best_score:.4f}")
            print(f"Best Formula: {engine.best_formula}")
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

