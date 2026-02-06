
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
        self.vocab_size = len(self.vocab)
        
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
        x = self.blocks(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        
        # Use the representation of the *last* token to predict the next token
        last_emb = x[:, -1, :] # [Batch, D_Model]
        
        # Multi-task pooling head for logits
        logits, task_probs = self.mtp_head(last_emb) # logits: [Batch, Vocab], task_probs: [Batch, Tasks]
        value = self.head_critic(last_emb)       # [Batch, 1]
        
        return logits, value, task_probs
