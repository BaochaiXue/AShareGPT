from typing import Optional, Callable
import torch
from .ops import get_ops_config
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
        
        # Build lookup tables for operators (frequency-adaptive)
        ops_config = get_ops_config()
        self.op_map: dict[int, Callable[..., torch.Tensor]] = {
            i + self.feat_offset: cfg[1] for i, cfg in enumerate(ops_config)
        }
        self.arity_map: dict[int, int] = {
            i + self.feat_offset: cfg[2] for i, cfg in enumerate(ops_config)
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
