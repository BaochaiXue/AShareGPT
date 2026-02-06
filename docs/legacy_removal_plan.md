# Legacy Package Removal (v0.6.0)

This project has completed removal of `model_core.infrastructure.legacy`.

## Current State

- `model_core.infrastructure.legacy` no longer exists in the codebase.
- The canonical compatibility trainer import is:
  - `from model_core.infrastructure.adapters import LegacyAlphaTrainer`
- CLI entrypoints remain unchanged:
  - `python run_cn_train.py`
  - `python run_cn_backtest.py ...`

## Migration Guide

Replace old imports:

```python
from model_core.infrastructure.legacy import LegacyAlphaTrainer
```

with:

```python
from model_core.infrastructure.adapters import LegacyAlphaTrainer
```

## Breaking Change Note

- This is a source-level breaking change only for callers importing the removed legacy namespace.
- Runtime behavior of training/backtest CLI workflows is unchanged.
