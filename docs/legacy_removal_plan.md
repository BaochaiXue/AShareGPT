# Legacy Package Removal Plan

This project is deprecating `model_core.infrastructure.legacy`.

## Version Plan

- **v0.5.0 (current deprecation release)**
  - Keep compatibility import path:
    - `model_core.infrastructure.legacy`
  - Emit runtime deprecation warning on legacy import.
  - Primary path is now:
    - `model_core.infrastructure.adapters`

- **v0.6.0 (next release)**
  - Remove `model_core.infrastructure.legacy` package entirely.
  - Remove compatibility re-export code and deprecation constants.
  - Keep only `model_core.infrastructure.adapters` imports.

## Migration Checklist

- Replace all imports:
  - `from model_core.infrastructure.legacy import LegacyAlphaTrainer`
  - with:
  - `from model_core.infrastructure.adapters import LegacyAlphaTrainer`
- Remove any references to legacy-only modules under `model_core/infrastructure/legacy`.
- Keep CLI entrypoints unchanged (`run_cn_train.py`, `run_cn_backtest.py`).
