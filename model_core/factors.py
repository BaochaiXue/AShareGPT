import torch


class FeatureEngineer:
    """Feature engineer for China A-share/ETF data."""

    FEATURES = ['RET', 'RET5', 'VOL_CHG', 'V_RET', 'TREND']
    INPUT_DIM = len(FEATURES)

    @staticmethod
    def _ts_mean(x: torch.Tensor, window: int) -> torch.Tensor:
        if window <= 1:
            return x
        pad = torch.zeros((x.shape[0], window - 1), device=x.device)
        x_pad = torch.cat([pad, x], dim=1)
        windows = x_pad.unfold(1, window, 1)
        return windows.mean(dim=-1)

    @staticmethod
    def compute_features(raw_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute 5 basic features aligned with A-share bars."""
        c = raw_dict['close']
        v = raw_dict['volume']

        prev_close = torch.roll(c, 1, dims=1)
        ret = (c - prev_close) / (prev_close + 1e-6)
        ret[:, 0] = 0.0

        c5 = torch.roll(c, 5, dims=1)
        ret5 = (c - c5) / (c5 + 1e-6)
        ret5[:, :5] = 0.0

        vol_ma20 = FeatureEngineer._ts_mean(v, 20)
        vol_chg = torch.where(vol_ma20 > 0, v / (vol_ma20 + 1e-6) - 1.0, torch.zeros_like(v))
        v_ret = ret * (vol_chg + 1.0)

        ma60 = FeatureEngineer._ts_mean(c, 60)
        trend = torch.where(ma60 > 0, c / (ma60 + 1e-6) - 1.0, torch.zeros_like(c))

        def robust_norm(t):
            median = torch.nanmedian(t, dim=1, keepdim=True)[0]
            mad = torch.nanmedian(torch.abs(t - median), dim=1, keepdim=True)[0] + 1e-6
            norm = (t - median) / mad
            return torch.clamp(norm, -5.0, 5.0)

        features = torch.stack([
            robust_norm(ret),
            robust_norm(ret5),
            robust_norm(vol_chg),
            robust_norm(v_ret),
            robust_norm(trend)
        ], dim=1)

        return features
