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
    def _get_col_values(
        df: pd.DataFrame,
        name_patterns: list[str],
        *,
        asset_idx: int,
        strict_indicator_mapping: bool,
        missing_indicator_patterns: set[str],
        fallback_values,
    ):
        """Return an indicator column by exact/prefix match with optional strictness."""
        for pat in name_patterns:
            if pat in df.columns:
                return df[pat].values
        for col in df.columns:
            for pat in name_patterns:
                if col.startswith(pat):
                    return df[col].values

        pattern_desc = " | ".join(name_patterns)
        if strict_indicator_mapping:
            raise ValueError(
                f"Missing indicator mapping [{pattern_desc}] for asset index {asset_idx}. "
                "Set CN_STRICT_FEATURE_INDICATORS=0 to downgrade to zero-fallback."
            )
        missing_indicator_patterns.add(pattern_desc)
        return fallback_values * 0

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
            name="NeuralSymbolicAlphaGenerator Strategy",
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
        def get_col(df: pd.DataFrame, asset_idx: int, name_patterns: list[str]):
            return FeatureEngineer._get_col_values(
                df,
                name_patterns,
                asset_idx=asset_idx,
                strict_indicator_mapping=strict_indicator_mapping,
                missing_indicator_patterns=missing_indicator_patterns,
                fallback_values=df["close"].values,
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
            # Keep raw volume for features; use a safe copy for TA indicators.
            df_safe = df.copy()
            df_safe['volume'] = df_safe['volume'].replace(0, 1e-4)
            
            # Run Strategy
            ta_accessor = df_safe.ta
            if hasattr(ta_accessor, "cores"):
                # Avoid multiprocessing path instability across pandas-ta variants.
                ta_accessor.cores = 0
            ta_accessor.strategy(CustomStrategy)
            
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
            feat_dict['RET'] = get_col(df_safe, i, ["PCTRET_1"])
            feat_dict['RET5'] = get_col(df_safe, i, ["PCTRET_5"])
            feat_dict['RET10'] = get_col(df_safe, i, ["PCTRET_10"])
            feat_dict['RET20'] = get_col(df_safe, i, ["PCTRET_20"])
            feat_dict['LOG_RET'] = get_col(df_safe, i, ["LOGRET_1"])
            
            # 4. Volatility
            feat_dict['TR'] = get_col(df_safe, i, ["TR", "TRUERANGE"])
            feat_dict['ATR14'] = get_col(df_safe, i, ["ATR_14", "ATRr_14"])
            feat_dict['NATR14'] = get_col(df_safe, i, ["NATR_14"])
            
            # 5. Momentum
            feat_dict['RSI14'] = get_col(df_safe, i, ["RSI_14"])
            feat_dict['RSI24'] = get_col(df_safe, i, ["RSI_24"])
            feat_dict['MACD'] = get_col(df_safe, i, ["MACD_12_26_9"])
            feat_dict['MACDh'] = get_col(df_safe, i, ["MACDh_12_26_9"])
            feat_dict['MACDs'] = get_col(df_safe, i, ["MACDs_12_26_9"])
            feat_dict['BOP'] = get_col(df_safe, i, ["BOP"])
            feat_dict['CCI14'] = get_col(df_safe, i, ["CCI_14_0.015"])
            feat_dict['CMO14'] = get_col(df_safe, i, ["CMO_14"])
            feat_dict['KDJ_K'] = get_col(df_safe, i, ["K_9_3"])
            feat_dict['KDJ_D'] = get_col(df_safe, i, ["D_9_3"])
            feat_dict['KDJ_J'] = get_col(df_safe, i, ["J_9_3"])
            feat_dict['MOM10'] = get_col(df_safe, i, ["MOM_10"])
            feat_dict['ROC10'] = get_col(df_safe, i, ["ROC_10"])
            feat_dict['PPO'] = get_col(df_safe, i, ["PPO_12_26_9"])
            feat_dict['PPOh'] = get_col(df_safe, i, ["PPOh_12_26_9"])
            feat_dict['PPOs'] = get_col(df_safe, i, ["PPOs_12_26_9"])
            feat_dict['TSI'] = get_col(df_safe, i, ["TSI_13_25_13"])
            feat_dict['UO'] = get_col(df_safe, i, ["UO_7_14_28"])
            feat_dict['WILLR'] = get_col(df_safe, i, ["WILLR_14"])
            
            # 6. Overlap
            feat_dict['SMA5'] = get_col(df_safe, i, ["SMA_5"])
            feat_dict['SMA10'] = get_col(df_safe, i, ["SMA_10"])
            feat_dict['SMA20'] = get_col(df_safe, i, ["SMA_20"])
            feat_dict['SMA60'] = get_col(df_safe, i, ["SMA_60"])
            feat_dict['EMA5'] = get_col(df_safe, i, ["EMA_5"])
            feat_dict['EMA10'] = get_col(df_safe, i, ["EMA_10"])
            feat_dict['EMA20'] = get_col(df_safe, i, ["EMA_20"])
            feat_dict['EMA60'] = get_col(df_safe, i, ["EMA_60"])
            feat_dict['TEMA10'] = get_col(df_safe, i, ["TEMA_10"])
            
            # Strategy requests BBANDS length=20, so map only to 20-bar outputs.
            feat_dict['BB_UPPER'] = get_col(df_safe, i, ["BBU_20_2.0", "BBU_20_2"])
            feat_dict['BB_MID'] = get_col(df_safe, i, ["BBM_20_2.0", "BBM_20_2"])
            feat_dict['BB_LOWER'] = get_col(df_safe, i, ["BBL_20_2.0", "BBL_20_2"])
            feat_dict['BB_WIDTH'] = get_col(df_safe, i, ["BBB_20_2.0", "BBB_20_2"])
            
            feat_dict['MIDPOINT'] = get_col(df_safe, i, ["MIDPOINT_2"])
            feat_dict['MIDPRICE'] = get_col(df_safe, i, ["MIDPRICE_2"])
            if "PSARl_0.02_0.2" in df_safe.columns and "PSARs_0.02_0.2" in df_safe.columns:
                sar_series = df_safe["PSARl_0.02_0.2"].combine_first(df_safe["PSARs_0.02_0.2"]).fillna(0.0)
                feat_dict['SAR'] = sar_series.values
            elif "PSARl_0.02_0.2" in df_safe.columns:
                feat_dict['SAR'] = df_safe["PSARl_0.02_0.2"].fillna(0.0).values
            elif "PSARs_0.02_0.2" in df_safe.columns:
                feat_dict['SAR'] = df_safe["PSARs_0.02_0.2"].fillna(0.0).values
            else:
                feat_dict['SAR'] = get_col(df_safe, i, ["SAR"])
            
            # 7. Volume
            feat_dict['OBV'] = get_col(df_safe, i, ["OBV"])
            feat_dict['AD'] = get_col(df_safe, i, ["AD"])
            feat_dict['ADOSC'] = get_col(df_safe, i, ["ADOSC_3_10"])
            feat_dict['CMF'] = get_col(df_safe, i, ["CMF_20"])
            typ_price = (df_safe["high"] + df_safe["low"] + df_safe["close"]) / 3.0
            raw_money = typ_price * df_safe["volume"]
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
