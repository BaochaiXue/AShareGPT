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
