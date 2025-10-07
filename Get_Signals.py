import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
import optuna

def generar_indicadores(df, rsi_window=14, stoch_window=14, smooth_window=3,
                        ema_lenta=26, ema_rapida=12, ema_signal=9):
    df = df.copy()
    df["RSI"] = RSIIndicator(df["Close"], window=rsi_window).rsi()
    st = StochasticOscillator(df["High"], df["Low"], df["Close"],
                              window=stoch_window, smooth_window=smooth_window)
    df["%K"] = st.stoch()
    df["%D"] = st.stoch_signal()
    macd = MACD(df["Close"], ema_lenta, ema_rapida, ema_signal)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    return df.dropna()

def generar_senal(df, rsi_lower=30, rsi_upper=70):
    señales = []
    for _, row in df.iterrows():
        senales_individuales = []
        # Señales individuales
        senales_individuales.append(1 if row["RSI"] < rsi_lower else (-1 if row["RSI"] > rsi_upper else 0))
        senales_individuales.append(1 if row["%K"] > row["%D"] and row["%K"] < 80 else (-1 if row["%K"] < row["%D"] and row["%K"] > 20 else 0))
        senales_individuales.append(1 if row["MACD"] > row["MACD_signal"] else (-1 if row["MACD"] < row["MACD_signal"] else 0))
        # Confirmación por mayoría
        decision = 1 if senales_individuales.count(1) >= 2 else (-1 if senales_individuales.count(-1) >= 2 else 0)
        señales.append(decision)
    df["signal"] = señales
    return df

def optimizar_parametros(df):
    def objective(trial):
        rsi_window = trial.suggest_int("rsi_window", 8, 25)
        stoch_window = trial.suggest_int("stoch_window", 8, 20)
        ema_rapida = trial.suggest_int("ema_rapida", 10, 14)
        ema_lenta = trial.suggest_int("ema_lenta", 20, 30)
        df_ind = generar_indicadores(df, rsi_window, stoch_window, 3, ema_lenta, ema_rapida)
        df_sig = generar_senal(df_ind)
        returns = df_sig["signal"].shift(1) * df_sig["Close"].pct_change()
        return (returns.mean() / returns.std()) * np.sqrt(252)  # Sharpe
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    return study.best_params

