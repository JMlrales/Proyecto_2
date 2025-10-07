import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD

def generar_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # === INDICADORES ===
    rsi = RSIIndicator(df["Close"], window=14).rsi()
    stoch = StochasticOscillator(df["High"], df["Low"], df["Close"], window=14, smooth_window=3)
    k = stoch.stoch()
    d = stoch.stoch_signal()
    macd_ind = MACD(df["Close"], window_fast=12, window_slow=26, window_sign=9)
    macd_line = macd_ind.macd()
    macd_signal = macd_ind.macd_signal()
    macd_hist = macd_ind.macd_diff()

    # === UNIR TODO ===
    df["rsi"] = rsi
    df["k"] = k
    df["d"] = d
    df["macd"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist

    # Eliminar las filas iniciales con NaN
    df.dropna(inplace=True)
    return df


def generar_senal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    señales = []

    for i in range(len(df)):
        rsi = df["rsi"].iloc[i]
        k = df["k"].iloc[i]
        d = df["d"].iloc[i]
        macd_line = df["macd"].iloc[i]
        macd_sig = df["macd_signal"].iloc[i]
        hist = df["macd_hist"].iloc[i]

        long_cond = 0
        short_cond = 0

        # === RSI ===
        if rsi < 40:
            long_cond += 1
        elif rsi > 60:
            short_cond += 1

        # === ESTOCÁSTICO ===
        if k < 30 and k > d:
            long_cond += 1
        elif k > 70 and k < d:
            short_cond += 1

        # === MACD ===
        if macd_line > macd_sig and hist > 0:
            long_cond += 1
        elif macd_line < macd_sig and hist < 0:
            short_cond += 1

        # === CONFIRMACIÓN (2 de 3 indicadores) ===
        if long_cond >= 2 and short_cond == 0:
            señales.append(1)   # Compra
        elif short_cond >= 2 and long_cond == 0:
            señales.append(-1)  # Venta
        else:
            señales.append(0)   # Sin señal

    df["signal"] = señales
    return df
