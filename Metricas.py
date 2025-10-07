import numpy as np
import pandas as pd

def calcular_metricas(df):
    df = df.copy()
    df["returns"] = df["equity"].pct_change().fillna(0)
    mean = df["returns"].mean()
    std = df["returns"].std()
    neg_std = df.loc[df["returns"] < 0, "returns"].std()
    sharpe = mean / std * np.sqrt(252*24)
    sortino = mean / neg_std * np.sqrt(252*24)

    max_equity = df["equity"].cummax()
    drawdown = (df["equity"] - max_equity) / max_equity
    max_dd = drawdown.min()

    total_return = df["equity"].iloc[-1] / df["equity"].iloc[0] - 1
    years = len(df) / (24 * 365)
    cagr = (1 + total_return) ** (1 / years) - 1
    calmar = cagr / abs(max_dd)

    trades = df["signal"].diff().abs().sum() / 2
    win_rate = (df["returns"] > 0).sum() / len(df)

    return {
        "Sharpe": round(sharpe, 3),
        "Sortino": round(sortino, 3),
        "Calmar": round(calmar, 3),
        "Max Drawdown": round(max_dd*100, 2),
        "Win Rate": round(win_rate*100, 2)
    }

def retornos_periodicos(df):
    df.index = pd.to_datetime(df["Datetime"])
    df["ret"] = df["equity"].pct_change()
    mensual = df["ret"].resample("M").apply(lambda x: (1+x).prod()-1)
    trimestral = df["ret"].resample("Q").apply(lambda x: (1+x).prod()-1)
    anual = df["ret"].resample("Y").apply(lambda x: (1+x).prod()-1)
    return mensual, trimestral, anual
