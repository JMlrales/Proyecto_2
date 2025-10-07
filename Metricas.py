import numpy as np
import pandas as pd


def get_calmar(df):
    equity = df["equity"].dropna()
    ret = equity.pct_change().dropna()
    cum_return = (1 + ret).prod() - 1
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1
    max_dd = abs(drawdown.min())
    if max_dd == 0:
        return 0.0
    return cum_return / max_dd


def sharpe_ratio(df):
    r = df["equity"].pct_change().dropna()
    if r.std() == 0:
        return 0.0
    return (r.mean() / r.std()) * np.sqrt(252 * 24)


def sortino_ratio(df):
    r = df["equity"].pct_change().dropna()
    neg = r[r < 0]
    if neg.std() == 0:
        return 0.0
    return (r.mean() / neg.std()) * np.sqrt(252 * 24)


def max_drawdown(df):
    equity = df["equity"]
    roll_max = equity.cummax()
    dd = equity / roll_max - 1
    return dd.min()


def win_rate(df):
    r = df["equity"].pct_change().dropna()
    return (r > 0).sum() / len(r)

