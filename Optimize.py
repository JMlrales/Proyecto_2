import numpy as np
import optuna
from Backtesting import backtest
from Metricas import get_calmar


def optimize(trial, train_data):
    """
    Función objetivo para Optuna basada en el esquema del pizarrón.
    Divide los datos en varios segmentos (n_splits) y promedia el Calmar ratio.
    """

    data = train_data.copy()

    # === trial.suggest_params (según pizarrón) ===
    stop_loss = trial.suggest_float('stop_loss', 0.02, 0.05)
    take_profit = trial.suggest_float('take_profit', 0.04, 0.15)
    n_shares = trial.suggest_int('n_shares', 1, 10)

    n_splits = 5
    len_data = len(data)
    calmars = []

    for i in range(n_splits):
        size = len_data // n_splits
        start_idx = i * size
        end_idx = (i + 1) * size if i < n_splits - 1 else len_data
        chunk = data.iloc[start_idx:end_idx].copy()

        # === Ejecutar backtest sobre el chunk ===
        port_vals = backtest(chunk, stop_loss, take_profit, n_shares)

        # === Calcular Calmar ratio del segmento ===
        calmar = get_calmar(port_vals)
        calmars.append(calmar)

    # === Promedio de Calmar en los splits ===
    avg_calmar = np.mean(calmars)

    return avg_calmar
