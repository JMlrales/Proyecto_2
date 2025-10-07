import pandas as pd
import numpy as np

def backtest(df, comision:float, capital_inicial:float):
    df = df.copy()
    cash = capital_inicial
    posicion = 0.0
    equity = []
    estado = "none"

    for i in range(1, len(df)):
        precio = df["Close"].iloc[i]
        señal = df["signal"].iloc[i]

        # Cerrar o revertir posición
        if estado == "long" and señal == -1:
            cash += posicion * precio * (1 - comision)
            posicion = 0
            estado = "none"
        elif estado == "short" and señal == 1:
            cash -= posicion * precio * (1 + comision)
            posicion = 0
            estado = "none"

        # Abrir posición
        if estado == "none" and señal == 1:
            cantidad = (cash * (1 - comision)) / precio
            posicion = cantidad
            cash = 0
            estado = "long"
        elif estado == "none" and señal == -1:
            cantidad = (cash * (1 - comision)) / precio
            posicion = -cantidad
            cash += cantidad * precio * (1 - comision)
            estado = "short"

        # Calcular equity
        equity_val = cash + posicion * precio
        equity.append(equity_val)

    df = df.iloc[1:].copy()
    df["equity"] = equity
    return df
