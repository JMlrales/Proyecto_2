import pandas as pd
from dataclasses import dataclass


@dataclass
class Operation:
    time: pd.Timestamp
    price: float
    stop_loss: float
    take_profit: float
    n_shares: float
    type: str  # "LONG" o "SHORT"


def get_portfolio_value(cash: float, long_ops: list[Operation],
                        short_ops: list[Operation], current_price: float,
                        COM: float) -> float:
    """
    Calcula el valor total del portafolio (efectivo + valor de posiciones abiertas).
    """
    val = cash

    # Valor de posiciones largas abiertas
    for pos in long_ops:
        val += current_price * pos.n_shares

    # Valor de posiciones cortas abiertas
    for pos in short_ops:
        val += (pos.price - current_price) * pos.n_shares * (1 - COM)

    return val


def backtest(data: pd.DataFrame, SL: float, TP: float, N: float,
             COM: float = 0.00125, capital_inicial: float = 1000000) -> pd.DataFrame:
    """
    Backtest con gestión de posiciones LONG y SHORT, stop loss, take profit y
    límite máximo de exposición equivalente a 1,000,000 USD en BTC.
    N representa un factor (0 a 1) del máximo de exposición permitida.
    """

    data = data.copy()
    cash = capital_inicial
    active_long = []
    active_short = []
    portfolio_values = []

    max_inversion_usd = 1_000_000  # límite de exposición
    n_shares_factor = N

    for i, row in data.iterrows():
        precio = float(row["Close"])
        señal = int(row.get("signal", 0))

        # === CIERRE DE POSICIONES LARGAS ===
        for pos in active_long.copy():
            if (pos.stop_loss > precio) or (pos.take_profit < precio):
                # Cierra la posición long
                cash += precio * pos.n_shares * (1 - COM)
                active_long.remove(pos)

        # === CIERRE DE POSICIONES CORTAS ===
        for pos in active_short.copy():
            if (pos.stop_loss < precio) or (pos.take_profit > precio):
                # Fórmula según pizarrón
                cash += (pos.price * pos.n_shares) + ((pos.price - precio) * pos.n_shares) * (1 - COM)
                active_short.remove(pos)

        # === APERTURA DE NUEVAS POSICIONES ===
        # Solo si no hay una posición abierta
        if len(active_long) == 0 and len(active_short) == 0:
            max_btc = max_inversion_usd / precio
            cantidad = n_shares_factor * max_btc

            # Apertura LONG
            if señal == 1 and cash >= cantidad * precio * (1 + COM):
                active_long.append(Operation(
                    time=row["Date"],
                    price=precio,
                    stop_loss=precio * (1 - SL),
                    take_profit=precio * (1 + TP),
                    n_shares=cantidad,
                    type="LONG"
                ))
                cash -= cantidad * precio * (1 + COM)

            # Apertura SHORT
            elif señal == -1 and cash >= cantidad * precio * (1 + COM):
                active_short.append(Operation(
                    time=row["Date"],
                    price=precio,
                    stop_loss=precio * (1 + SL),
                    take_profit=precio * (1 - TP),
                    n_shares=cantidad,
                    type="SHORT"
                ))
                cash -= cantidad * precio * (1 + COM)

        # === VALOR TOTAL DEL PORTAFOLIO ===
        port_val = get_portfolio_value(cash, active_long, active_short, precio, COM)
        portfolio_values.append(port_val)

    # === CIERRE FINAL ===
    if len(active_long) > 0:
        for pos in active_long:
            cash += data["Close"].iloc[-1] * pos.n_shares * (1 - COM)

    if len(active_short) > 0:
        for pos in active_short:
            cash += (pos.price * pos.n_shares) + ((pos.price - data["Close"].iloc[-1]) * pos.n_shares) * (1 - COM)

    data["equity"] = portfolio_values
    return data

