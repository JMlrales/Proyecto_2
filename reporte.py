import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from Metricas import sharpe_ratio, sortino_ratio, get_calmar, max_drawdown, win_rate

sns.set(style="whitegrid", context="talk")

# === FUNCIONES DE ANÁLISIS ===
def retornos_periodicos(df):
    """Agrupa retornos por mes, trimestre y año."""
    df = df.copy()
    df["ret"] = df["equity"].pct_change()
    df["Date"] = pd.to_datetime(df["Date"])

    mensual = df.resample("M", on="Date")["ret"].apply(lambda x: (1 + x).prod() - 1)
    trimestral = df.resample("Q", on="Date")["ret"].apply(lambda x: (1 + x).prod() - 1)
    anual = df.resample("Y", on="Date")["ret"].apply(lambda x: (1 + x).prod() - 1)

    return mensual, trimestral, anual


# === GRÁFICAS ===
def graficar_equity(df):
    plt.figure(figsize=(12,6))
    plt.plot(df["Date"], df["equity"], color="royalblue", lw=2, label="Portafolio")
    plt.title("Evolución del valor del portafolio")
    plt.xlabel("Fecha")
    plt.ylabel("Valor (USD)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def graficar_drawdown(df):
    df = df.copy()
    df["drawdown"] = df["equity"] / df["equity"].cummax() - 1
    plt.figure(figsize=(12,4))
    plt.fill_between(df["Date"], df["drawdown"], color="crimson", alpha=0.4)
    plt.title("Drawdown (%)")
    plt.ylabel("Porcentaje")
    plt.xlabel("Fecha")
    plt.tight_layout()
    plt.show()


def histograma_retornos(df):
    df = df.copy()
    df["returns"] = df["equity"].pct_change()
    plt.figure(figsize=(10,5))
    sns.histplot(df["returns"].dropna(), bins=60, kde=True, color="teal")
    plt.title("Distribución de retornos horarios")
    plt.xlabel("Retorno (%)")
    plt.tight_layout()
    plt.show()


def heatmap_mensual(df):
    df = df.copy()
    df["ret"] = df["equity"].pct_change()
    df["Year"] = pd.to_datetime(df["Date"]).dt.year
    df["Month"] = pd.to_datetime(df["Date"]).dt.month
    pivot = df.groupby(["Year", "Month"])["ret"].apply(lambda x: (1+x).prod()-1).unstack()
    plt.figure(figsize=(10,6))
    sns.heatmap(pivot*100, annot=True, fmt=".2f", cmap="RdYlGn", center=0)
    plt.title("Mapa de calor de rendimientos mensuales (%)")
    plt.ylabel("Año")
    plt.xlabel("Mes")
    plt.show()


# === REPORTE FINAL ===
def generar_reporte_visual(df):
    print("\n📈 === MÉTRICAS CLAVE ===")
    metrics = {
        "Sharpe Ratio": sharpe_ratio(df),
        "Sortino Ratio": sortino_ratio(df),
        "Calmar Ratio": get_calmar(df),
        "Max Drawdown": max_drawdown(df),
        "Win Rate": win_rate(df)
    }
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:<20}: {v:.4f}")
        else:
            print(f"{k:<20}: {v}")

    mensual, trimestral, anual = retornos_periodicos(df)
    print("\n📊 === RENDIMIENTOS AGREGADOS ===")
    print("\nMensuales:\n", (mensual.tail(12) * 100).round(2))
    print("\nTrimestrales:\n", (trimestral.tail(8) * 100).round(2))
    print("\nAnuales:\n", (anual * 100).round(2))

    # === VISUALIZACIONES ===
    graficar_equity(df)
    graficar_drawdown(df)
    histograma_retornos(df)
    heatmap_mensual(df)

