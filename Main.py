import pandas as pd
import optuna
from Get_Signals import generar_indicadores, generar_senal
from Backtesting import backtest
from Metricas import sharpe_ratio, sortino_ratio, get_calmar, max_drawdown, win_rate
from Optimize import optimize
from reporte import generar_reporte_visual


# ===========================
# CONFIGURACIÓN GENERAL
# ===========================
DATA_PATH = "Binance_BTCUSDT_1h.csv"
CAPITAL_INICIAL = 1_000_000  # dinero invertido real
COMISION = 0.00125


# ===========================
# CARGA Y LIMPIEZA DE DATOS
# ===========================
df = pd.read_csv(DATA_PATH)

# La columna de fecha se llama “Date” en tu dataset original
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y %H:%M", errors="coerce")
df.dropna(subset=["Date", "Close", "High", "Low"], inplace=True)
df = df.sort_values("Date").reset_index(drop=True)

print(f"✅ Datos cargados: {len(df)} filas")
print(f"Rango temporal: {df['Date'].min()} → {df['Date'].max()}")


# ===========================
# OPTIMIZACIÓN DE PARÁMETROS
# ===========================
print("\n⚙️ Iniciando optimización con Optuna (Calmar Ratio)...")
study = optuna.create_study(direction="maximize")
study.optimize(lambda trial: optimize(trial, df), n_trials=30)

best_params = study.best_params
print("\n=== MEJORES PARÁMETROS ENCONTRADOS ===")
for k, v in best_params.items():
    print(f"{k}: {v:.5f}")
print(f"Mejor Calmar promedio: {study.best_value:.4f}")


# ===========================
# GENERACIÓN DE SEÑALES
# ===========================
print("\n📊 Generando señales con parámetros óptimos...")
df_ind = generar_indicadores(df)
df_sig = generar_senal(df_ind)

# Mostrar conteo de señales para confirmar que sí opera
print("\n📈 Conteo de señales:")
print(df_sig["signal"].value_counts(dropna=False))


# ===========================
# BACKTEST FINAL
# ===========================
print("\n🚀 Ejecutando backtest final con 1,000,000 USD de inversión...")
resultados = backtest(
    df_sig,
    SL=best_params["stop_loss"],
    TP=best_params["take_profit"],
    N=best_params["n_shares"],
    COM=COMISION,
    capital_inicial=CAPITAL_INICIAL
)

# Confirmar que equity se movió
print(f"\nEquity inicial: {resultados['equity'].iloc[0]:.2f}")
print(f"Equity final:   {resultados['equity'].iloc[-1]:.2f}")


# ===========================
# MÉTRICAS DE DESEMPEÑO
# ===========================
print("\n📈 === MÉTRICAS DE DESEMPEÑO ===")
print(f"Sharpe Ratio:     {sharpe_ratio(resultados):.4f}")
print(f"Sortino Ratio:    {sortino_ratio(resultados):.4f}")
print(f"Calmar Ratio:     {get_calmar(resultados):.4f}")
print(f"Max Drawdown:     {max_drawdown(resultados):.2%}")
print(f"Win Rate:         {win_rate(resultados):.2%}")


# ===========================
# REPORTE VISUAL Y TABLAS
# ===========================
print("\n📊 Generando reporte visual y tablas de rendimiento...")
generar_reporte_visual(resultados)

print("\n✅ Proceso completado exitosamente.")

