"""Screener con Yahoo Finance: acciones ARG (.BA) y subyacentes de CEDEARs (USA).
Uso:  pip install yfinance pandas
      python screener_yahoo.py
Si existe tickers.txt (un ticker por linea), se usa en lugar de la lista incluida.
Tickers con .BA => ARG. Sin sufijo => USA (subyacente del CEDEAR, con mejor volumen).
Salida: resultado_screener.csv (cargalo en el panel). No es recomendacion de inversion."""
import os
import numpy as np
import pandas as pd
import yfinance as yf

ARG = ["GGAL.BA", "YPFD.BA", "PAMP.BA", "BMA.BA", "TXAR.BA", "ALUA.BA", "CEPU.BA",
       "EDN.BA", "SUPV.BA", "TECO2.BA", "TGSU2.BA", "TRAN.BA", "CRES.BA", "LOMA.BA",
       "VALO.BA", "BYMA.BA", "COME.BA", "MIRG.BA", "TGNO4.BA", "CVH.BA"]
USA = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "KO", "MELI", "AMD",
       "NFLX", "DIS", "JPM", "XOM", "PBR", "VIST", "BABA", "SPY", "QQQ", "AVGO",
       "TSM", "MU", "PLTR", "COIN", "V", "WMT", "PEP", "CVX", "BAC", "INTC"]

def leer_lista():
    if os.path.exists("tickers.txt"):
        with open("tickers.txt") as f:
            return [x.strip() for x in f if x.strip()]
    return ARG + USA

def fvg_alcista(h, l, c, ventana=10):
    """Ultimo FVG alcista sin cerrar en las ultimas `ventana` velas, en % del cierre."""
    n = len(c)
    for i in range(n - 1, max(n - 1 - ventana, 1), -1):
        if l[i] > h[i - 2]:
            cerrado = bool((l[i + 1:] <= h[i - 2]).any()) if i + 1 < n else False
            if not cerrado:
                return (l[i] - h[i - 2]) / c[-1] * 100
    return 0.0

def analizar(df, t):
    df = df.dropna(subset=["Close", "High", "Low"])
    if len(df) < 25:
        return None
    c = df["Close"].astype(float).values
    h = df["High"].astype(float).values
    l = df["Low"].astype(float).values
    v = df["Volume"].astype(float)
    prom = v.iloc[-21:-1].mean()
    return {"Símbolo": t, "Precio": c[-1],
            "Cambio %": (c[-1] / c[-2] - 1) * 100,
            "Perf % 1 mes": (c[-1] / c[-22] - 1) * 100,
            "Volumen relativo": v.iloc[-1] / prom if prom else np.nan,
            "FVG %": fvg_alcista(h, l, c),
            "País": "ARG" if t.upper().endswith(".BA") else "USA"}

if __name__ == "__main__":
    lista = leer_lista()
    filas = []
    for i in range(0, len(lista), 25):  # de a 25 para no saturar a Yahoo
        lote = lista[i:i + 25]
        datos = yf.download(lote, period="6mo", interval="1d", group_by="ticker",
                            auto_adjust=False, progress=False, threads=True)
        for t in lote:
            try:
                d = datos[t] if len(lote) > 1 else datos
                r = analizar(d, t)
                if r:
                    filas.append(r)
                else:
                    print(f"{t}: pocos datos, se omite")
            except Exception as e:
                print(f"{t}: {e}")
        print(f"{min(i + 25, len(lista))}/{len(lista)}")
    pd.DataFrame(filas).to_csv("resultado_screener.csv", index=False)
    print(f"Listo: {len(filas)} especies en resultado_screener.csv")
