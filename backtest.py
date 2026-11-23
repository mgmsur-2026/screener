"""Rendimiento histórico de la señal POTENCIAL ALTO (usa los mismos parámetros que el screener).
Recorre 5 años de cada ticker, día por día, como si se hubiera corrido ese día, y mide qué pasó
5, 10 y 20 ruedas después. Salida: backtest_resumen.json y backtest_eventos.csv.
Limitaciones: sin comisiones ni dólar, solo 5 años, muestra pequeña, sin garantía a futuro."""
import json
import numpy as np
import pandas as pd
import screener_yahoo as S

HORIZ, ENFRIAR, MINBARS = (5, 10, 20), 10, 300

def stats(x):
    x = np.array(x, dtype=float)
    return {"n": int(len(x)), "gana_pct": round(float((x > 0).mean() * 100), 1),
            "promedio_pct": round(float(x.mean() * 100), 2), "mediana_pct": round(float(np.median(x) * 100), 2)} if len(x) else {"n": 0}

if __name__ == "__main__":
    lista, eventos = S.cargar(), []
    spy = S.bajar(["SPY"], "5y")["SPY"]["Close"].dropna()
    base = {f"h{h}": stats((spy.shift(-h) / spy - 1).dropna().values) for h in HORIZ}
    for i in range(0, len(lista), 20):
        lote = lista[i:i + 20]
        datos = S.bajar([m["Yahoo"] for m in lote], "5y")
        for m in lote:
            d = datos[m["Yahoo"]].dropna(subset=["Close", "High", "Low", "Open"])
            n, ult = len(d), -999
            for k in range(MINBARS, n - 1):
                if k - ult < ENFRIAR:
                    continue
                r = S.analizar(d.iloc[:k + 1], m)
                if r and r["Señal"] == "POTENCIAL ALTO":
                    ult, e = k, d.iloc[k]
                    ev = {"Símbolo": m["Símbolo"], "Fecha": str(d.index[k].date()), "Precio": float(e["Close"]),
                          "Escenario": r["Escenario"], "Semáforo": "", "Etapa": r["Etapa"]}
                    for h in HORIZ:
                        ev[f"ret{h}"] = d["Close"].iloc[k + h] / e["Close"] - 1 if k + h < n else np.nan
                    f = d.iloc[k + 1:k + 21]
                    ev["max_alza20"] = f["High"].max() / e["Close"] - 1
                    ev["max_caida20"] = f["Low"].min() / e["Close"] - 1
                    ev["bajo_piso"] = bool(f["Low"].min() < e["Close"] * (1 - r["Riesgo piso %"] / 100))
                    eventos.append(ev)
        print(f"{min(i + 20, len(lista))}/{len(lista)}  eventos: {len(eventos)}")
    df = pd.DataFrame(eventos)
    df.to_csv("backtest_eventos.csv", index=False)
    res = {"eventos": int(len(df)), "tickers": int(df["Símbolo"].nunique()) if len(df) else 0,
           "referencia_SPY": base, "nota": "Sin comisiones ni dólar; 5 años; no garantiza resultados futuros."}
    if len(df):
        for h in HORIZ:
            res[f"h{h}"] = stats(df[f"ret{h}"].dropna().values)
        res["max_caida20_promedio_pct"] = round(float(df["max_caida20"].mean() * 100), 2)
        res["max_alza20_promedio_pct"] = round(float(df["max_alza20"].mean() * 100), 2)
        res["tocaron_el_piso_pct"] = round(float(df["bajo_piso"].mean() * 100), 1)
        res["por_escenario"] = {k: stats(g["ret10"].dropna().values) for k, g in df.groupby("Escenario")}
    with open("backtest_resumen.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False)[:600])
