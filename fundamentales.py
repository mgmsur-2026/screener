"""Datos fundamentales automáticos (cambian poco): se actualizan una vez por semana.
Salida: fundamentales.csv (el screener diario lo une al resultado).
Reemplaza los campos manuales del Excel: Rubro, P/E, Fwd P/E, PEG, y suma beta, capitalización,
dividendos, márgenes, deuda, recomendación de analistas y precio objetivo."""
import csv, time
import yfinance as yf
import screener_yahoo as S

CAMPOS = ["Yahoo", "Sector", "Rubro", "P/E", "Fwd P/E", "PEG", "Beta", "Cap MUSD", "Div yield",
          "Margen %", "ROE %", "Crec ventas %", "Deuda/Patrim", "Recomendación", "Analistas",
          "Objetivo USD", "Upside %"]

def n(x, k=1, nd=2):
    return "" if x in (None, "") or isinstance(x, str) else round(float(x) * k, nd)

def fila(t):
    i = yf.Ticker(t).info
    px, obj = i.get("currentPrice") or i.get("regularMarketPrice"), i.get("targetMeanPrice")
    return {"Yahoo": t, "Sector": i.get("sector", ""), "Rubro": i.get("industry", ""),
            "P/E": n(i.get("trailingPE")), "Fwd P/E": n(i.get("forwardPE")),
            "PEG": n(i.get("trailingPegRatio", i.get("pegRatio"))), "Beta": n(i.get("beta")),
            "Cap MUSD": n(i.get("marketCap"), 1e-6, 0), "Div yield": n(i.get("dividendYield")),
            "Margen %": n(i.get("profitMargins"), 100, 1), "ROE %": n(i.get("returnOnEquity"), 100, 1),
            "Crec ventas %": n(i.get("revenueGrowth"), 100, 1), "Deuda/Patrim": n(i.get("debtToEquity")),
            "Recomendación": i.get("recommendationKey", ""), "Analistas": i.get("numberOfAnalystOpinions", ""),
            "Objetivo USD": n(obj), "Upside %": n((obj / px - 1) * 100 if obj and px else None, 1, 1)}

if __name__ == "__main__":
    filas = []
    for m in S.cargar():
        if m["País"] != "USA":
            continue
        try:
            filas.append(fila(m["Yahoo"]))
        except Exception as e:
            print(m["Yahoo"], e)
        time.sleep(0.4)
    with open("fundamentales.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        w.writeheader()
        w.writerows(filas)
    print(f"fundamentales.csv: {len(filas)} tickers")
