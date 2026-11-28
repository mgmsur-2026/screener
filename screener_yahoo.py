"""Screener v3 (Yahoo Finance). Lee tickers.csv y parametros.json.
Por ticker: estado y etapa, 7 condiciones de entrada con semáforo (V verde, A amarillo, R rojo),
agotamiento de la baja (volumen), fuerza relativa vs S&P 500 (o MERVAL), precio en pesos (CCL/MEP),
y, para los candidatos, balances, dividendos e insiders.
Material informativo, no es recomendación de inversión."""
import csv, json, os, urllib.request
import numpy as np
import pandas as pd
import yfinance as yf

P = {"meses_baja_min": 9, "caida_min_pct": 20, "piso_min_dias": 5, "piso_max_dias": 120,
     "vol_usd_min": 5e6, "vol_ars_min": 2e8, "vol_x_min": 4, "fvg_min_pct": 1.0, "fvg_max_dias": 15,
     "sube_a_pct": 2.0, "sube_de_pct": 1.0, "ventana": 420, "agot_puntos_min": 3,
     "agot_vol_ratio_max": 0.8, "absorcion_vol_x": 1.5, "absorcion_rango_atr": 0.8,
     "compra_fuerte_vol_x": 3.0, "amarillo_factor": 0.75, "ruptura_cerca_pct": 2.0,
     "alerta_balance_dias": 7, "alerta_dividendo_dias": 7, "alerta_rsi": 75,
     "enriquecer_desde_cumple": 5, "dolar": "contadoconliqui", "dolar_manual": None}
if os.path.exists("parametros.json"):
    with open("parametros.json", encoding="utf-8") as _f:
        P.update(json.load(_f))

NOMBRES = ["Baja previa", "Piso", "Liquidez", "Agotamiento de baja", "Ruptura del máximo",
           "Volumen de compra", "FVG"]
SEM = ["S Baja", "S Piso", "S Liq", "S Agot", "S Ruptura", "S Vol", "S FVG", "S Esc"]
RATIOS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 25, 30, 32, 35, 40, 48, 50, 60, 70, 72,
          75, 80, 100, 120, 150, 200, 250, 300]

def tend(x, a=20, b=50):
    """Alcista / Bajista / Lateral según precio y medias de a y b períodos."""
    x = pd.Series(x).dropna()
    if len(x) < b + 1:
        return "—"
    m1, m2 = x.rolling(a).mean().iloc[-1], x.rolling(b).mean().iloc[-1]
    return "Alcista" if x.iloc[-1] > m1 > m2 else "Bajista" if x.iloc[-1] < m1 < m2 else "Lateral"

def sv(ok, amar):
    return "V" if ok else ("A" if amar else "R")

def rsi(c, n=14):
    d = pd.Series(c).diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return float(100 - 100 / (1 + up.iloc[-1] / dn.iloc[-1])) if dn.iloc[-1] else 100.0

def fvg_en(h, l, c, i, minimo=None):
    """FVG alcista cuya 3ª vela es i -> (base, techo) o None."""
    m = P["fvg_min_pct"] if minimo is None else minimo
    if i >= 2 and l[i] > h[i - 2] and (l[i] - h[i - 2]) / c[i] * 100 >= m:
        return h[i - 2], l[i]
    return None

def escenarios(o, h, l, c, i, base, techo):
    """1 a 7 ruedas tras el FVG (vela i) -> (letras A-F, descartado, nota)."""
    n = len(c)
    post = list(range(i + 1, min(i + 8, n)))
    descartado = bool((c[i + 1:] < base).any())      # un cuerpo cerró bajo el gap
    if not post:
        return [], descartado, "FVG recién formado: aún sin ruedas para evaluar"
    ref, letras = c[i], []
    if not [j for j in post if l[j] <= techo]:
        letras.append("A" if max(h[j] for j in post) >= ref * (1 + P["sube_a_pct"] / 100) else "C")
    if any(fvg_en(h, l, c, j) for j in range(i + 2, min(i + 8, n))):
        letras.append("B")
    k = next((j for j in post if h[j] >= ref * (1 + P["sube_de_pct"] / 100)), None)
    if k is not None:
        for j in post:
            if j > k and l[j] <= techo and c[j] >= base:
                lo, hi = min(o[j], c[j]), max(o[j], c[j])
                letras.append("F" if (lo >= base and hi <= techo and l[j] < base)
                              else "E" if lo <= techo else "D")
                break
    return sorted(set(letras)), descartado, ""

def agotamiento(o, h, l, c, v, piso, techo_i):
    """Puntos 0-4 de agotamiento de la baja: (1) merma de volumen en el piso, (2) absorción
    (volumen alto sin movimiento), (3) mínimo más alto, (4) más volumen comprador que vendedor."""
    n = len(c)
    cp = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - cp), abs(l - cp)))
    atr = pd.Series(tr).rolling(14).mean().values
    v20 = pd.Series(v).rolling(20).mean().shift(1).values
    baja = v[techo_i:piso + 1].mean() or 1.0
    ratio = v[max(0, piso - 10):min(n, piso + 11)].mean() / baja
    ab = sum(1 for j in range(max(21, piso - 30), min(n, piso + 16))
             if v20[j] and v[j] >= P["absorcion_vol_x"] * v20[j]
             and (h[j] - l[j]) <= P["absorcion_rango_atr"] * atr[j])
    p3 = (n - 1 - piso) >= 3 and l[piso + 1:].min() >= l[piso] * 1.005
    up = sum(v[j] for j in range(piso + 1, n) if c[j] > c[j - 1])
    dn = sum(v[j] for j in range(piso + 1, n) if c[j] < c[j - 1])
    rud = up / dn if dn else (2.0 if up else 0.0)
    pts = int(ratio <= P["agot_vol_ratio_max"]) + int(ab >= 1) + int(p3) + int(rud >= 1.1)
    return pts, ratio, ab, rud

def fuerza_rel(df, c, bench):
    if bench is None:
        return np.nan, np.nan
    try:
        b = bench.copy()
        b.index = b.index.tz_localize(None) if b.index.tz is not None else b.index
        ix = df.index.tz_localize(None) if df.index.tz is not None else df.index
        b = b.reindex(ix, method="ffill").values.astype(float)
        f = lambda k: (c[-1] / c[-k - 1] - b[-1] / b[-k - 1]) * 100 if len(c) > k else np.nan
        return f(63), f(126)
    except Exception:
        return np.nan, np.nan

def analizar(df, meta, bench=None, dolar=None):
    if df is None or df.empty or "Close" not in df:
        return None
    df = df.dropna(subset=["Close", "High", "Low", "Open"])
    if len(df) < 200:
        return None
    o, h, l, c = (df[k].astype(float).values for k in ("Open", "High", "Low", "Close"))
    v = df["Volume"].astype(float).fillna(0).values
    n = len(c)
    sma = lambda k: pd.Series(c).rolling(k).mean().values
    s50, s200 = sma(50), sma(200)
    es_ars = meta["Yahoo"].endswith(".BA")
    r = {"Precio": c[-1], "Cambio %": (c[-1] / c[-2] - 1) * 100,
         "Perf % 1 mes": (c[-1] / c[-22] - 1) * 100,
         "Volumen relativo": v[-1] / v[-21:-1].mean() if v[-21:-1].mean() else np.nan,
         "RSI": rsi(c)}
    mfv = ((c - l) - (h - c)) / np.where(h - l == 0, np.nan, h - l) * v
    r["CMF 20"] = float(np.nansum(mfv[-20:]) / v[-20:].sum()) if v[-20:].sum() else np.nan
    r["FR 3m %"], r["FR 6m %"] = fuerza_rel(df, c, bench)
    s200c = s200[-1]
    ret1a = c[-1] / c[-253] - 1 if n > 252 else 0.0
    r["Tend. Anual"] = "Alcista" if ret1a > .10 and c[-1] > s200c else "Bajista" if ret1a < -.10 and c[-1] < s200c else "Lateral"
    try:
        r["Tend. 1W"] = tend(df["Close"].astype(float).resample("W").last(), 5, 10)
    except Exception:
        r["Tend. 1W"] = "—"
    r["Tend. 1D"], r["Tend. 4H"] = tend(c, 20, 50), ""
    # ---- precio en pesos (comparable con el gráfico de BYMA) ----
    r["Precio ARS"], r["Ratio"] = (c[-1], "") if es_ars else ("", meta.get("Ratio", ""))
    if not es_ars and dolar:
        ratio = float(meta["Ratio"]) if meta.get("Ratio") else None
        px = float(meta.get("Precio Excel ARS") or 0)
        if ratio is None and px > 0:
            raw = c[-1] * dolar / px
            ratio = min(RATIOS, key=lambda x: abs(np.log(x / raw)))
        if ratio:
            r["Precio ARS"], r["Ratio"] = c[-1] * dolar / ratio, ratio
    # ---- estado y etapa ----
    w = slice(max(0, n - 252), n)
    pos = (c[-1] - l[w].min()) / (h[w].max() - l[w].min() + 1e-9)
    if c[-1] > s50[-1] > s200[-1]:
        est, etapa = "ALCISTA", "Inicio" if pos < .40 else "Medio" if pos < .75 else "Final"
    elif c[-1] < s50[-1] < s200[-1]:
        est, etapa = "BAJISTA", "Inicio" if pos > .60 else "Medio" if pos > .25 else "Final"
    else:
        est, dias = "ACUMULACION", 10
        for k in range(10, 121):
            seg = c[-k:]
            if (seg.max() - seg.min()) / seg.mean() <= .20:
                dias = k
        etapa = "Inicio" if dias < 20 else "Medio" if dias < 60 else "Final"
    r.update({"Estado calc": est, "Etapa": etapa, "Pos 52s %": pos * 100})
    # ---- condiciones ----
    af = P["amarillo_factor"]
    ini = max(0, n - int(P["ventana"]))
    piso = ini + int(np.argmin(l[ini:]))
    techo_i = ini + int(np.argmax(h[ini:piso + 1]))
    meses = (piso - techo_i) / 21
    caida = (h[techo_i] - l[piso]) / h[techo_i] * 100
    dp = n - 1 - piso
    M, C = P["meses_baja_min"], P["caida_min_pct"]
    c1 = meses >= M and caida >= C
    c2 = P["piso_min_dias"] <= dp <= P["piso_max_dias"]
    usd = c * v
    vmin = P["vol_ars_min"] if es_ars else P["vol_usd_min"]
    m3, m6 = usd[-63:].mean(), usd[-126:].mean()
    c3 = m3 >= vmin and m6 >= vmin
    pts, vratio, ab, rud = agotamiento(o, h, l, c, v, piso, techo_i)
    c7 = pts >= P["agot_puntos_min"]
    nivel, bos = np.nan, None
    if piso - ini >= 5:      # hace falta historia previa al piso para hallar el último máximo
        nivel = h[max(ini, piso - 20):piso].max()
        bos = next((j for j in range(piso + 1, n) if c[j] > nivel), None)
    c4 = bos is not None
    gap = None
    if c4:
        for i in range(max(bos, n - 1 - int(P["fvg_max_dias"])), n):
            g = fvg_en(h, l, c, i)
            if g:
                gap = (i, *g)
                break
    c6 = gap is not None
    fin = gap[0] if c6 else n - 1
    mx = 0.0
    for j in range(max(21, fin - 5), fin + 1):
        prom = v[j - 20:j].mean()
        if prom and c[j] > o[j]:
            mx = max(mx, v[j] / prom)
    c5 = mx >= P["vol_x_min"]
    chico = any(fvg_en(h, l, c, i, P["fvg_min_pct"] * af) for i in
                range(max(piso + 2, n - 1 - int(P["fvg_max_dias"])), n))
    letras, desc, nota = [], False, ""
    if c6:
        letras, desc, nota = escenarios(o, h, l, c, *gap)
    conds = [c1, c2, c3, c7, c4, c5, c6]
    r["S Baja"] = sv(c1, meses >= M * af and caida >= C * af)
    r["S Piso"] = sv(c2, 1 <= dp < P["piso_min_dias"] or P["piso_max_dias"] < dp <= P["piso_max_dias"] * 1.5)
    r["S Liq"] = sv(c3, max(m3, m6) >= vmin * af)
    r["S Agot"] = sv(c7, pts == P["agot_puntos_min"] - 1)
    r["S Ruptura"] = sv(c4, not c4 and not np.isnan(nivel) and c[-1] >= nivel * (1 - P["ruptura_cerca_pct"] / 100))
    r["S Vol"] = sv(c5, mx >= P["vol_x_min"] * af)
    r["S FVG"] = sv(c6, chico)
    r["S Esc"] = "R" if (not c6 or desc) else ("V" if letras else "A")
    todas = all(conds)
    if todas and desc:
        senal = "CUMPLIÓ TODO · DESCARTADO"
    elif todas and letras and c[-1] >= gap[1] and c[-1] > nivel:
        senal = "POTENCIAL ALTO"
    elif todas or sum(conds) >= 4:
        senal = "EN SEGUIMIENTO"
    else:
        senal = "—"
    fuertes = sum(1 for j in range(max(21, n - 10), n) if v[j] >= P["compra_fuerte_vol_x"] * v[j - 20:j].mean()
                  and c[j] > o[j] and (c[j] - l[j]) >= 0.7 * (h[j] - l[j] + 1e-9))
    alertas = []
    if r["RSI"] > P["alerta_rsi"]:
        alertas.append(f"RSI alto ({r['RSI']:.0f})")
    if not np.isnan(r["FR 3m %"]) and r["FR 3m %"] < 0:
        alertas.append("Fuerza relativa 3m negativa")
    if not np.isnan(r["CMF 20"]) and r["CMF 20"] < 0:
        alertas.append("Flujo de dinero (CMF) negativo")
    r.update({"Señal": senal, "Escenario": "".join(letras) if c6 else "", "Cumple": f"{sum(conds)}/7",
              "Faltan": "; ".join(f for f, ok in zip(NOMBRES, conds) if not ok),
              "Baja meses": meses, "Caída %": caida, "Días desde piso": dp,
              "Riesgo piso %": (c[-1] - l[piso]) / c[-1] * 100,
              "Agotamiento": f"{pts}/4", "Vol zona piso / baja": vratio, "Días absorción": ab,
              "Vol compra / venta": rud, "Vol rel máx": mx, "Compra fuerte 10r": fuertes,
              "FVG %": (gap[2] - gap[1]) / c[gap[0]] * 100 if c6 else 0.0,
              "FVG días": n - 1 - gap[0] if c6 else "", "Nota": nota,
              "Alertas": "; ".join(alertas), "Balance": "", "Dividendo": "", "Insiders 60d": ""})
    return {**meta, **r}

def cerrar(row):
    """Semáforo general: rojo si falta algo; amarillo si todo cumple pero hay alertas."""
    s = [row[k] for k in SEM]
    row["Semáforo"] = "R" if "R" in s else ("A" if "A" in s or row["Alertas"] else "V")

def enriquecer(row):
    """Balances, dividendos e insiders (solo candidatos, para no saturar a Yahoo)."""
    if row["País"] != "USA":
        return
    al, hoy = [], pd.Timestamp.today().normalize()
    try:
        h1 = yf.download(row["Yahoo"], period="60d", interval="1h", progress=False, auto_adjust=False)
        if isinstance(h1.columns, pd.MultiIndex):
            h1.columns = h1.columns.get_level_values(0)
        row["Tend. 4H"] = tend(h1["Close"].astype(float).resample("4h").last(), 20, 50)
    except Exception:
        pass
    try:
        tk = yf.Ticker(row["Yahoo"])
        cal = tk.calendar or {}
        for e in (cal.get("Earnings Date") or [])[:1]:
            d = (pd.Timestamp(e) - hoy).days
            row["Balance"] = str(pd.Timestamp(e).date())
            if 0 <= d <= P["alerta_balance_dias"]:
                al.append(f"Balance en {d} días")
        ex = cal.get("Ex-Dividend Date")
        div = tk.dividends
        if len(div):
            ult = div.index[-1].tz_localize(None) if div.index[-1].tzinfo else div.index[-1]
            row["Dividendo"] = f"{div.iloc[-1]:.2f} USD (último {ult.date()}; {int((div.index > hoy - pd.Timedelta(days=365)).sum())} pagos/año)"
        if ex is not None:
            d = (pd.Timestamp(ex) - hoy).days
            row["Dividendo"] = (row["Dividendo"] + f" · ex-div {pd.Timestamp(ex).date()}").strip(" ·")
            if 0 <= d <= P["alerta_dividendo_dias"]:
                al.append(f"Ex-dividendo en {d} días")
        ins = tk.insider_transactions
        if ins is not None and len(ins):
            fch = pd.to_datetime(ins.get("Start Date"), errors="coerce")
            txt = ins.get("Text", ins.get("Transaction", pd.Series("", index=ins.index))).astype(str)
            row["Insiders 60d"] = int(((fch > hoy - pd.Timedelta(days=60)) & txt.str.contains("Purchase|Buy", case=False)).sum())
    except Exception as e:
        print(f"{row['Yahoo']}: sin datos de balance/dividendo ({e})")
    row["Alertas"] = "; ".join(x for x in [row["Alertas"]] + al if x)

def dolar():
    if P.get("dolar_manual"):
        return float(P["dolar_manual"])
    for _ in range(2):
        try:
            with urllib.request.urlopen(f"https://dolarapi.com/v1/dolares/{P['dolar']}", timeout=15) as r:
                return float(json.load(r)["venta"])
        except Exception as e:
            print("dólar:", e)
    return None

def cargar():
    if os.path.exists("tickers.csv"):
        with open("tickers.csv", encoding="utf-8") as f:
            lista = list(csv.DictReader(f))
        fund = {}
        if os.path.exists("fundamentales.csv"):
            with open("fundamentales.csv", encoding="utf-8") as f:
                fund = {x["Yahoo"]: x for x in csv.DictReader(f)}
        for m in lista:
            for k in ("Anual", "1W", "1D", "4H"):          # tus notas manuales del Excel
                m["Nota " + k] = m.pop("Tend. " + k, "")
            for k, v in fund.get(m["Yahoo"], {}).items():  # datos automáticos (semanales)
                if k != "Yahoo" and v not in ("", None):
                    m[k] = v
        return lista
    with open("tickers.txt") as f:
        return [{"Símbolo": s.strip(), "Yahoo": s.strip(), "País": "ARG" if s.strip().endswith(".BA") else "USA"}
                for s in f if s.strip()]

def bajar(lista, periodo="2y"):
    """Descarga y devuelve {ticker: DataFrame}; funciona con uno o varios tickers
    (las versiones nuevas de yfinance devuelven columnas de dos niveles incluso con uno)."""
    datos = yf.download(lista, period=periodo, interval="1d", group_by="ticker",
                        auto_adjust=False, progress=False, threads=True)
    out = {}
    for t in lista:
        d = datos
        if isinstance(d.columns, pd.MultiIndex):
            d = d[t] if t in d.columns.get_level_values(0) else pd.DataFrame()
        out[t] = d
    return out

def vol_byma(filas):
    """Alerta de volumen en BYMA (CEDEAR): volumen diario > Vol. Alerta de tu Excel
    (promedio de 12 meses x 4). Usa el volumen de la especie .BA en Yahoo."""
    for i in range(0, len(filas), 25):
        lote = filas[i:i + 25]
        ts = [r["Yahoo"] if r["Yahoo"].endswith(".BA") else r["Yahoo"] + ".BA" for r in lote]
        try:
            datos = bajar(ts, "1y")
        except Exception as e:
            print("volumen BYMA:", e)
            continue
        for r, t in zip(lote, ts):
            try:
                v = datos[t]["Volume"].dropna().astype(float)
                if len(v) < 60:
                    continue
                hoy, prom = v.iloc[-1], v.iloc[-253:-1].mean()
                try:
                    alerta = float(r.get("Vol. Alerta") or 0)
                except ValueError:
                    alerta = 0.0
                calc = prom * 4
                umbral = alerta or calc
                n5 = int((v.iloc[-5:] > umbral).sum())
                r.update({"Vol BYMA hoy": hoy, "Vol BYMA prom 12m": prom, "Vol Alerta calc": calc,
                          "Dif Vol Alerta %": (calc / alerta - 1) * 100 if alerta else "",
                          "Días alerta 5r": n5, "Alerta volumen": "SÍ" if hoy > umbral else ""})
                if hoy > umbral:
                    msg = f"Volumen BYMA {hoy:,.0f} supera tu alerta ({umbral:,.0f})"
                elif n5:
                    msg = f"Volumen BYMA superó tu alerta en {n5} de las últimas 5 ruedas"
                else:
                    msg = ""
                r["Alertas"] = "; ".join(x for x in [r["Alertas"], msg] if x)
            except Exception as e:
                print(f"{t}: sin volumen BYMA ({e})")

if __name__ == "__main__":
    lista, filas, usd = cargar(), [], dolar()
    print(f"Dólar {P['dolar']}: {usd}")
    ref = bajar(["SPY", "^MERV"])
    bench = {"USA": ref["SPY"]["Close"].dropna(), "ARG": ref["^MERV"]["Close"].dropna()}
    for i in range(0, len(lista), 25):
        lote = lista[i:i + 25]
        datos = bajar([m["Yahoo"] for m in lote])
        for m in lote:
            try:
                res = analizar(datos[m["Yahoo"]], m, bench.get(m["País"]), usd)
                filas.append(res) if res else print(f"{m['Yahoo']}: pocos datos")
            except Exception as e:
                print(f"{m['Yahoo']}: {e}")
        print(f"{min(i + 25, len(lista))}/{len(lista)}")
    vol_byma(filas)
    for r in filas:
        if int(r["Cumple"][0]) >= P["enriquecer_desde_cumple"]:
            enriquecer(r)
        cerrar(r)
    pd.DataFrame(filas).to_csv("resultado_screener.csv", index=False)
    print(f"Listo: {len(filas)} especies en resultado_screener.csv")
