# ---------------------------------------------------------
# Description:
#   Calcula el valor teórico y griegas de una opción ASIÁTICA
#   a partir de un diccionario de entrada `data` que replica
#   los inputs de la calculadora de CBOE, más campos propios
#   de asiáticas (promedio y parámetros de Monte Carlo).
#
#   NOTA IMPORTANTE:
#   - El proyecto exige leer SIEMPRE parámetros desde fichero.
#     Este módulo asume que ya llega un `data` cargado/validado.
#     La lectura/validación hazla en otro módulo.
#
#   - Aquí mostramos estructura y explicamos CADA paso.
#     Implementamos MC con dividendos discretos y dejamos
#     la geométrica cerrada como función a completar si no
#     usas q (dividend yield).
# ---------------------------------------------------------

from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import math
import numpy as np

# =========================================================
# API principal
# =========================================================

def calculate_option_value(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calcula precio teórico y griegas para una opción asiática.
    `data` debe contener al menos (strings o números):
      - type: "CALL" | "PUT"
      - exercise_style: "American" | "European"  (asiáticas suelen ser European)
      - start_date: "YYYY-MM-DD"
      - start_time: "HH:MM:SS" | "AM"/"PM"
      - expiration_date: "YYYY-MM-DD"
      - expiration_time: "AM" | "PM" | "HH:MM:SS"
      - strike: float
      - stock_price: float
      - volatility: float  (en %, ej 20 → 20.0; internamente se pasa a 0.20)
      - interest_rate: float (en %, ej 1.5 → 1.5; internamente 0.015)
      - dividends: lista (ver formatos admitidos más abajo)

    Campos opcionales específicos de ASIÁTICAS:
      - average_type: "arithmetic" | "geometric"  (default: "arithmetic")
      - n_fixings: int  (nº de observaciones del promedio; default: 12)
      - mc_sims: int  (nº de simulaciones Monte Carlo; default: 100_000)
      - mc_dt: float  (paso temporal en años; default: T/n_fixings)
      - seed: int  (semilla RNG para reproducibilidad; default: 42)
      - dividend_yield: float (%)  → si deseas usar q en vez de dividendos discretos

    Formatos de dividendos en data["dividends"]:
      1) Discreto: {"date":"YYYY-MM-DD","amount":float}
      2) Recurrente: {"start_date":"YYYY-MM-DD","day_interval":int,"amount":float,
                      "end_date":"YYYY-MM-DD"(opcional)}
         → Se expande automáticamente a una serie de ex-dates.
    """

    # -----------------------------
    # 1) Inicializamos outputs con 0
    #    (estructura válida incluso si algo falla).
    # -----------------------------
    price = 0.0
    delta = 0.0
    gamma = 0.0
    rho   = 0.0
    theta = 0.0
    vega  = 0.0

    # -----------------------------
    # 2) Log de parámetros recibidos (debug/trazabilidad).
    # -----------------------------
    print(f"Option type: {data['type']}")
    print(f"Exercise style: {data['exercise_style']}")
    print(f"Start date: {data['start_date']}")
    print(f"Start time: {data['start_time']}")
    print(f"Expiration date: {data['expiration_date']}")
    print(f"Expiration time: {data['expiration_time']}")
    print(f"Strike: {data['strike']}")
    print(f"Stock price: {data['stock_price']}")
    print(f"Volatility: {data['volatility']}")
    print(f"Interest rate: {data['interest_rate']}")
    print(f"Dividends: {data.get('dividends', [])}")

    # -----------------------------
    # 3) Parseo y normalización de inputs
    #    - Fechas/horas → datetime
    #    - Tasas % → decimales
    #    - T = (exp - start) en años ACT/365
    #    - Dividendos: expandir y llevar a tiempos en años desde start
    # -----------------------------
    start_dt = _parse_datetime(data["start_date"], data["start_time"])
    exp_dt   = _parse_datetime(data["expiration_date"], data["expiration_time"])
    if exp_dt <= start_dt:
        raise ValueError("Expiration must be after start date/time.")

    # Tiempo a vencimiento en años (ACT/365):
    T = (exp_dt - start_dt).total_seconds() / (365.0 * 24 * 3600.0)

    # Números clave:
    S0    = float(data["stock_price"])
    K     = float(data["strike"])
    sigma = float(data["volatility"]) / 100.0
    r     = float(data["interest_rate"]) / 100.0

    # Dividend yield opcional (para modelos BS-like):
    q = float(data.get("dividend_yield", 0.0)) / 100.0

    # Expandimos dividendos discretos del estilo CBOE y los convertimos a (t_en_años, amount)
    # - Si hay schedule recurrente (day_interval), se expande entre start_date y expiration_date.
    div_list = _expand_dividends(
        data.get("dividends", []),
        start_dt=start_dt,
        end_dt=exp_dt
    )
    # Convertimos fechas a tiempos en años desde start:
    div_schedule = _dividends_to_year_times(div_list, start_dt)
    # Nota: div_schedule es List[Tuple[float, float]] con tiempos en (0,T]; filtramos fuera-de-rango.
    div_schedule = [(t, amt) for (t, amt) in div_schedule if 0.0 < t <= T]

    # -----------------------------
    # 4) Parámetros de asiática (defaults razonables)
    # -----------------------------
    avg_type = data.get("average_type", "arithmetic").lower()
    n_fix    = int(data.get("n_fixings", 12))
    sims     = int(data.get("mc_sims", 100_000))
    seed     = int(data.get("seed", 42))
    mc_dt    = float(data.get("mc_dt", T / max(n_fix, 1)))  # por defecto: un paso por fixing

    # -----------------------------
    # 5) Selección de método según tipo de promedio
    #    - Geometric: fórmula cerrada (si usas q). Con dividendos discretos puros
    #                 no hay fórmula exacta: avisamos y sugerimos q.
    #    - Arithmetic: Monte Carlo con dividendos discretos en la trayectoria.
    # -----------------------------
    is_call = (data["type"].upper() == "CALL")

    if avg_type == "geometric":
        if div_schedule and q == 0.0:
            print("[AVISO] Hay dividendos discretos pero no se ha proporcionado 'dividend_yield (q)'. "
                  "La fórmula cerrada geométrica con discretos no es exacta. "
                  "Considera usar q (rendimiento por dividendos) o usar Monte Carlo.")
        # Implementación cerrada usando q si está disponible:
        price = _asian_geom_closed_form_price(S0, K, r, q, sigma, T, n_fix, is_call=is_call)

        # Griegas (recomendación): calcular por bump & reprice
        delta, gamma, vega, theta, rho = _bump_and_reprice_asian(
            S0, K, r, q, sigma, T, n_fix, sims, mc_dt, seed,
            avg_type="geometric",
            is_call=is_call,
            div_schedule=div_schedule  # se ignora en la cerrada; se usa en bumps si caen a MC
        )

    else:
        # avg_type == "arithmetic" (por defecto): Monte Carlo con dividendos discretos
        price = _asian_arith_mc_price(
            S0, K, r, q, sigma, T, n_fix, sims, mc_dt, seed,
            is_call=is_call,
            use_antithetic=True,
            use_control_variate=True,   # control variate: geométrica cerrada con q
            div_schedule=div_schedule   # aplicar caídas de dividendo en la trayectoria
        )

        # Griegas por bump & reprice (mismo motor MC, CRN recomendadas):
        delta, gamma, vega, theta, rho = _bump_and_reprice_asian(
            S0, K, r, q, sigma, T, n_fix, sims, mc_dt, seed,
            avg_type="arithmetic",
            is_call=is_call,
            div_schedule=div_schedule
        )

    # -----------------------------
    # 6) Redondeo y retorno
    # -----------------------------
    return {
        "theoretical_price": round(price, 4),
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "rho": round(rho, 4),
        "theta": round(theta, 4),
        "vega": round(vega, 4),
    }


# =========================================================
# Helpers: fechas / dividendos
# =========================================================

def _parse_datetime(date_str: str, time_str: str) -> datetime:
    """
    Convierte 'YYYY-MM-DD' + ('HH:MM:SS' | 'AM'|'PM') en datetime.
    - Si time_str es 'AM' o 'PM', usamos horas por defecto (configurables):
        AM → 09:30:00, PM → 15:30:00
      (elige valores coherentes con tu proyecto/mercado).
    """
    if time_str in ("AM", "PM"):
        hhmmss = "09:30:00" if time_str == "AM" else "15:30:00"
    else:
        hhmmss = time_str
    return datetime.fromisoformat(f"{date_str} {hhmmss}")

def _expand_dividends(dividends: List[Dict[str, Any]],
                      start_dt: datetime,
                      end_dt: datetime) -> List[Tuple[datetime, float]]:
    """
    Normaliza la entrada de dividendos a una lista de tuplas (ex_date_datetime, amount).
    Acepta:
      - {"date":"YYYY-MM-DD","amount":float}
      - {"start_date":"YYYY-MM-DD","day_interval":int,"amount":float,"end_date":"YYYY-MM-DD"(opt)}
    Reglas:
      - Se ignoran dividendos fuera del rango [start_dt, end_dt].
      - Si 'end_date' falta en recurrentes, se usa end_dt.
    """
    out: List[Tuple[datetime, float]] = []

    for d in dividends:
        # Caso 1: fecha única
        if "date" in d:
            ex_dt = datetime.fromisoformat(f"{d['date']} 00:00:00")
            amt = float(d["amount"])
            if start_dt <= ex_dt <= end_dt and amt != 0.0:
                out.append((ex_dt, amt))
            continue

        # Caso 2: recurrente por intervalo de días
        if "start_date" in d and "day_interval" in d:
            cur = datetime.fromisoformat(f"{d['start_date']} 00:00:00")
            every = int(d["day_interval"])
            amt = float(d["amount"])
            until = datetime.fromisoformat(f"{d.get('end_date', end_dt.date().isoformat())} 00:00:00")
            if every <= 0:
                continue
            # Avanzamos en saltos de 'every' días
            while cur <= until and cur <= end_dt:
                if start_dt <= cur <= end_dt and amt != 0.0:
                    out.append((cur, amt))
                cur += timedelta(days=every)

    # Orden cronológico y sin duplicados exactos
    out.sort(key=lambda x: x[0])
    # Opcional: merge de duplicados en misma fecha sumando amounts
    merged: List[Tuple[datetime, float]] = []
    for ex_dt, amt in out:
        if merged and merged[-1][0] == ex_dt:
            merged[-1] = (ex_dt, merged[-1][1] + amt)
        else:
            merged.append((ex_dt, amt))
    return merged

def _dividends_to_year_times(divs: List[Tuple[datetime, float]],
                             start_dt: datetime) -> List[Tuple[float, float]]:
    """
    Convierte [(ex_dt, amount)] → [(t_en_años, amount)] con convención ACT/365.
    t = (ex_dt - start_dt) / 365 días.
    """
    out: List[Tuple[float, float]] = []
    for ex_dt, amt in divs:
        t = (ex_dt - start_dt).total_seconds() / (365.0 * 24 * 3600.0)
        out.append((t, amt))
    return out


# =========================================================
# Pricing: fórmulas y Monte Carlo
# =========================================================

def _asian_geom_closed_form_price(S0, K, r, q, sigma, T, n_fix, is_call=True) -> float:
    """
    Precio cerrado de una ASIÁTICA GEOMÉTRICA (fechas equiespaciadas).
    IMPORTANTE:
      - Con dividendos DISCRETOS no hay fórmula exacta. Aquí modelamos dividendos
        mediante 'q' (dividend_yield). Si no usas q y tienes discretos, considera MC.

    Esquema (completar si quieres 100% analítico):
      1) Ajustes del promedio geométrico (discreto, equiespaciado):
         mu_g  = ( (r - q) - 0.5*sigma^2 ) * (n_fix + 1) / (2*n_fix)
         sig_g = sigma * sqrt( (n_fix + 1)*(2*n_fix + 1) / (6*n_fix**2) )
      2) Sg efectivo: Sg = S0 * exp(mu_g * T)
      3) Black-Scholes “modificado” con vol = sig_g*sqrt(T), spot=Sg, carry=(r - q).
    """
    # Placeholder mínimo (devuelve 0.0 si no implementas la cerrada):
    try:
        mu_g  = ((r - q) - 0.5 * sigma**2) * (n_fix + 1) / (2.0 * n_fix)
        sig_g = sigma * math.sqrt((n_fix + 1) * (2*n_fix + 1) / (6.0 * n_fix**2))
        Sg    = S0 * math.exp(mu_g * T)
        vol   = sig_g * math.sqrt(T)
        if vol <= 0:
            return 0.0

        # d1/d2 con carry q:
        # Usamos la forma BS con dividendo continuo q (cost of carry) para Sg
        # d1 = [ln(Sg/K) + (r - q + 0.5*vol^2)T] / (vol)
        # d2 = d1 - vol
        d1 = (math.log(Sg / K) + (r - q + 0.5 * vol**2) * T) / (vol)
        d2 = d1 - vol

        Nd1 = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
        Nd2 = 0.5 * (1 + math.erf(d2 / math.sqrt(2)))

        disc = math.exp(-r * T)
        carry = math.exp(-q * T)

        if is_call:
            return disc * (Sg * carry / disc * Nd1 - K * Nd2)  # algebra simple: e^{-qT} Sg N(d1) - K e^{-rT} N(d2)
        else:
            Nmd1 = 1.0 - Nd1
            Nmd2 = 1.0 - Nd2
            return disc * (K * Nmd2 - (Sg * carry / disc) * Nmd1)
    except Exception:
        return 0.0

def _asian_arith_mc_price(S0, K, r, q, sigma, T, n_fix, sims, mc_dt, seed,
                          is_call=True, use_antithetic=True, use_control_variate=True,
                          div_schedule: List[Tuple[float, float]] = None) -> float:
    """
    Monte Carlo para ASIÁTICA ARITMÉTICA con dividendos discretos en la trayectoria.

    Pasos:
      A) Construir tiempos de fixing equiespaciados: t_i = i*(T/n_fix), i=1..n_fix
      B) Simular bajo RN: d ln S = (r - q - 0.5σ^2) dt + σ dW
      C) En cada paso, si hay ex-div dentro del intervalo (con tolerancia), aplicar S ← max(S - D, ε)
      D) Promedio aritmético de S(t_i) y payoff
      E) Descuento y media; opcional control variate con geométrica cerrada (vía q)

    Notas:
      - Tolerancia para alinear ex-dates a la malla: eps_t = mc_dt/2
      - Evitamos valores negativos tras dividendos con un piso ε pequeño.
    """
    if div_schedule is None:
        div_schedule = []

    rng = np.random.default_rng(seed)
    dt = float(mc_dt)
    if dt <= 0:
        dt = T / max(n_fix, 1)

    # Mallado fino para simular (puede ser más fino que los fixings)
    num_steps = max(1, int(math.ceil(T / dt)))
    dt = T / num_steps  # re-ajustamos dt exacto para cubrir [0,T] en num_steps iguales

    # Tiempos de fixing (para el promedio)
    fix_times = np.linspace(dt if n_fix > 0 else T, T, num=n_fix) if n_fix > 0 else np.array([T])

    # Preparamos dividendos: vector de tiempos
    div_times = np.array([t for (t, _) in div_schedule], dtype=float)
    div_amts  = np.array([amt for (_, amt) in div_schedule], dtype=float)
    eps_t = dt / 2.0  # tolerancia para "caer" en la malla

    # Antitéticos: simulamos la mitad y duplicamos con Z→-Z si procede
    path_batches = sims if not use_antithetic else sims // 2
    if path_batches <= 0:
        path_batches = 1

    disc = math.exp(-r * T)
    mu_dt = (r - q - 0.5 * sigma**2) * dt
    sig_sdt = sigma * math.sqrt(dt)

    payoffs = []

    for _ in range(path_batches):
        # Generamos shocks (num_paths x num_steps)
        Z = rng.standard_normal((1, num_steps))
        Zs = [Z]
        if use_antithetic:
            Zs.append(-Z)

        for Zblock in Zs:
            # Simulación de una trayectoria (vectorizada en 1 path para claridad)
            S = S0
            t = 0.0
            next_fix_idx = 0
            running_sum = 0.0  # acumulador para promedio aritmético
            # Iteramos pasos de tiempo
            for k in range(num_steps):
                # Evolución exacta en log:
                S = S * math.exp(mu_dt + sig_sdt * float(Zblock[0, k]))
                t += dt

                # Aplicar dividendos si hay ex-date dentro del paso (|t_div - t| <= eps_t)
                if div_times.size > 0:
                    # Encontrar índices de divs “cayendo” cerca de t actual
                    mask = np.abs(div_times - t) <= eps_t
                    if mask.any():
                        D_total = float(div_amts[mask].sum())
                        # Caída de efectivo en el precio (floor pequeño para evitar negativos)
                        S = max(S - D_total, 1e-12)

                # Capturar fixing si hemos alcanzado o pasado su tiempo
                while next_fix_idx < len(fix_times) and t + 1e-12 >= fix_times[next_fix_idx]:
                    running_sum += S
                    next_fix_idx += 1

            # Si por redondeo faltó algún fixing, completamos:
            while next_fix_idx < len(fix_times):
                running_sum += S
                next_fix_idx += 1

            # Promedio aritmético y payoff
            avg_S = running_sum / max(n_fix, 1)
            payoff = max(avg_S - K, 0.0) if is_call else max(K - avg_S, 0.0)
            payoffs.append(payoff)

    payoffs = np.array(payoffs, dtype=float)
    crude_est = disc * payoffs.mean()

    # Control variate con asiática geométrica (usando q):
    if use_control_variate:
        # Geométrica simulada (misma trayectoria no la guardamos; aproximamos con precio cerrado)
        Pg = _asian_geom_closed_form_price(S0, K, r, q, sigma, T, n_fix, is_call=is_call)
        # Estimación del payoff geométrico “simulado” sin recalcular trayectorias:
        # Para una CV más “real”, habría que calcular también avg geom sobre la marcha.
        # Aquí usamos una corrección simple: mantener crude_est si Pg=0.
        if Pg > 0:
            # Técnica práctica: sin Y_sim concreto, aplicamos un ajuste leve (placeholder):
            # En un setup completo, calcular Yd_sim y ajustar: est = Xd - c*(Yd - Pg).
            # Dejamos crude_est si no implementas Y_sim.
            pass

    return float(crude_est)


def _bump_and_reprice_asian(S0, K, r, q, sigma, T, n_fix, sims, mc_dt, seed,
                            avg_type: str, is_call: bool,
                            div_schedule: List[Tuple[float, float]] = None):
    """
    Griegas por 'bump & reprice'.
      - Reutiliza el MISMO seed (Common Random Numbers) para reducir ruido.
      - Para geométrica: puedes llamar a la cerrada; si hay dividendos discretos,
        forzamos MC para los bumps (consistencia del tratamiento de dividendos).

      Fórmulas:
        Delta ≈ (P(S0+ε) - P(S0-ε)) / (2ε)
        Gamma ≈ (P(S0+ε) - 2P(S0) + P(S0-ε)) / ε^2
        Vega  ≈ (P(σ+η) - P(σ-η)) / (2η)
        Rho   ≈ (P(r+ρb) - P(r-ρb)) / (2ρb)
        Theta ≈ (P(T-τ) - P(T+τ)) / (2τ)
    """
    # Tamaños de bump (puedes parametrizarlos)
    eps_S = 0.01 * S0 if S0 != 0 else 0.01
    eps_v = 0.01 * sigma if sigma != 0 else 0.001
    eps_r = 0.0001
    eps_t = 1.0 / 365.0  # un día

    # Función de precio según avg_type y dividendos
    def price_fn(S0_, sigma_, r_, T_):
        if avg_type == "geometric" and not div_schedule:
            return _asian_geom_closed_form_price(S0_, K, r_, q, sigma_, T_, n_fix, is_call=is_call)
        # Para asegurar consistencia cuando hay dividendos discretos, usamos MC:
        return _asian_arith_mc_price(S0_, K, r_, q, sigma_, T_, n_fix, sims, mc_dt, seed,
                                     is_call=is_call, use_antithetic=True, use_control_variate=True,
                                     div_schedule=div_schedule or [])

    # Precio base
    P0 = price_fn(S0, sigma, r, T)

    # Delta & Gamma (bump en S)
    P_upS = price_fn(S0 + eps_S, sigma, r, T)
    P_dnS = price_fn(S0 - eps_S, sigma, r, T)
    delta = (P_upS - P_dnS) / (2.0 * eps_S)
    gamma = (P_upS - 2.0 * P0 + P_dnS) / (eps_S ** 2)

    # Vega (bump en sigma)
    P_upV = price_fn(S0, sigma + eps_v, r, T)
    P_dnV = price_fn(S0, sigma - eps_v, r, T)
    vega = (P_upV - P_dnV) / (2.0 * eps_v)

    # Rho (bump en r)
    P_upR = price_fn(S0, sigma, r + eps_r, T)
    P_dnR = price_fn(S0, sigma, r - eps_r, T)
    rho = (P_upR - P_dnR) / (2.0 * eps_r)

    # Theta (bump en T) – ojo con T-τ no negativo
    T_up = max(T - eps_t, 1e-8)
    T_dn = T + eps_t
    P_upT = price_fn(S0, sigma, r, T_up)
    P_dnT = price_fn(S0, sigma, r, T_dn)
    theta = (P_upT - P_dnT) / (2.0 * eps_t)

    return float(delta), float(gamma), float(vega), float(theta), float(rho)

