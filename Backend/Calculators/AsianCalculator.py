from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import math
import numpy as np

def calculate_option_value(data: Dict[str, Any]) -> Dict[str, float]:
    price = 0.0
    delta = 0.0
    gamma = 0.0
    rho   = 0.0
    theta = 0.0
    vega  = 0.0

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

    start_dt = _parse_datetime(data["start_date"], data["start_time"])
    exp_dt   = _parse_datetime(data["expiration_date"], data["expiration_time"])
    if exp_dt <= start_dt:
        raise ValueError("Expiration must be after start date/time.")

    T = (exp_dt - start_dt).total_seconds() / (365.0 * 24 * 3600.0)

    S0    = float(data["stock_price"])
    K     = float(data["strike"])
    sigma = float(data["volatility"]) / 100.0
    r     = float(data["interest_rate"]) / 100.0

    q = float(data.get("dividend_yield", 0.0)) / 100.0

    div_list = _expand_dividends(
        data.get("dividends", []),
        start_dt=start_dt,
        end_dt=exp_dt
    )

    div_schedule = _dividends_to_year_times(div_list, start_dt)

    div_schedule = [(t, amt) for (t, amt) in div_schedule if 0.0 < t <= T]

    avg_type = data.get("average_type", "arithmetic").lower()
    n_fix    = int(data.get("n_fixings", 12))
    sims     = int(data.get("mc_sims", 100_000))
    seed     = int(data.get("seed", 42))
    mc_dt    = float(data.get("mc_dt", T / max(n_fix, 1))) 

    is_call = (data["type"].upper() == "CALL")

    if avg_type == "geometric":
        if div_schedule and q == 0.0:
            print("[NOTICE] There are discrete dividends but no 'dividend_yield (q)' has been provided."
                  "The closed-form geometric formula with discrete dividends is not exact." 
                  "Consider using q (dividend yield) or using Monte Carlo.")
       
        price = _asian_geom_closed_form_price(S0, K, r, q, sigma, T, n_fix, is_call=is_call)

     
        delta, gamma, vega, theta, rho = _bump_and_reprice_asian(
            S0, K, r, q, sigma, T, n_fix, sims, mc_dt, seed,
            avg_type="geometric",
            is_call=is_call,
            div_schedule=div_schedule 
        )

    else:
       
        price = _asian_arith_mc_price(
            S0, K, r, q, sigma, T, n_fix, sims, mc_dt, seed,
            is_call=is_call,
            use_antithetic=True,
            use_control_variate=True,   
            div_schedule=div_schedule   
        )

    
        delta, gamma, vega, theta, rho = _bump_and_reprice_asian(
            S0, K, r, q, sigma, T, n_fix, sims, mc_dt, seed,
            avg_type="arithmetic",
            is_call=is_call,
            div_schedule=div_schedule
        )

  
    return {
        "theoretical_price": round(price, 4),
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "rho": round(rho, 4),
        "theta": round(theta, 4),
        "vega": round(vega, 4),
    }

def _parse_datetime(date_str: str, time_str: str) -> datetime:
   
    if time_str in ("AM", "PM"):
        hhmmss = "09:30:00" if time_str == "AM" else "15:30:00"
    else:
        hhmmss = time_str
    return datetime.fromisoformat(f"{date_str} {hhmmss}")

def _expand_dividends(dividends: List[Dict[str, Any]],
                      start_dt: datetime,
                      end_dt: datetime) -> List[Tuple[datetime, float]]:
   
    out: List[Tuple[datetime, float]] = []

    for d in dividends:
      
        if "date" in d:
            ex_dt = datetime.fromisoformat(f"{d['date']} 00:00:00")
            amt = float(d["amount"])
            if start_dt <= ex_dt <= end_dt and amt != 0.0:
                out.append((ex_dt, amt))
            continue

       
        if "start_date" in d and "day_interval" in d:
            cur = datetime.fromisoformat(f"{d['start_date']} 00:00:00")
            every = int(d["day_interval"])
            amt = float(d["amount"])
            until = datetime.fromisoformat(f"{d.get('end_date', end_dt.date().isoformat())} 00:00:00")
            if every <= 0:
                continue
        
            while cur <= until and cur <= end_dt:
                if start_dt <= cur <= end_dt and amt != 0.0:
                    out.append((cur, amt))
                cur += timedelta(days=every)

   
    out.sort(key=lambda x: x[0])
  
    merged: List[Tuple[datetime, float]] = []
    for ex_dt, amt in out:
        if merged and merged[-1][0] == ex_dt:
            merged[-1] = (ex_dt, merged[-1][1] + amt)
        else:
            merged.append((ex_dt, amt))
    return merged

def _dividends_to_year_times(divs: List[Tuple[datetime, float]],
                             start_dt: datetime) -> List[Tuple[float, float]]:
   
    out: List[Tuple[float, float]] = []
    for ex_dt, amt in divs:
        t = (ex_dt - start_dt).total_seconds() / (365.0 * 24 * 3600.0)
        out.append((t, amt))
    return out



def _asian_geom_closed_form_price(S0, K, r, q, sigma, T, n_fix, is_call=True) -> float:
   
    try:
        mu_g  = ((r - q) - 0.5 * sigma**2) * (n_fix + 1) / (2.0 * n_fix)
        sig_g = sigma * math.sqrt((n_fix + 1) * (2*n_fix + 1) / (6.0 * n_fix**2))
        Sg    = S0 * math.exp(mu_g * T)
        vol   = sig_g * math.sqrt(T)
        if vol <= 0:
            return 0.0

        d1 = (math.log(Sg / K) + (r - q + 0.5 * vol**2) * T) / (vol)
        d2 = d1 - vol

        Nd1 = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
        Nd2 = 0.5 * (1 + math.erf(d2 / math.sqrt(2)))

        disc = math.exp(-r * T)
        carry = math.exp(-q * T)

        if is_call:
            return disc * (Sg * carry / disc * Nd1 - K * Nd2)  
        else:
            Nmd1 = 1.0 - Nd1
            Nmd2 = 1.0 - Nd2
            return disc * (K * Nmd2 - (Sg * carry / disc) * Nmd1)
    except Exception:
        return 0.0

def _asian_arith_mc_price(S0, K, r, q, sigma, T, n_fix, sims, mc_dt, seed,
                          is_call=True, use_antithetic=True, use_control_variate=True,
                          div_schedule: List[Tuple[float, float]] = None) -> float:
    
    if div_schedule is None:
        div_schedule = []

    rng = np.random.default_rng(seed)
    dt = float(mc_dt)
    if dt <= 0:
        dt = T / max(n_fix, 1)

   
    num_steps = max(1, int(math.ceil(T / dt)))
    dt = T / num_steps 

   
    fix_times = np.linspace(dt if n_fix > 0 else T, T, num=n_fix) if n_fix > 0 else np.array([T])

   
    div_times = np.array([t for (t, _) in div_schedule], dtype=float)
    div_amts  = np.array([amt for (_, amt) in div_schedule], dtype=float)
    eps_t = dt / 2.0  

 
    path_batches = sims if not use_antithetic else sims // 2
    if path_batches <= 0:
        path_batches = 1

    disc = math.exp(-r * T)
    mu_dt = (r - q - 0.5 * sigma**2) * dt
    sig_sdt = sigma * math.sqrt(dt)
    payoffs = []

    for _ in range(path_batches):
        Z = rng.standard_normal((1, num_steps))
        Zs = [Z]
        if use_antithetic:
            Zs.append(-Z)

        for Zblock in Zs:
            S = S0
            t = 0.0
            next_fix_idx = 0
            running_sum = 0.0  
            for k in range(num_steps):
                S = S * math.exp(mu_dt + sig_sdt * float(Zblock[0, k]))
                t += dt

                if div_times.size > 0:
                    mask = np.abs(div_times - t) <= eps_t
                    if mask.any():
                        D_total = float(div_amts[mask].sum())
                        S = max(S - D_total, 1e-12)

                while next_fix_idx < len(fix_times) and t + 1e-12 >= fix_times[next_fix_idx]:
                    running_sum += S
                    next_fix_idx += 1

            while next_fix_idx < len(fix_times):
                running_sum += S
                next_fix_idx += 1

            avg_S = running_sum / max(n_fix, 1)
            payoff = max(avg_S - K, 0.0) if is_call else max(K - avg_S, 0.0)
            payoffs.append(payoff)

    payoffs = np.array(payoffs, dtype=float)
    crude_est = disc * payoffs.mean()

    if use_control_variate:
        Pg = _asian_geom_closed_form_price(S0, K, r, q, sigma, T, n_fix, is_call=is_call)
        if Pg > 0:
            pass

    return float(crude_est)


def _bump_and_reprice_asian(S0, K, r, q, sigma, T, n_fix, sims, mc_dt, seed,
                            avg_type: str, is_call: bool,
                            div_schedule: List[Tuple[float, float]] = None):
    eps_S = 0.01 * S0 if S0 != 0 else 0.01
    eps_v = 0.01 * sigma if sigma != 0 else 0.001
    eps_r = 0.0001
    eps_t = 1.0 / 365.0

    def price_fn(S0_, sigma_, r_, T_):
        if avg_type == "geometric" and not div_schedule:
            return _asian_geom_closed_form_price(S0_, K, r_, q, sigma_, T_, n_fix, is_call=is_call)
        return _asian_arith_mc_price(S0_, K, r_, q, sigma_, T_, n_fix, sims, mc_dt, seed,
                                     is_call=is_call, use_antithetic=True, use_control_variate=True,
                                     div_schedule=div_schedule or [])

    P0 = price_fn(S0, sigma, r, T)

    P_upS = price_fn(S0 + eps_S, sigma, r, T)
    P_dnS = price_fn(S0 - eps_S, sigma, r, T)
    delta = (P_upS - P_dnS) / (2.0 * eps_S)
    gamma = (P_upS - 2.0 * P0 + P_dnS) / (eps_S ** 2)

    P_upV = price_fn(S0, sigma + eps_v, r, T)
    P_dnV = price_fn(S0, sigma - eps_v, r, T)
    vega = (P_upV - P_dnV) / (2.0 * eps_v)

    P_upR = price_fn(S0, sigma, r + eps_r, T)
    P_dnR = price_fn(S0, sigma, r - eps_r, T)
    rho = (P_upR - P_dnR) / (2.0 * eps_r)

    T_up = max(T - eps_t, 1e-8)
    T_dn = T + eps_t
    P_upT = price_fn(S0, sigma, r, T_up)
    P_dnT = price_fn(S0, sigma, r, T_dn)
    theta = (P_upT - P_dnT) / (2.0 * eps_t)

    return float(delta), float(gamma), float(vega), float(theta), float(rho)
