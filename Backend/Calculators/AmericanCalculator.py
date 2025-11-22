"""
Minimal American Option Pricer using Longstaff-Schwartz Algorithm
Calculates option price and Greeks using Monte Carlo simulation
"""

import numpy as np
from datetime import datetime
from typing import Dict, Tuple
import json
import os


def read_inputs_from_file(filename=None):
    """
    Read and normalize input parameters from a JSON file.
    If no filename is provided, use ../Input/input.json relative to this script.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if filename is None:
        filename = os.path.join(base_dir, "..", "Input", "input.json")

    filename = os.path.normpath(filename)

    if not os.path.exists(filename):
        raise FileNotFoundError(f"JSON file not found: {filename}")

    with open(filename, "r") as f:
        data = json.load(f)

    # Normalize exercise style
    data["exercise_style"] = data.get("exercise_style", "").lower()
    
    # Normalize option type
    data["type"] = data.get("type", "").lower()

    # Ensure dividends is a list
    if not isinstance(data.get("dividends"), list):
        data["dividends"] = []

    return data


class AmericanOptionPricer:
    """
    American option pricing using Longstaff-Schwartz algorithm
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the pricer with configuration
        
        Args:
            config: Dictionary containing option parameters
        """
        self.type = config['type']  # 'call' or 'put'
        self.strike = config['strike']
        self.stock_price = config['stock_price']
        self.volatility = config['volatility']
        self.interest_rate = config['interest_rate']
        self.number_of_simulations = config.get('number_of_simulations', 10000)
        self.number_of_steps = config['number_of_steps']
        
        # Calculate time to maturity
        start_dt = datetime.strptime(
            f"{config['start_date']} {config['start_time']}", 
            "%Y-%m-%d %H:%M:%S"
        )
        exp_dt = datetime.strptime(
            f"{config['expiration_date']} {config['expiration_time']}", 
            "%Y-%m-%d %H:%M:%S"
        )
        self.time_to_maturity = (exp_dt - start_dt).total_seconds() / (365.25 * 24 * 3600)
        
        # Handle dividends
        self.dividends = config.get('dividends', [])
        
        # Random seed for reproducibility
        self.rng = np.random.RandomState(42)
        
        # Store random numbers for common random numbers technique
        self._base_random_numbers = None
        
    def simulate_paths(self, S0: float, use_stored_randoms: bool = False) -> np.ndarray:
        """
        Simulate stock price paths using Geometric Brownian Motion
        Uses Common Random Numbers technique for variance reduction
        
        Args:
            S0: Initial stock price
            use_stored_randoms: If True, use stored random numbers for CRN
            
        Returns:
            Array of shape (number_of_steps + 1, number_of_simulations)
        """
        dt = self.time_to_maturity / self.number_of_steps
        n_steps = self.number_of_steps
        n_sims = self.number_of_simulations
        
        # Initialize paths array
        paths = np.zeros((n_steps + 1, n_sims))
        paths[0] = S0
        
        # Generate or reuse random shocks (Common Random Numbers)
        if use_stored_randoms and self._base_random_numbers is not None:
            z = self._base_random_numbers
        else:
            z = self.rng.standard_normal((n_steps, n_sims))
            if not use_stored_randoms:
                # Store for future use with CRN
                self._base_random_numbers = z
        
        # Simulate paths using GBM
        for t in range(1, n_steps + 1):
            drift = (self.interest_rate - 0.5 * self.volatility ** 2) * dt
            diffusion = self.volatility * np.sqrt(dt) * z[t - 1]
            paths[t] = paths[t - 1] * np.exp(drift + diffusion)
        
        return paths
    
    def payoff(self, spot: np.ndarray) -> np.ndarray:
        """
        Calculate option payoff
        
        Args:
            spot: Stock prices
            
        Returns:
            Payoff values
        """
        if self.type == 'call':
            return np.maximum(spot - self.strike, 0.0)
        else:  # put
            return np.maximum(self.strike - spot, 0.0)
    
    def discount_factor(self, t_from: float, t_to: float) -> float:
        """
        Calculate discount factor
        
        Args:
            t_from: Start time
            t_to: End time
            
        Returns:
            Discount factor
        """
        return np.exp(-self.interest_rate * (t_to - t_from))
    
    def fit_continuation_value(self, x: np.ndarray, y: np.ndarray) -> np.polynomial.Polynomial:
        """
        Fit polynomial to approximate continuation value
        
        Args:
            x: Stock prices (independent variable)
            y: Discounted cash flows (dependent variable)
            
        Returns:
            Fitted polynomial
        """
        return np.polynomial.Polynomial.fit(x, y, deg=2, rcond=None)
    
    def longstaff_schwartz(self, paths: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Implement Longstaff-Schwartz algorithm for American option pricing
        
        Args:
            paths: Simulated stock price paths
            
        Returns:
            Tuple of (option_value, optimal_exercise_times)
        """
        n_steps = paths.shape[0] - 1
        n_sims = paths.shape[1]
        dt = self.time_to_maturity / n_steps
        
        # Initialize cash flow matrix
        cash_flows = np.zeros((n_steps + 1, n_sims))
        
        # Set terminal payoff
        cash_flows[-1] = self.payoff(paths[-1])
        
        # Backward induction
        for t in range(n_steps - 1, 0, -1):
            # Current spot prices
            spot = paths[t]
            
            # Immediate exercise value
            exercise_value = self.payoff(spot)
            
            # Find in-the-money paths
            itm = exercise_value > 0
            
            if np.sum(itm) > 0:
                # Discount future cash flows
                discount = self.discount_factor(t * dt, (t + 1) * dt)
                continuation_cf = cash_flows[t + 1] * discount
                
                # Fit continuation value for ITM paths only
                try:
                    poly = self.fit_continuation_value(
                        spot[itm], 
                        continuation_cf[itm]
                    )
                    continuation_value = poly(spot)
                    continuation_value[~itm] = 0  # Set OTM to 0
                except:
                    # If fitting fails, use zero continuation
                    continuation_value = np.zeros_like(spot)
                
                # Exercise if immediate value > continuation value
                exercise = (exercise_value > continuation_value) & itm
                
                # Update cash flows
                cash_flows[t] = np.where(
                    exercise,
                    exercise_value,
                    cash_flows[t + 1] * discount
                )
            else:
                # No ITM paths, continue holding
                discount = self.discount_factor(t * dt, (t + 1) * dt)
                cash_flows[t] = cash_flows[t + 1] * discount
        
        # Discount cash flows to present value
        option_value = np.mean(cash_flows[1] * self.discount_factor(0, dt))
        
        return option_value, cash_flows
    
    def calculate_greeks(self, base_price: float) -> Dict[str, float]:
        """
        Calculate option Greeks using finite differences with variance reduction
        Uses Common Random Numbers (CRN) and optimized bumps
        
        Args:
            base_price: Base option price
            
        Returns:
            Dictionary containing Greeks
        """
        greeks = {}
        
        # Store original values
        original_S = self.stock_price
        original_vol = self.volatility
        original_r = self.interest_rate
        original_ttm = self.time_to_maturity
        
        # Optimized bumps (smaller = less bias, but need CRN to reduce variance)
        bump_s_pct = 0.01  # 1% for stock
        bump_v = 0.01      # 1% absolute for volatility
        bump_t = 1/365.25  # 1 day
        bump_r = 0.01      # 1% absolute for rate (was 0.0001, too small)
        
        # === DELTA with CRN ===
        bump_s = bump_s_pct * self.stock_price
        
        self.stock_price = original_S + bump_s
        paths_up = self.simulate_paths(self.stock_price, use_stored_randoms=True)
        price_up, _ = self.longstaff_schwartz(paths_up)
        
        self.stock_price = original_S - bump_s
        paths_down = self.simulate_paths(self.stock_price, use_stored_randoms=True)
        price_down, _ = self.longstaff_schwartz(paths_down)
        
        self.stock_price = original_S
        greeks['delta'] = (price_up - price_down) / (2 * bump_s)
        
        # === GAMMA with CRN ===
        greeks['gamma'] = (price_up - 2 * base_price + price_down) / (bump_s ** 2)
        
        # === VEGA with CRN ===
        self.volatility = original_vol + bump_v
        paths_vega_up = self.simulate_paths(self.stock_price, use_stored_randoms=True)
        price_vega_up, _ = self.longstaff_schwartz(paths_vega_up)
        
        self.volatility = original_vol - bump_v
        paths_vega_down = self.simulate_paths(self.stock_price, use_stored_randoms=True)
        price_vega_down, _ = self.longstaff_schwartz(paths_vega_down)
        
        self.volatility = original_vol
        greeks['vega'] = (price_vega_up - price_vega_down) / (2 * bump_v)
        
        # === THETA with CRN (central difference) ===
        # Theta = dV/dt (change in value as time passes)
        # We reduce time, so if value decreases, theta is negative (time decay)
        self.time_to_maturity = max(original_ttm - bump_t, 0.001)
        paths_theta = self.simulate_paths(self.stock_price, use_stored_randoms=True)
        price_theta, _ = self.longstaff_schwartz(paths_theta)
        self.time_to_maturity = original_ttm
        
        # theta = (V(t-dt) - V(t)) / dt, with convention that theta is negative for decay
        greeks['theta'] = (price_theta - base_price) / bump_t
        
        # === RHO with CRN and larger bump ===
        self.interest_rate = original_r + bump_r
        paths_rho_up = self.simulate_paths(self.stock_price, use_stored_randoms=True)
        price_rho_up, _ = self.longstaff_schwartz(paths_rho_up)
        
        self.interest_rate = original_r - bump_r
        paths_rho_down = self.simulate_paths(self.stock_price, use_stored_randoms=True)
        price_rho_down, _ = self.longstaff_schwartz(paths_rho_down)
        
        self.interest_rate = original_r
        greeks['rho'] = (price_rho_up - price_rho_down) / (2 * bump_r)
        
        # Restore original values
        self.stock_price = original_S
        self.volatility = original_vol
        self.interest_rate = original_r
        self.time_to_maturity = original_ttm
        
        return greeks
    
    def price(self) -> Dict[str, any]:
        """
        Calculate American option price and Greeks
        
        Returns:
            Dictionary containing price and Greeks
        """
        # Simulate paths and store random numbers for CRN
        paths = self.simulate_paths(self.stock_price, use_stored_randoms=False)
        
        # Calculate option price
        option_price, _ = self.longstaff_schwartz(paths)
        
        # Calculate Greeks using CRN
        greeks = self.calculate_greeks(option_price)
        
        return {
            'theoretical_price': round(option_price, 4),
            'delta': round(greeks['delta'], 4),
            'gamma': round(greeks['gamma'], 6),
            'vega': round(greeks['vega'], 4),
            'theta': round(greeks['theta'], 4),
            'rho': round(greeks['rho'], 4),
            'parameters': {
                'stock_price': self.stock_price,
                'strike': self.strike,
                'volatility': self.volatility,
                'interest_rate': self.interest_rate,
                'time_to_maturity': round(self.time_to_maturity, 4),
                'simulations': self.number_of_simulations,
                'steps': self.number_of_steps
            }
        }


def main():
    """
    Main function with interactive input similar to binary option calculator
    """
    print("\n" + "="*60)
    print("AMERICAN OPTION CALCULATOR (Longstaff-Schwartz)")
    print("="*60)
    
    choice = input("\nUse JSON file input? (y/n): ").strip().lower()

    if choice == "y":
        try:
            data = read_inputs_from_file()
            print("✓ Loaded input from ../Input/input.json")
        except Exception as e:
            print(f"Error loading JSON: {e}")
            print("\nTrying to load from current directory...")
            try:
                data = read_inputs_from_file("input.json")
                print("✓ Loaded input from ./input.json")
            except:
                print("✗ Could not find input.json file")
                exit(1)
    else:
        print("\nEnter parameters manually:\n")
        data = {
            "type": input("Option type (call / put): ").strip().lower(),
            "exercise_style": "american",
            "start_date": input("Start date (YYYY-MM-DD): ").strip(),
            "start_time": input("Start time (HH:MM:SS, default 09:00:00): ").strip() or "09:00:00",
            "expiration_date": input("Expiration date (YYYY-MM-DD): ").strip(),
            "expiration_time": input("Expiration time (HH:MM:SS, default 16:00:00): ").strip() or "16:00:00",
            "strike": float(input("Strike: ")),
            "stock_price": float(input("Stock price: ")),
            "volatility": float(input("Volatility (decimal, e.g. 0.2 for 20%): ")),
            "interest_rate": float(input("Interest rate (decimal, e.g. 0.05 for 5%): ")),
            "number_of_steps": int(input("Number of time steps (default 50): ") or 50),
            "number_of_simulations": int(input("Number of simulations (default 10000): ") or 10000),
            "average_type": "arithmetic",
            "dividends": []
        }
    
    # Validate exercise style
    if data.get("exercise_style", "").lower() != "american":
        print(f"\n⚠️  Warning: This calculator is for American options only.")
        print(f"   Exercise style '{data.get('exercise_style')}' will be treated as American.")
        data["exercise_style"] = "american"
    
    print("\n" + "="*60)
    print("CALCULATING...")
    print("="*60)
    
    try:
        # Create pricer and calculate
        pricer = AmericanOptionPricer(data)
        results = pricer.price()
        
        # Print results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"\nOption Type: {data['type'].upper()}")
        print(f"Exercise Style: American")
        
        print(f"\nParameters:")
        print(f"  Stock Price: ${results['parameters']['stock_price']:.2f}")
        print(f"  Strike Price: ${results['parameters']['strike']:.2f}")
        print(f"  Volatility: {results['parameters']['volatility']*100:.2f}%")
        print(f"  Risk-free Rate: {results['parameters']['interest_rate']*100:.2f}%")
        print(f"  Time to Maturity: {results['parameters']['time_to_maturity']:.4f} years")
        print(f"  Simulations: {results['parameters']['simulations']:,}")
        print(f"  Time Steps: {results['parameters']['steps']}")
        
        print(f"\nPricing Results:")
        print(f"  theoretical_price: {results['theoretical_price']}")
        
        print(f"\nGreeks:")
        print(f"  delta: {results['delta']}")
        print(f"  gamma: {results['gamma']}")
        print(f"  vega: {results['vega']}")
        print(f"  theta: {results['theta']}")
        print(f"  rho: {results['rho']}")
        print("="*60)
        
        # Save results to JSON in Output directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, "..", "Output")
        
        # Create Output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, "american_option_output.json")
        
        # Prepare output in same format as binary
        output_data = {
            "theoretical_price": results['theoretical_price'],
            "delta": results['delta'],
            "gamma": results['gamma'],
            "vega": results['vega'],
            "theta": results['theta'],
            "rho": results['rho']
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\nCalculation error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
