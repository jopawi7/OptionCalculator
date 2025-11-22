import pytest
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Backend.Calculators.AmericanCalculator import calculate_option_value as american_calc
from Backend.Calculators.AsianCalculator import calculate_option_value as asian_calc
from Backend.Calculators.BinaryCalculator import calculate_option_value as binary_calc
from Backend.Calculators.EuropeanCalculator import calculate_option_value as european_calc


# ===================================================================
# AMERICAN CALCULATOR TESTS (12 test cases)
# ===================================================================

class TestAmericanCalculator:
    """12 American option test cases with strict input format."""

    def test_american_call_atm_standard(self):
        """Test 1: American Call ATM, standard parameters."""
        data = {
            "type": "call",
            "exercise_style": "american",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 200,
            "number_of_simulations": 10000,
            "dividends": []
        }
        result = american_calc(data)
        assert result["theoretical_price"] > 0
        assert 0 <= result["delta"] <= 1

    def test_american_put_atm_standard(self):
        """Test 2: American Put ATM, standard parameters."""
        data = {
            "type": "put",
            "exercise_style": "american",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 200,
            "number_of_simulations": 10000,
            "dividends": []
        }
        result = american_calc(data)
        assert result["theoretical_price"] > 0
        assert -1 <= result["delta"] <= 0

    def test_american_call_itm(self):
        """Test 3: American Call ITM."""
        data = {
            "type": "call",
            "exercise_style": "american",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 120.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 200,
            "number_of_simulations": 10000,
            "dividends": []
        }
        result = american_calc(data)
        intrinsic = max(120.0 - 100.0, 0)
        assert result["theoretical_price"] >= intrinsic - 0.1

    def test_american_call_otm(self):
        """Test 4: American Call OTM."""
        data = {
            "type": "call",
            "exercise_style": "american",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 80.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 200,
            "number_of_simulations": 10000,
            "dividends": []
        }
        result = american_calc(data)
        assert result["theoretical_price"] >= 0

    def test_american_call_1week(self):
        """Test 5: American Call 1 week to expiry."""
        start = datetime.strptime("2025-11-16", "%Y-%m-%d")
        end = start + timedelta(days=7)
        data = {
            "type": "call",
            "exercise_style": "american",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": end.strftime("%Y-%m-%d"),
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 200,
            "number_of_simulations": 10000,
            "dividends": []
        }
        result = american_calc(data)
        assert result["theoretical_price"] >= 0

    def test_american_call_6months(self):
        """Test 6: American Call 6 months to expiry."""
        start = datetime.strptime("2025-11-16", "%Y-%m-%d")
        end = start + timedelta(days=180)
        data = {
            "type": "call",
            "exercise_style": "american",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": end.strftime("%Y-%m-%d"),
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = american_calc(data)
        assert result["theoretical_price"] > 0

    def test_american_call_high_vol(self):
        """Test 7: American Call high volatility."""
        data = {
            "type": "call",
            "exercise_style": "american",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.80,
            "interest_rate": 0.05,
            "number_of_steps": 200,
            "number_of_simulations": 10000,
            "dividends": []
        }
        result = american_calc(data)
        assert result["theoretical_price"] > 0
        assert result["vega"] > 0

    def test_american_call_low_vol(self):
        """Test 8: American Call low volatility."""
        data = {
            "type": "call",
            "exercise_style": "american",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.05,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = american_calc(data)
        assert result["theoretical_price"] >= 0

    def test_american_put_with_dividends(self):
        """Test 9: American Put with dividends."""
        data = {
            "type": "put",
            "exercise_style": "american",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": [
                {"date": "2025-12-15", "amount": 1.5},
                {"date": "2026-03-15", "amount": 1.5}
            ]
        }
        result = american_calc(data)
        assert result["theoretical_price"] > 0

    def test_american_call_high_rate(self):
        """Test 10: American Call high interest rate."""
        data = {
            "type": "call",
            "exercise_style": "american",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.15,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = american_calc(data)
        assert result["theoretical_price"] > 0
        assert result["rho"] > 0

    def test_american_put_low_rate(self):
        """Test 11: American Put low interest rate."""
        data = {
            "type": "put",
            "exercise_style": "american",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.01,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = american_calc(data)
        assert result["theoretical_price"] > 0

    def test_american_call_deep_itm(self):
        """Test 12: American Call deep ITM."""
        data = {
            "type": "call",
            "exercise_style": "american",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 50.0,
            "stock_price": 150.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = american_calc(data)
        intrinsic = max(150.0 - 50.0, 0)
        assert result["theoretical_price"] >= intrinsic - 0.5


# ===================================================================
# EUROPEAN CALCULATOR TESTS (12 test cases)
# ===================================================================

class TestEuropeanCalculator:
    """12 European option test cases with strict input format."""

    def test_european_call_atm(self):
        """Test 13: European Call ATM."""
        data = {
            "type": "call",
            "exercise_style": "european",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = european_calc(data)
        assert result["theoretical_price"] > 0
        assert 0 <= result["delta"] <= 1

    def test_european_put_atm(self):
        """Test 14: European Put ATM."""
        data = {
            "type": "put",
            "exercise_style": "european",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = european_calc(data)
        assert result["theoretical_price"] > 0
        assert -1 <= result["delta"] <= 0

    def test_european_call_itm(self):
        """Test 15: European Call ITM."""
        data = {
            "type": "call",
            "exercise_style": "european",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 110.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = european_calc(data)
        assert result["theoretical_price"] > 0
        assert result["delta"] > 0.5

    def test_european_call_otm(self):
        """Test 16: European Call OTM."""
        data = {
            "type": "call",
            "exercise_style": "european",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 90.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = european_calc(data)
        assert 0 <= result["theoretical_price"] <= 10
        assert result["delta"] < 0.5

    def test_european_call_1week(self):
        """Test 17: European Call 1 week to expiry."""
        start = datetime.strptime("2025-11-16", "%Y-%m-%d")
        end = start + timedelta(days=7)
        data = {
            "type": "call",
            "exercise_style": "european",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": end.strftime("%Y-%m-%d"),
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = european_calc(data)
        assert result["theoretical_price"] >= 0

    def test_european_call_2years(self):
        """Test 18: European Call 2 years to expiry."""
        start = datetime.strptime("2025-11-16", "%Y-%m-%d")
        end = start + timedelta(days=730)
        data = {
            "type": "call",
            "exercise_style": "european",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": end.strftime("%Y-%m-%d"),
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = european_calc(data)
        assert result["theoretical_price"] > 0

    def test_european_put_with_dividends(self):
        """Test 19: European Put with multiple dividends."""
        data = {
            "type": "put",
            "exercise_style": "european",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": [
                {"date": "2025-12-15", "amount": 1.0},
                {"date": "2026-03-15", "amount": 1.0},
                {"date": "2026-06-15", "amount": 1.0}
            ]
        }
        result = european_calc(data)
        assert result["theoretical_price"] > 0

    def test_european_call_extreme_vol(self):
        """Test 20: European Call extreme volatility."""
        data = {
            "type": "call",
            "exercise_style": "european",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 1.50,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = european_calc(data)
        assert result["theoretical_price"] > 0

    def test_european_put_neg_rate(self):
        """Test 21: European Put with negative rate."""
        data = {
            "type": "put",
            "exercise_style": "european",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": -0.01,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = european_calc(data)
        assert result["theoretical_price"] > 0

    def test_european_call_parity_check(self):
        """Test 22: European Put-Call Parity."""
        data_call = {
            "type": "call",
            "exercise_style": "european",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result_call = european_calc(data_call)

        data_put = data_call.copy()
        data_put["type"] = "put"
        result_put = european_calc(data_put)

        # C - P ≈ S - K*e^(-rT)
        c_minus_p = result_call["theoretical_price"] - result_put["theoretical_price"]
        assert 0 <= c_minus_p <= 5  # Loose check for parity

    def test_european_call_strike_150(self):
        """Test 23: European Call with strike 150."""
        data = {
            "type": "call",
            "exercise_style": "european",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 150.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = european_calc(data)
        assert result["theoretical_price"] >= 0

    def test_european_put_strike_50(self):
        """Test 24: European Put with strike 50."""
        data = {
            "type": "put",
            "exercise_style": "european",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 50.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = european_calc(data)
        assert result["theoretical_price"] >= 0


# ===================================================================
# ASIAN CALCULATOR TESTS (12 test cases)
# ===================================================================

class TestAsianCalculator:
    """12 Asian option test cases with strict input format."""

    def test_asian_call_arithmetic(self):
        """Test 25: Asian Call arithmetic average."""
        data = {
            "type": "call",
            "exercise_style": "asian",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "average_type": "arithmetic",
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = asian_calc(data)
        assert result["theoretical_price"] > 0

    def test_asian_put_arithmetic(self):
        """Test 26: Asian Put arithmetic average."""
        data = {
            "type": "put",
            "exercise_style": "asian",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "average_type": "arithmetic",
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = asian_calc(data)
        assert result["theoretical_price"] > 0

    def test_asian_call_geometric(self):
        """Test 27: Asian Call geometric average."""
        data = {
            "type": "call",
            "exercise_style": "asian",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "average_type": "geometric",
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = asian_calc(data)
        assert result["theoretical_price"] > 0

    def test_asian_put_geometric(self):
        """Test 28: Asian Put geometric average."""
        data = {
            "type": "put",
            "exercise_style": "asian",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "average_type": "geometric",
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = asian_calc(data)
        assert result["theoretical_price"] > 0

    def test_asian_call_itm_arithmetic(self):
        """Test 29: Asian Call ITM arithmetic."""
        data = {
            "type": "call",
            "exercise_style": "asian",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 120.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "average_type": "arithmetic",
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = asian_calc(data)
        assert result["theoretical_price"] > 0

    def test_asian_call_otm_arithmetic(self):
        """Test 30: Asian Call OTM arithmetic."""
        data = {
            "type": "call",
            "exercise_style": "asian",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 80.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "average_type": "arithmetic",
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = asian_calc(data)
        assert result["theoretical_price"] >= 0

    def test_asian_call_high_vol(self):
        """Test 31: Asian Call high volatility."""
        data = {
            "type": "call",
            "exercise_style": "asian",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.60,
            "interest_rate": 0.05,
            "average_type": "arithmetic",
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = asian_calc(data)
        assert result["theoretical_price"] > 0

    def test_asian_put_with_dividends(self):
        """Test 32: Asian Put with dividends."""
        data = {
            "type": "put",
            "exercise_style": "asian",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "average_type": "arithmetic",
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": [
                {"date": "2025-12-15", "amount": 2.0},
                {"date": "2026-06-15", "amount": 2.0}
            ]
        }
        result = asian_calc(data)
        assert result["theoretical_price"] > 0

    def test_asian_call_1month(self):
        """Test 33: Asian Call 1 month to expiry."""
        start = datetime.strptime("2025-11-16", "%Y-%m-%d")
        end = start + timedelta(days=30)
        data = {
            "type": "call",
            "exercise_style": "asian",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": end.strftime("%Y-%m-%d"),
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "average_type": "arithmetic",
            "number_of_steps": 50,
            "number_of_simulations": 3000,
            "dividends": []
        }
        result = asian_calc(data)
        assert result["theoretical_price"] >= 0

    def test_asian_call_3months(self):
        """Test 34: Asian Call 3 months to expiry."""
        start = datetime.strptime("2025-11-16", "%Y-%m-%d")
        end = start + timedelta(days=90)
        data = {
            "type": "call",
            "exercise_style": "asian",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": end.strftime("%Y-%m-%d"),
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "average_type": "arithmetic",
            "number_of_steps": 75,
            "number_of_simulations": 4000,
            "dividends": []
        }
        result = asian_calc(data)
        assert result["theoretical_price"] > 0

    def test_asian_put_low_rate(self):
        """Test 35: Asian Put low interest rate."""
        data = {
            "type": "put",
            "exercise_style": "asian",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.01,
            "average_type": "geometric",
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = asian_calc(data)
        assert result["theoretical_price"] > 0

    def test_asian_call_strike_120(self):
        """Test 36: Asian Call strike 120."""
        data = {
            "type": "call",
            "exercise_style": "asian",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 120.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "average_type": "arithmetic",
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = asian_calc(data)
        assert result["theoretical_price"] >= 0


# ===================================================================
# BINARY CALCULATOR TESTS (12 test cases)
# ===================================================================

class TestBinaryCalculator:
    """12 Binary option test cases with strict input format."""

    def test_binary_call_atm(self):
        """Test 37: Binary Call ATM."""
        data = {
            "type": "call",
            "exercise_style": "binary",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = binary_calc(data)
        assert 0 <= result["theoretical_price"] <= 1

    def test_binary_put_atm(self):
        """Test 38: Binary Put ATM."""
        data = {
            "type": "put",
            "exercise_style": "binary",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = binary_calc(data)
        assert 0 <= result["theoretical_price"] <= 1

    def test_binary_call_itm(self):
        """Test 39: Binary Call ITM."""
        data = {
            "type": "call",
            "exercise_style": "binary",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 120.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = binary_calc(data)
        assert result["theoretical_price"] > 0.5

    def test_binary_call_otm(self):
        """Test 40: Binary Call OTM."""
        data = {
            "type": "call",
            "exercise_style": "binary",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 80.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = binary_calc(data)
        assert result["theoretical_price"] < 0.5

    def test_binary_put_itm(self):
        """Test 41: Binary Put ITM."""
        data = {
            "type": "put",
            "exercise_style": "binary",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 80.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = binary_calc(data)
        assert result["theoretical_price"] > 0.5

    def test_binary_call_deep_itm(self):
        """Test 42: Binary Call deep ITM."""
        data = {
            "type": "call",
            "exercise_style": "binary",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 150.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = binary_calc(data)
        assert result["theoretical_price"] > 0.9

    def test_binary_call_deep_otm(self):
        """Test 43: Binary Call deep OTM."""
        data = {
            "type": "call",
            "exercise_style": "binary",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 50.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = binary_calc(data)
        assert result["theoretical_price"] < 0.1

    def test_binary_call_high_vol(self):
        """Test 44: Binary Call high volatility."""
        data = {
            "type": "call",
            "exercise_style": "binary",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.80,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = binary_calc(data)
        assert 0 <= result["theoretical_price"] <= 1

    def test_binary_call_low_vol(self):
        """Test 45: Binary Call low volatility."""
        data = {
            "type": "call",
            "exercise_style": "binary",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.05,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = binary_calc(data)
        assert 0 <= result["theoretical_price"] <= 1

    def test_binary_put_with_dividends(self):
        """Test 46: Binary Put with dividends."""
        data = {
            "type": "put",
            "exercise_style": "binary",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": [
                {"date": "2025-12-15", "amount": 1.0},
                {"date": "2026-06-15", "amount": 1.0}
            ]
        }
        result = binary_calc(data)
        assert 0 <= result["theoretical_price"] <= 1

    def test_binary_call_1week(self):
        """Test 47: Binary Call 1 week to expiry."""
        start = datetime.strptime("2025-11-16", "%Y-%m-%d")
        end = start + timedelta(days=7)
        data = {
            "type": "call",
            "exercise_style": "binary",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": end.strftime("%Y-%m-%d"),
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 50,
            "number_of_simulations": 3000,
            "dividends": []
        }
        result = binary_calc(data)
        assert 0 <= result["theoretical_price"] <= 1

    def test_binary_call_high_rate(self):
        """Test 48: Binary Call high interest rate."""
        data = {
            "type": "call",
            "exercise_style": "binary",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.15,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = binary_calc(data)
        assert 0 <= result["theoretical_price"] <= 1


# ===================================================================
# CROSS-CALCULATOR TESTS (12 test cases)
# ===================================================================

class TestCrossCalculatorValidation:
    """12 cross-calculator validation test cases."""

    def test_american_gte_european_call(self):
        """Test 49: American Call >= European Call."""
        data = {
            "type": "call",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        data["exercise_style"] = "american"
        result_american = american_calc(data)

        data["exercise_style"] = "european"
        result_european = european_calc(data)

        assert result_american["theoretical_price"] >= result_european["theoretical_price"] - 1.0

    def test_american_gte_european_put(self):
        """Test 50: American Put >= European Put."""
        data = {
            "type": "put",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        data["exercise_style"] = "american"
        result_american = american_calc(data)

        data["exercise_style"] = "european"
        result_european = european_calc(data)

        assert result_american["theoretical_price"] >= result_european["theoretical_price"] - 1.0

    def test_call_delta_positive(self):
        """Test 51: Call delta always positive."""
        data = {
            "type": "call",
            "exercise_style": "european",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = european_calc(data)
        assert result["delta"] >= 0

    def test_put_delta_negative(self):
        """Test 52: Put delta always negative."""
        data = {
            "type": "put",
            "exercise_style": "european",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = european_calc(data)
        assert result["delta"] <= 0

    def test_gamma_always_positive(self):
        """Test 53: Gamma always positive."""
        for option_type in ["call", "put"]:
            data = {
                "type": option_type,
                "exercise_style": "european",
                "start_date": "2025-11-16",
                "start_time": "09:30:00",
                "expiration_date": "2026-11-16",
                "expiration_time": "16:00:00",
                "strike": 100.0,
                "stock_price": 100.0,
                "volatility": 0.20,
                "interest_rate": 0.05,
                "number_of_steps": 100,
                "number_of_simulations": 5000,
                "dividends": []
            }
            result = european_calc(data)
            assert result["gamma"] >= 0

    def test_vega_always_positive(self):
        """Test 54: Vega always positive."""
        for option_type in ["call", "put"]:
            data = {
                "type": option_type,
                "exercise_style": "european",
                "start_date": "2025-11-16",
                "start_time": "09:30:00",
                "expiration_date": "2026-11-16",
                "expiration_time": "16:00:00",
                "strike": 100.0,
                "stock_price": 100.0,
                "volatility": 0.20,
                "interest_rate": 0.05,
                "number_of_steps": 100,
                "number_of_simulations": 5000,
                "dividends": []
            }
            result = european_calc(data)
            assert result["vega"] > 0

    def test_call_rho_positive(self):
        """Test 55: Call rho positive."""
        data = {
            "type": "call",
            "exercise_style": "european",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = european_calc(data)
        assert result["rho"] > 0

    def test_put_rho_negative(self):
        """Test 56: Put rho negative."""
        data = {
            "type": "put",
            "exercise_style": "european",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = european_calc(data)
        assert result["rho"] < 0

    def test_option_value_nonnegative(self):
        """Test 57: Option values always non-negative."""
        for calc_func, exercise_style in [
            (american_calc, "american"),
            (european_calc, "european"),
            (asian_calc, "asian"),
            (binary_calc, "binary")
        ]:
            data = {
                "type": "call",
                "exercise_style": exercise_style,
                "start_date": "2025-11-16",
                "start_time": "09:30:00",
                "expiration_date": "2026-11-16",
                "expiration_time": "16:00:00",
                "strike": 100.0,
                "stock_price": 100.0,
                "volatility": 0.20,
                "interest_rate": 0.05,
                "number_of_steps": 100,
                "number_of_simulations": 5000,
                "dividends": []
            }
            if exercise_style == "asian":
                data["average_type"] = "arithmetic"
            result = calc_func(data)
            assert result["theoretical_price"] >= 0

    def test_deep_itm_call_intrinsic_lower_bound(self):
        """Test 58: Deep ITM call >= intrinsic value."""
        data = {
            "type": "call",
            "exercise_style": "european",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 50.0,
            "stock_price": 150.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }
        result = european_calc(data)
        intrinsic = max(150.0 - 50.0, 0)
        assert result["theoretical_price"] >= intrinsic - 0.1

    def test_deep_itm_put_intrinsic_lower_bound(self):
        """Test 59: Deep ITM put >= intrinsic value."""
        data = {
            "type": "put",
            "exercise_style": "european",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 150.0,
            "stock_price": 50.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 6000,
            "dividends": []
        }
        result = european_calc(data)
        intrinsic = max(150.0 - 50.0, 0)
        assert result["theoretical_price"] >= intrinsic - 0.1

    def test_all_calculators_return_dict(self):
        """Test 60: All calculators return proper dictionary."""
        data = {
            "type": "call",
            "start_date": "2025-11-16",
            "start_time": "09:30:00",
            "expiration_date": "2026-11-16",
            "expiration_time": "16:00:00",
            "strike": 100.0,
            "stock_price": 100.0,
            "volatility": 0.20,
            "interest_rate": 0.05,
            "number_of_steps": 100,
            "number_of_simulations": 5000,
            "dividends": []
        }

        for calc_func, exercise_style in [
            (american_calc, "american"),
            (european_calc, "european"),
            (asian_calc, "asian"),
            (binary_calc, "binary")
        ]:
            data["exercise_style"] = exercise_style
            if exercise_style == "asian":
                data["average_type"] = "arithmetic"
            result = calc_func(data)

            assert isinstance(result, dict)
            assert "theoretical_price" in result
            assert "delta" in result
            assert "gamma" in result
            assert "vega" in result
            assert "theta" in result
            assert "rho" in result


if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short"
    ])