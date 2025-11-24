# OptionCalculator

## Overview

**OptionCalculator** is a sophisticated, full-stack financial application designed for **precise valuation of derivative options** across multiple exercise styles and pricing models. The system combines a robust **Python backend** (FastAPI) with a modern **Angular frontend**, providing both programmatic API access and an intuitive interactive interface for quantitative analysts, traders, and financial professionals.

The calculator implements state-of-the-art numerical methods and financial mathematics to compute theoretical option prices along with all critical Greeks (Delta, Gamma, Theta, Vega, Rho) required for portfolio risk management and hedging strategies.

---

## Key Features & Specialities

### 1. **Multi-Style Option Pricing**

The system supports four distinct option exercise styles, each with specialized pricing methodologies:

#### **European Options**
- **Pricing Model**: Black-Scholes analytical formula
- **Characteristics**: Can only be exercised at maturity
- **Dividend Treatment**: 
  - Discrete dividends applied - decucted from current stock price
  - Present value of dividends deducted from stock price
  - Greeks computed without dividend adjustments (per industry standard)
- **Use Case**: Index options, currency options, standard equity derivatives

#### **American Options**
- **Pricing Model**: Monte Carlo simulation with **Longstaff-Schwartz (LSM) algorithm**
- **Key Feature**: Supports **early exercise** optimization
- **Characteristics**:
  - Backward induction for optimal stopping time determination
  - Polynomial basis functions (1, S, S²) for continuation value estimation
  - Discrete dividend support with precise jump adjustments
  - Common random numbers for consistent Greeks calculation
- **Use Case**: Equity options, corporate securities, any American-style contract

#### **Asian Options**
- **Pricing Models**: 
  - **Arithmetic Average**: Monte Carlo simulation with variance reduction techniques
  - **Geometric Average**: Closed-form analytical solution
- **Advanced Techniques**:
  - Antithetic variates for variance reduction
  - Control variate methodology for improved convergence
  - Continuous dividend yield calculation from discrete dividends
  - Equally-spaced fixing dates with configurable granularity
- **Performance**: Geometric pricing is near-instantaneous; arithmetic uses efficient MC
- **Use Case**: Commodity options, averaging contracts, structured products

#### **Binary Options**
- **Pricing Model**: Black-Scholes with continuous dividend yield
- **Payoff Structures** (all fully supported):
  - **Cash-or-Nothing**: Pays fixed $1 if in-the-money
  - **Asset-or-Nothing**: Pays underlying spot price if in-the-money
  - **Custom**: Pays user-defined fixed payout if in-the-money
- **Greeks Calculation**: Finite difference method for robust Greeks
- **Dividend Treatment**: Continuous yield derived from discrete dividends
- **Use Case**: Barrier alternatives, FX binary options, exotic derivatives

---

### 2. **Comprehensive Greeks Calculation**

All option styles return the complete Greeks toolbox:

| Greek | Formula | Interpretation |
|-------|---------|-----------------|
| **Delta (Δ)** | ∂Price/∂S | Directional sensitivity; hedge ratio for dynamic replication |
| **Gamma (Γ)** | ∂²Price/∂S² | Convexity; rebalancing risk; acceleration of delta changes |
| **Theta (Θ)** | -∂Price/∂t | Time decay; P&L from calendar progression with constant spot |
| **Vega (ν)** | ∂Price/∂σ | Volatility sensitivity; exposure to realized vs. implied vol |
| **Rho (ρ)** | ∂Price/∂r | Interest rate sensitivity; portfolio duration metric |

**Implementation Details**:
- Black-Scholes models use analytical derivatives
- Binary options employ finite differences (step sizes: 1e-4)
- Asian model: bumb and reprice
- American models use numerical differentiation with variance reduction
- All Greeks rounded to 3 decimal places for precision

---

### 3. **Advanced Dividend Handling**

The system offers three sophisticated dividend management modes:

#### **No Dividends**
- Simple pricing without yield adjustments
- Full Greeks alignment

#### **Discrete Dividend Payments**
- User-defined dividend dates and amounts
- Format: `{date: "YYYY-MM-DD", amount: float}`
- **Processing**:
  - Present value calculation: PV = Σ D_i × e^(-r×t_i)
  - European: Applied as stock price adjustment
  - American: Precise discrete jumps in simulation paths
  - Asian: Converted to continuous yield (q)
  - Binary: Converted to continuous yield (q)
- **Validation**: Dividends strictly between start and expiration dates

#### **Dividend Streams** (Automatic Generation)
- Specify: Start date, dividend amount, day interval
- Automatically generates repeating dividends across option life
- Example: $0.50 quarterly = 4 payments per year

**Dividend Conversion Formula** (for Asian/Binary):
```
q = -(1/T) × ln((S - PV_div) / S)
```
where PV_div is the discounted present value of all dividends.

---

### 4. **Precision Time Calculation**

Options trading relies on exact time measurement:

- **Day Count Convention**: Actual/365 (market standard for equity derivatives)
- **Time Representation**: Years as floating-point decimals
- **Time Parsing**:
  - 24-hour format: `HH:MM:SS` (e.g., "14:30:00")
  - Business convention: `AM` (defaults to 09:30) / `PM` (defaults to 16:00)
- **Validation**: Expiration must be strictly after inception (expiry_datetime > start_datetime)

**Time Calculation**:
```
T = (expiry_datetime - start_datetime).total_seconds() / (365 × 24 × 3600)
```

---

### 5. **Input Validation & Schema**

**Robust JSON Schema validation** ensures data integrity:

| Parameter | Type | Range | Notes |
|-----------|------|-------|-------|
| **type** | enum | "call", "put" | Case-insensitive |
| **exercise_style** | enum | "european", "american", "asian", "binary" | Case-insensitive |
| **start_date** | string | Format: YYYY-MM-DD | Must be valid calendar date |
| **start_time** | string | HH:MM:SS or AM/PM | Flexible time parsing |
| **expiration_date** | string | Format: YYYY-MM-DD | Must be valid calendar date |
| **expiration_time** | string | HH:MM:SS or AM/PM | Flexible time parsing |
| **strike** | number | ≥ 0.01 | Strike price in option currency |
| **stock_price** | number | ≥ 0.01 | Current underlying spot price |
| **volatility** | number | > 0 | Accepts 20 (20%) or 0.20 (20%) automatically normalized |
| **interest_rate** | number | Any | Accepts 1.5 (1.5%) or 0.015 (1.5%) automatically normalized |
| **number_of_steps** | integer | 1 - 10,000 | MC simulation grid points (American/Asian) |
| **number_of_simulations** | integer | 1 - 100,000 | MC iteration count (American/Asian) |
| **average_type** | enum | "arithmetic", "geometric" | Asian-specific; **geometric** = fast closed-form |
| **binary_payoff_structure** | enum | "cash", "asset", "custom" | Binary-specific payoff type |
| **binary_payout** | number | ≥ 0.01 | Payout amount for binary options (ignored if cash/asset) |
| **dividends** | array | List of {date, amount} | Optional; each amount ≥ 0 |

**Automatic Normalization**:
- Volatility/Interest Rate: If `|x| > 1`, treated as percentage: `x / 100`
- Dates: Enforced YYYY-MM-DD ISO format
- Times: Case-insensitive (AM/PM) and flexible HH:MM:SS parsing

---

### 6. **Architecture: Full-Stack Separation**

#### **Backend (Python)**
- **Framework**: FastAPI with Pydantic validation
- **Core Modules**:
  - `server.py`: RESTful API orchestration
  - `MainCalculator.py`: CLI interface for direct execution
  - `AmericanCalculator.py`: LSM-based pricing
  - `AsianCalculator.py`: Geometric + MC arithmetic
  - `EuropeanCalculator.py`: Black-Scholes closed-form
  - `BinaryCalculator.py`: Binary option pricing
  - `Utils.py`: Core financial mathematics (Greeks, time calculation, discounting)
  - `UtilsInput.py`: Input validation, CLI prompts, dividend utilities

- **CORS Configuration**:
  - Allows: `http://localhost:4200`, `http://127.0.0.1:4200`
  - Methods: All HTTP verbs
  - Headers: All custom headers

#### **Frontend (Angular)**
- **Framework**: Angular 16+ (standalone components)
- **UI Features**:
  - Reactive forms with real-time validation
  - Dynamic dividend entry (add/remove)
  - Conditional field display (binary payout, average type)
  - Loading state with animated spinner
  - Error banner with detailed messages
  - Live Greeks display with 3 decimal precision

- **State Management**: Angular signals for reactive updates

---

### 7. **Usage Modes**

#### **Mode 1: API Server (Production)**
```bash
# Terminal 1: Start FastAPI
cd Backend
python -m uvicorn server:app --reload

# Terminal 2: Start Angular frontend
cd Frontend
npx ng serve
```
Access: `http://localhost:4200`

Programmatic API requests:
```bash
curl -X POST http://127.0.0.1:8000/api/price \
  -H "Content-Type: application/json" \
  -d '{
    "type": "call",
    "exercise_style": "european",
    "start_date": "2025-11-23",
    "start_time": "09:30:00",
    "expiration_date": "2026-05-23",
    "expiration_time": "16:00:00",
    "strike": 100.0,
    "stock_price": 105.0,
    "volatility": 0.20,
    "interest_rate": 0.05,
    "average_type": "arithmetic",
    "number_of_steps": 100,
    "number_of_simulations": 10000,
    "binary_payout": 1.0,
    "binary_payoff_structure": "cash",
    "dividends": []
  }'
```

#### **Mode 2: Command-Line Interface (Batch/Testing)**
```bash
cd Backend/Calculators
python MainCalculator.py
# Interactive prompts guide user through option configuration
```

Then:
```bash
# Option 1: Modify input.json and use it
python MainCalculator.py  # Choose "json" mode

# Option 2: Enter new data interactively
python MainCalculator.py  # Choose "new" mode
```

---

### 8. **Output Specification**

**Response JSON**:
```json
{
  "theoretical_price": 23.746,
  "delta": 0.853,
  "gamma": 0.009,
  "rho": 86.967,
  "theta": -3.801,
  "vega": 28.976
}
```

**Schema Validation**: All outputs validated against `output_schema.json`
- Theoretical price: ≥ 0
- Delta: [-1, 1]
- Gamma, Theta, Vega, Rho: No bounds (can be negative)

---

### 9. **Mathematical Engines**

#### **Black-Scholes (European, Binary Base)**
```
d1 = [ln(S/K) + (r - q + 0.5σ²)T] / (σ√T)
d2 = d1 - σ√T

Call: C = S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)
Put:  P = K·e^(-rT)·N(-d2) - S·e^(-qT)·N(-d1)
```

#### **Longstaff-Schwartz (American)**
1. Simulate GBM paths with discrete dividend jumps
2. Backward induction from maturity to inception
3. At each node: Compare intrinsic value vs. continuation value
4. Continuation value: Regression using (1, S, S²) basis
5. Exercise immediately if intrinsic > continuation

#### **Asian Geometric (Closed-Form)**
```
σ_g = σ · √[(n+1)(2n+1) / (6n²)]
μ_g = (r - q - σ²/2) · (n+1) / (2n)
```
Then apply Black-Scholes with adjusted volatility.

#### **Asian Arithmetic (Monte Carlo)**
```
Σ S_i / n = A_T (arithmetic average at expiry)
Payoff = max(A_T - K, 0) for calls
Variance reduction: Antithetic variates + control variate
```

#### **Binary Options**
- **Cash-or-Nothing**: `e^(-rT) · Payout · N(±d2)`
- **Asset-or-Nothing**: `S·e^(-qT) · N(±d1)`

---

### 10. **Numerical Stability Features**

- **Dividend Handling**: Prevents negative stock prices via `max(S - div, 0.0)`
- **Volatility Edge Cases**: Minimum volatility check in Asian geometric formula
- **Time Decay**: Graceful handling of T → 0 (returns intrinsic value)
- **Greeks Stability**: 
  - Finite difference step sizes scale with parameter magnitude
  - Prevents division by zero and numerical underflow
  - Fallback to intrinsic values when analytical formulas invalid

---

### 11. **Performance Characteristics**

| Option Type | Method | Speed | Accuracy | Best For |
|-------------|--------|-------|----------|----------|
| **European** | Analytical | **Instant** | Very High | Benchmarking, simple derivatives |
| **American** | LSM (10k sims) | 2-5 sec | Very High | Equity options, early exercise value |
| **Asian (Geometric)** | Analytical | **Instant** | Very High | Quick Asian pricing |
| **Asian (Arithmetic)** | MC (10k sims) | 3-8 sec | High | Accurate arithmetic averaging |
| **Binary** | FD Greeks | 1-2 sec | High | Binary/exotic derivatives |

---

## Installation & Setup

### Requirements
```
Python 3.10+
FastAPI, Pydantic, NumPy, SciPy
Angular 16+, Node.js 16+
```

### Backend Setup
```bash
cd Backend
pip install -r requirements.txt
```

### Frontend Setup
```bash
cd Frontend
npm install
npm install -g @angular/cli
```

### Makefile (Cross-Platform)
```bash
make backend-install   # Install Python deps
make frontend-install  # Install Node deps
make backend-start     # Start FastAPI server
make frontend-start    # Start Angular dev server
make frontend-clean    # Remove build artifacts
make all              # Full setup and launch
```

---

## Input/Output Examples

### Example 1: European Call with Discrete Dividends
**Input** (via API):
```json
{
  "type": "call",
  "exercise_style": "european",
  "start_date": "2025-11-23",
  "start_time": "09:30:00",
  "expiration_date": "2026-11-23",
  "expiration_time": "16:00:00",
  "strike": 100.0,
  "stock_price": 105.0,
  "volatility": 0.20,
  "interest_rate": 0.05,
  "average_type": "arithmetic",
  "number_of_steps": 100,
  "number_of_simulations": 10000,
  "binary_payout": 1.0,
  "binary_payoff_structure": "cash",
  "dividends": [
    {"date": "2026-02-23", "amount": 2.0},
    {"date": "2026-05-23", "amount": 2.0},
    {"date": "2026-08-23", "amount": 2.0},
    {"date": "2026-11-23", "amount": 2.0},
    {"date": "2026-02-23", "amount": 2.0}
  ]
}
```

**Output**:
```json
{
  "theoretical_price": 12.486,
  "delta": 0.641,
  "gamma": 0.018,
  "theta": -0.045,
  "vega": 25.340,
  "rho": 52.190
}
```

### Example 2: American Put with Early Exercise
**Input**:
```json
{
  "type": "put",
  "exercise_style": "american",
  "start_date": "2025-11-23",
  "start_time": "09:30:00",
  "expiration_date": "2026-05-23",
  "expiration_time": "16:00:00",
  "strike": 100.0,
  "stock_price": 95.0,
  "volatility": 0.25,
  "interest_rate": 0.04,
  "number_of_steps": 500,
  "number_of_simulations": 10000,
  "dividends": []
}
```

**Output**: Premium reflects early exercise optionality (higher than European equivalent)

### Example 3: Asian Arithmetic with Continuous Dividends
**Input**: Use dividend stream feature to auto-generate quarterly $1.50 dividends

**Output**: Asian arithmetic average benefits from dividend yield

---

## Error Handling

**Validation Errors** (HTTP 422):
```json
{
  "detail": [
    {
      "loc": ["body", "volatility"],
      "msg": "ensure this value is greater than 0",
      "type": "value_error.number.not_gt"
    }
  ]
}
```

**Logic Errors** (HTTP 400):
```json
{
  "detail": "Expiration datetime must be after start datetime."
}
```

**Server Errors** (HTTP 500):
- Logged to console with full traceback
- Generic error message returned to client for security

---

## Testing

### Backend Tests
```bash
cd Backend
# Manual test with provided input.json
python Calculators/MainCalculator.py

# API tests via curl
curl -X POST http://127.0.0.1:8000/api/price -d @input.json
```

### Frontend Tests
```bash
cd Frontend
ng test
ng e2e
```

---

## Known Limitations & Assumptions

1. **Dividend Dates**: Must fall strictly between start and expiration (not inclusive of expiration)
2**American Early Exercise**: Optimal stopping not guaranteed for all path combinations (Longstaff-Schwartz is biased low)
3**Asian Averaging**: Equal weighting of all fixings; no scheduled vs. actual date adjustments
4**Greeks Calculation**: Numerical differences for some option types; analytical only for vanilla European5
5**Time Convention**: Assumes 365-day year (not business day calendar)

---

## Technical Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Backend | Python | 3.10+ |
| API Framework | FastAPI | Latest |
| Numerical Computing | NumPy | 1.x |
| Statistics | SciPy | 1.x |
| Frontend | Angular | 16+ |
| Styling | CSS + Custom Design System | - |
| Build Tool | Angular CLI | - |
| Package Manager | pip (Python), npm (Node) | - |

---

## File Structure

```
OptionCalculator/
├── Backend/
│   ├── Calculators/
│   │   ├── __init__.py
│   │   ├── MainCalculator.py
│   │   ├── AmericanCalculator.py
│   │   ├── AsianCalculator.py
│   │   ├── EuropeanCalculator.py
│   │   ├── BinaryCalculator.py
│   │   ├── Utils.py
│   │   └── UtilsInput.py
│   ├── Input/
│   │   ├── input.json
│   │   └── input_schema.json
│   ├── Output/
│   │   ├── output.json
│   │   └── output_schema.json
│   ├── server.py
│   └── requirements.txt
├── Frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── app.ts
│   │   │   ├── app.html
│   │   │   └── app.css
│   │   └── ...
│   ├── package.json
│   └── angular.json
├── Makefile
└── README.md
```

---

## Contributing

Suggestions for enhancements:
- Add support for **quarterly** specified interest rates
- Implement **barrier options** (knock-in/knock-out)
- Support for **multiple underlyings** (basket options)
- Dashboard with **historical implied volatility surface**
- Real-time market data integration (Bloomberg, Yahoo Finance)

---
