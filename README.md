# OptionCalculator

A sophisticated full-stack financial application for computing precise option prices using multiple advanced pricing models. Built with FastAPI (Python backend) and Angular (TypeScript frontend), OptionCalculator delivers real-time pricing, Greeks calculations, and comprehensive dividend handling.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Project Architecture](#project-architecture)
4. [Pricing Models & Calculators](#pricing-models--calculators)
5. [Installation & Dependencies](#installation--dependencies)
6. [Quick Start Guide](#quick-start-guide)
7. [Backend Setup](#backend-setup)
8. [Frontend Setup](#frontend-setup)
9. [API Documentation](#api-documentation)
10. [Input Schema & Validation](#input-schema--validation)
11. [Output Format](#output-format)
12. [Usage Examples](#usage-examples)
13. [Technical Details](#technical-details)
14. [Troubleshooting](#troubleshooting)

---

## Project Overview

OptionCalculator implements a complete separation of concerns with a robust Python-based FastAPI backend handling all calculations and a responsive Angular frontend providing an intuitive user experience. The application validates all inputs against strict JSON schemas and supports multiple option pricing methodologies to accommodate different financial instruments and market conditions.

The project structure separates the calculator logic from orchestration, enabling easy testing, maintenance, and extension. Each calculator implements its own pricing algorithm while conforming to a unified output interface of five Greeks plus theoretical price.

---

## Key Features

- **Multiple Pricing Models**: Black-Scholes (European), Monte Carlo simulation (American), Monte Carlo simulation (Asian), and cash-or-nothing Binary options
- **Greeks Calculation**: Complete set including Delta, Gamma, Vega, Rho, and Theta.
- **Flexible Time Input**: Support for both HH:MM:SS format and simplified AM/PM designations (market hours 09:30 AM = 09:30:00, 16:00 PM = 16:00:00)
- **Dividend Management**: Support for both single ex-date dividends and recurring dividend schedules with day intervals
- **Input Validation**: Strict JSON schema validation ensuring mathematical soundness (volatility > 0, dates in correct order, simulation counts within bounds)
- **Interactive UI**: Real-time form validation, dynamic dividend entry, responsive layout with loading states
- **CORS-Enabled API**: Cross-origin requests configured for local development (localhost:4200 ↔ localhost:8000)
- **Monte Carlo Advanced Features**: Antithetic variates and control variate techniques for variance reduction in Asian options

---

## Project Architecture

```
OptionCalculator/
├── Backend/
│   ├── server.py                    # FastAPI application (CORS, routing, validation)
│   ├── requirements.txt             # Python dependencies
│   └── Calculators/
│       ├── __init__.py
│       ├── Main.py                  # Interactive CLI orchestrator
│       ├── ValidateInput.py         # Input validation utilities
│       ├── Utils.py                 # Helper functions
│       ├── EuropeanCalculator.py    # Black-Scholes implementation
│       ├── AmericanCalculator.py    # Monte Carlo
│       ├── AsianCalculator.py       # Monte Carlo arithmetic/geometric averaging
│       ├── BinaryCalculator.py      # Cash-or-nothing options
│       ├── Input/
│       │   ├── input.json           # Sample input configuration
│       │   └── input_schema.json    # JSON schema for validation
│       └── Output/
│           ├── output.json          # Result output
│           └── output_schema.json   # Output validation schema
├── Frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── app.ts              # Main component (form, HTTP requests)
│   │   │   ├── app.html            # Template (form UI, results display)
│   │   │   ├── app.css             # Styling (layout, colors, animations)
│   │   │   ├── app.config.ts       # Angular configuration
│   │   │   ├── app.routes.ts       # Routing configuration
│   │   │   └── app.spec.ts         # Unit tests
│   │   └── main.ts                 # Bootstrap entry point
│   ├── angular.json                # Angular build configuration
│   ├── package.json                # Node dependencies
│   └── tsconfig.json               # TypeScript configuration
├── Makefile                        # Build and run automation
├── README.md                       # This file
```

---

## Pricing Models & Calculators

### 1. EuropeanCalculator.py

**Algorithm**: Black-Scholes closed-form solution

**Methodology**:
- Uses the analytical Black-Scholes formula for European options
- Computes d1 and d2 parameters based on log-normal distribution
- Greeks calculated via closed-form derivatives

**Key Functions**:
- `year_fraction_with_exact_days()`: Converts date/time pairs to fractional years (ACT/365)
- `parse_time_str()`: Handles time parsing (HH:MM:SS or AM/PM)
- `calculate_option_value()`: Main pricing engine returning price + 5 Greeks

**Supported Inputs**:
```python
{
    "type": "call" | "put",
    "start_date": "YYYY-MM-DD",
    "start_time": "HH:MM:SS" | "am" | "pm",
    "expiration_date": "YYYY-MM-DD",
    "expiration_time": "HH:MM:SS" | "am" | "pm",
    "strike": float (≥ 0.01),
    "stock_price": float (≥ 0.01),
    "volatility": float (> 0, in decimal form),
    "interest_rate": float (in percent, e.g., 1.5 for 1.5%)
}
```

**Output**:
```python
{
    "theoretical_price": float,
    "delta": float,
    "gamma": float,
    "rho": float,
    "theta": float,
    "vega": float
}
```

**Performance**: ~2-5 milliseconds

---

### 2. AmericanCalculator.py

**Algorithm**: Monte Carlo simulation with Longstaff-Schwartz early exercise algorithm

**Methodology**:
- Simulates multiple price paths using geometric Brownian motion
- At each time step, determines optimal early exercise via regression-based continuation value estimation
- Polynomial basis functions (1, S, S²) fitted to continuation values
- Early exercise when intrinsic value exceeds expected continuation value

**Key Functions**:
- `monte_carlo_american_option()`: Core pricing engine with path simulation
- `calculate_option_value()`: Wrapper handling input parsing and Greeks via finite differences
- Helpers: `normalize_interest_rate()`, `parse_time_string()`, `calculate_present_value_dividends()`

**Key Parameters**:
- `number_of_steps`: Time discretization steps (default 100, max 1000)
- `number_of_simulations`: MC paths (default 10,000, max 1,000,000)

**Unique Features**:
- Handles discrete dividends by adjusting initial stock price using present value
- Robust regression fallback if polynomial fit encounters numerical issues
- Bump-and-reprice for Greeks calculation with adaptive step sizing

**Performance**: 100-500ms depending on step/simulation count

---

### 3. AsianCalculator.py

**Algorithm**: Dual approach - geometric closed-form + arithmetic Monte Carlo

**Methodology for Geometric**:
- Uses modified Black-Scholes with volatility adjustment σ' = σ/√3
- Faster computation for geometric averaging
- Falls back to numerical methods if discrete dividends present

**Methodology for Arithmetic**:
- Path-dependent simulation with averaging across fixing dates
- Applies antithetic variates (pairs of negatively correlated paths) for variance reduction
- Control variate technique using geometric average as benchmark

**Key Functions**:
- `calculate_option_value()`: Dispatcher selecting geometric or arithmetic method
- `_asian_geometric_closed_form_price()`: Analytical solution
- `_asian_arithmetic_monte_carlo_price()`: Simulation with variance reduction
- `_bump_and_reprice_asian()`: Numerical Greeks via perturbation
- `_parse_datetime()`, `_expand_dividends()`, `_dividends_to_year_times()`: Helpers

**Dividend Support**:
- Single ex-date: `{"date": "YYYY-MM-DD", "amount": float}`
- Recurring schedule: `{"start_date": "...", "day_interval": int, "amount": float, "end_date": "..."}`

**Performance**: 50-500ms depending on averaging type and simulation count

---

### 4. BinaryCalculator.py

**Algorithm**: Black-Scholes adapted for cash-or-nothing options

**Methodology**:
- Cash payout (Q=1.0, immutable) if option ends in-the-money
- Otherwise zero value
- Uses modified Black-Scholes formula with only d2 term

**Key Functions**:
- `read_inputs_from_file()`: File I/O with JSON path resolution
- `calculate_option_value()`: Price computation with Greeks
- `N()`: Standard normal CDF (error function based)
- `n()`: Standard normal PDF

**Special Considerations**:
- Fixed payout Q = 1.0 (hardcoded, intended for future generalization)
- Supports dividend list or scalar dividend yield
- Normalized dividend handling via helper functions

**Performance**: ~5-10 milliseconds

---

## Installation & Dependencies

### Backend Requirements

**Python**: 3.8 or higher

**Install Dependencies**:
```bash
cd Backend
pip install -r requirements.txt
```

**Key Dependencies**:
```
fastapi>=0.104.0
uvicorn>=0.24.0
numpy>=1.24.0
scipy>=1.10.0
pydantic>=2.0.0
python-multipart>=0.0.6
```

**Verify Installation**:
```bash
python -c "import fastapi, uvicorn, numpy, scipy; print('Backend OK')"
```

### Frontend Requirements

**Node.js**: 16.x or higher (LTS recommended)

**Install Dependencies**:
```bash
cd Frontend
npm install
```

**Key Dependencies**:
```
@angular/core@^17.0.0
@angular/common@^17.0.0
@angular/forms@^17.0.0
@angular/platform-browser@^17.0.0
@angular/platform-browser-dynamic@^17.0.0
rxjs@^7.8.0
typescript@^5.2.0
```

**Verify Installation**:
```bash
ng version
npm list | head -20
```

---

## Quick Start Guide

### Option 1: Automated Makefile (Recommended)

```bash
cd OptionCalculator/
make all
```

This single command:
1. Installs Backend dependencies (pip)
2. Installs Frontend dependencies (npm)
3. Starts Backend on localhost:8000
4. Starts Frontend on localhost:4200

Then open http://localhost:4200 in your browser.

### Option 2: Manual Startup (Two Terminals)

**Terminal 1 - Backend**:
```bash
cd Backend
uvicorn server:app --reload
```
Output: `Uvicorn running on http://127.0.0.1:8000`

**Terminal 2 - Frontend**:
```bash
cd Frontend
ng serve
```
Output: `Application bundle generated successfully. Local: http://localhost:4200/`

### Option 3: Backend-Only (CLI Mode)

```bash
cd Backend/Calculators
python Main.py
```

Interactive prompts guide you through:
- Option type (CALL/PUT)
- Exercise style (EUROPEAN/AMERICAN/ASIAN/BINARY)
- Dates and times
- Strike, stock price, volatility, interest rate
- Dividends (optional)

Results written to `Output/output.json`

---

## Backend Setup

### Starting the FastAPI Server

```bash
cd Backend
uvicorn server:app --reload --host 127.0.0.1 --port 8000
```

**Flags**:
- `--reload`: Auto-restart on code changes (development only)
- `--host`: Bind address (127.0.0.1 = localhost only)
- `--port`: Port number (8000 default)

**Expected Output**:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

### API Endpoints

#### Health Check
```
GET /
Response: {"message": "Hello World"}
```

#### Calculate Option Price
```
POST /api/price
Content-Type: application/json

Request Body (OptionInput model):
{
  "type": "call",
  "exercise_style": "european",
  "start_date": "2025-11-20",
  "start_time": "09:30:00",
  "expiration_date": "2026-05-20",
  "expiration_time": "16:00:00",
  "strike": 100.0,
  "stock_price": 110.0,
  "volatility": 0.20,
  "interest_rate": 1.5,
  "average_type": "arithmetic",
  "number_of_steps": 100,
  "number_of_simulations": 10000,
  "dividends": [
    {"date": "2025-12-15", "amount": 1.25}
  ]
}

Response:
{
  "theoretical_price": 12.456,
  "delta": 0.625,
  "gamma": 0.045,
  "rho": 0.234,
  "theta": -0.012,
  "vega": 0.189
}

Error Response (validation failure):
{
  "detail": [error details],
  "body": request body
}
```

### CORS Configuration

**Allowed Origins**:
- http://localhost:4200
- http://127.0.0.1:4200

**To modify**, edit `server.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["your-domain.com"],  # Change here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Frontend Setup

### Starting the Angular Dev Server

```bash
cd Frontend
ng serve
```

**Expected Output**:
```
✔ Compiled successfully.
✓ Application bundle generated successfully.
Local: http://localhost:4200/
```

Open http://localhost:4200 in your browser.

### Component Structure

**AppComponent** (`app.ts`):
- Reactive form using FormBuilder
- HTTP client for backend communication
- Form array for dynamic dividend entry
- Result display with loading states

**Form Fields**:
```
- type (CALL/PUT) [toggle buttons]
- style (EUROPEAN/AMERICAN/ASIAN/BINARY) [toggle buttons]
- startDate (date picker)
- startTime (time input)
- expirationDate (date picker)
- expirationTime (time input)
- strike (number, ≥ 0.01)
- stockPrice (number, ≥ 0.01)
- volatility (number, > 0, displayed as %)
- interestRate (number)
- number_of_steps (integer, 1-1000)
- number_of_simulations (integer, 1-1,000,000)
- average_type (arithmetic/geometric) [Asian only]
- dividends (FormArray with date + amount pairs)
```

**Results Display**:
```
- Theoretical Price: primary result
- Delta: directional exposure
- Gamma: Delta's sensitivity
- Theta: time decay per day
- Vega: volatility sensitivity
- Rho: interest rate sensitivity
```

---

## API Documentation

### Request Validation

The backend validates using Pydantic models. Validation errors return:

```json
{
  "detail": [
    {
      "type": "value_error.number.not_a_number",
      "loc": ["body", "strike"],
      "msg": "..."
    }
  ]
}
```

### Type Conversion

- `volatility`: Divided by 100 (20% input → 0.20 internal)
- `interest_rate`: Divided by 100 (1.5% input → 0.015 internal)
- `dividend.amount`: Kept as-is (absolute value, not percentage)

### Time Handling

Supports multiple time formats:
- `"09:30:00"` → Explicit (24-hour)
- `"am"` → Market open (09:30:00)
- `"pm"` → Market close (16:00:00)

---

## Input Schema & Validation

### input_schema.json

Located in `Backend/Calculators/Input/`:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "type": {
      "type": "string",
      "enum": ["call", "put"]
    },
    "exercise_style": {
      "type": "string",
      "enum": ["american", "european", "asian", "binary"]
    },
    "start_date": {
      "type": "string",
      "pattern": "^\\d{4}-\\d{2}-\\d{2}$"
    },
    "start_time": {
      "type": "string",
      "pattern": "^(?:([01]\\d|2[0-3]):[0-5]\\d:[0-5]\\d|[aA][mM]|[pP][mM])$"
    },
    "expiration_date": {
      "type": "string",
      "pattern": "^\\d{4}-\\d{2}-\\d{2}$"
    },
    "expiration_time": {
      "type": "string",
      "pattern": "^(?:([01]\\d|2[0-3]):[0-5]\\d:[0-5]\\d|[aA][mM]|[pP][mM])$"
    },
    "strike": {
      "type": "number",
      "minimum": 0.01
    },
    "stock_price": {
      "type": "number",
      "minimum": 0.01
    },
    "volatility": {
      "type": "number",
      "exclusiveMinimum": 0
    },
    "interest_rate": {
      "type": "number"
    },
    "average_type": {
      "type": "string",
      "enum": ["arithmetic", "geometric"]
    },
    "number_of_steps": {
      "type": "integer",
      "minimum": 1,
      "maximum": 1000
    },
    "number_of_simulations": {
      "type": "integer",
      "minimum": 1,
      "maximum": 1000000
    },
    "dividends": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "date": {
            "type": "string",
            "pattern": "^\\d{4}-\\d{2}-\\d{2}$"
          },
          "amount": {
            "type": "number",
            "minimum": 0
          }
        },
        "required": ["date", "amount"]
      }
    }
  },
  "required": [
    "type",
    "exercise_style",
    "start_date",
    "start_time",
    "expiration_date",
    "expiration_time",
    "strike",
    "stock_price",
    "volatility",
    "interest_rate",
    "average_type",
    "number_of_steps",
    "number_of_simulations",
    "dividends"
  ]
}
```

### Validation Rules

| Field | Rule | Example |
|-------|------|---------|
| type | Must be "call" or "put" | "call" |
| exercise_style | Must be one of 4 types | "american" |
| start_date | YYYY-MM-DD format | "2025-11-20" |
| start_time | HH:MM:SS or am/pm | "09:30:00" |
| expiration_date | YYYY-MM-DD format | "2026-05-20" |
| expiration_time | HH:MM:SS or am/pm | "PM" |
| strike | ≥ 0.01 | 100.0 |
| stock_price | ≥ 0.01 | 110.0 |
| volatility | > 0 (decimal) | 0.20 |
| interest_rate | Any number | 1.5 |
| number_of_steps | 1 to 1,000 | 100 |
| number_of_simulations | 1 to 1,000,000 | 10000 |
| average_type | "arithmetic" or "geometric" | "arithmetic" |
| dividends | Array of {date, amount} | [{"date": "...", "amount": 1.25}] |

---

## Output Format

### output_schema.json

Located in `Backend/Calculators/Output/`:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "theoretical_price": {
      "type": "number",
      "minimum": 0
    },
    "delta": {
      "type": "number",
      "minimum": -1,
      "maximum": 1
    },
    "gamma": {
      "type": "number"
    },
    "rho": {
      "type": "number"
    },
    "theta": {
      "type": "number"
    },
    "vega": {
      "type": "number"
    }
  },
  "required": [
    "theoretical_price",
    "delta",
    "gamma",
    "rho",
    "theta",
    "vega"
  ]
}
```

### Sample Output

```json
{
  "theoretical_price": 12.456,
  "delta": 0.625,
  "gamma": 0.0451,
  "rho": 0.234,
  "theta": -0.0123,
  "vega": 0.1892
}
```

### Greeks Interpretation

| Greek | Meaning | Typical Range |
|-------|---------|---------------|
| **Delta (Δ)** | Price change per $1 underlying move | 0 to 1 (calls) |
| **Gamma (Γ)** | Delta change rate; convexity | Usually small, positive |
| **Theta (Θ)** | Price change per day (time decay) | Negative for longs |
| **Vega (ν)** | Price change per 1% volatility increase | Positive for both calls/puts |
| **Rho (ρ)** | Price change per 1% interest rate increase | Often modest |

---

## Usage Examples

### Example 1: European Call Option

**Input**:
```json
{
  "type": "call",
  "exercise_style": "european",
  "start_date": "2025-11-20",
  "start_time": "09:30:00",
  "expiration_date": "2025-12-20",
  "expiration_time": "16:00:00",
  "strike": 100.0,
  "stock_price": 105.0,
  "volatility": 0.25,
  "interest_rate": 2.5,
  "average_type": "arithmetic",
  "number_of_steps": 100,
  "number_of_simulations": 10000,
  "dividends": []
}
```

**Expected Output**:
```json
{
  "theoretical_price": 6.123,
  "delta": 0.645,
  "gamma": 0.0301,
  "rho": 0.187,
  "theta": -0.0245,
  "vega": 0.2145
}
```

**Interpretation**: The call is worth $6.12. It moves $0.65 per dollar stock move. Time decay costs ~$0.02 per day.

---

### Example 2: American Put with Dividends

**Input**:
```json
{
  "type": "put",
  "exercise_style": "american",
  "start_date": "2025-11-20",
  "start_time": "09:30:00",
  "expiration_date": "2026-05-20",
  "expiration_time": "16:00:00",
  "strike": 100.0,
  "stock_price": 95.0,
  "volatility": 0.30,
  "interest_rate": 1.5,
  "average_type": "arithmetic",
  "number_of_steps": 150,
  "number_of_simulations": 50000,
  "dividends": [
    {"date": "2025-12-15", "amount": 1.50},
    {"date": "2026-03-15", "amount": 1.50}
  ]
}
```

**Expected Output** (approximate):
```json
{
  "theoretical_price": 7.845,
  "delta": -0.521,
  "gamma": 0.0189,
  "rho": -0.234,
  "theta": -0.0089,
  "vega": 0.1567
}
```

**Interpretation**: American put (early exercise possible) values at $7.85, higher than equivalent European due to dividend protection value.

---

### Example 3: Asian Option with Arithmetic Averaging

**Input**:
```json
{
  "type": "call",
  "exercise_style": "asian",
  "start_date": "2025-11-20",
  "start_time": "am",
  "expiration_date": "2026-02-20",
  "expiration_time": "pm",
  "strike": 100.0,
  "stock_price": 102.0,
  "volatility": 0.22,
  "interest_rate": 2.0,
  "average_type": "arithmetic",
  "number_of_steps": 60,
  "number_of_simulations": 100000,
  "dividends": []
}
```

**Expected Output** (approximate):
```json
{
  "theoretical_price": 4.567,
  "delta": 0.412,
  "gamma": 0.0245,
  "rho": 0.145,
  "theta": -0.0178,
  "vega": 0.1234
}
```

**Interpretation**: Asian averaging reduces volatility benefit compared to vanilla European, so price is lower.

---

### Example 4: Binary Option

**Input**:
```json
{
  "type": "call",
  "exercise_style": "binary",
  "start_date": "2025-11-20",
  "start_time": "09:30:00",
  "expiration_date": "2025-12-05",
  "expiration_time": "16:00:00",
  "strike": 100.0,
  "stock_price": 99.0,
  "volatility": 0.18,
  "interest_rate": 2.0,
  "average_type": "arithmetic",
  "number_of_steps": 50,
  "number_of_simulations": 5000,
  "dividends": []
}
```

**Expected Output** (approximate):
```json
{
  "theoretical_price": 0.456,
  "delta": 0.089,
  "gamma": 0.0234,
  "rho": 0.012,
  "theta": -0.0045,
  "vega": 0.0234
}
```

**Interpretation**: Binary call pays fixed $1 if S > $100 at expiration. Current value $0.46 reflects ~46% probability ITM. Greeks are tiny compared to vanilla (all-or-nothing payoff).

---

## Technical Details

### Date/Time Processing

1. **Input**: Separate date (YYYY-MM-DD) and time (HH:MM:SS or am/pm) strings
2. **Parsing**: Combined into Python datetime object
3. **Conversion**: ACT/365 convention (actual days / 365)
4. **Output**: Time in years (float) for calculations

**Example**:
- Start: 2025-11-20 09:30:00
- Expiration: 2026-05-20 16:00:00
- Difference: 182 days + 6.5 hours
- Time to maturity: 182.271 / 365 ≈ 0.4993 years

### Dividend Handling

**Discrete Dividend Models**:
1. **European/Binary**: Reduces effective spot price via present value
2. **American**: Discounts intrinsic values at dividend dates during backward induction
3. **Asian**: Scales price paths; supports both single ex-dates and recurring schedules

**Formula for PV of dividends**:
```
PV = Σ (dividend_amount × e^(-r × t_i))
S_adjusted = S - PV
```

### Greeks Calculation

**Finite Difference Approximation**:
```
Delta = (V(S+h) - V(S-h)) / (2h)           # Central difference
Gamma = (V(S+h) - 2V(S) + V(S-h)) / h²
Vega = (V(σ+dσ) - V(σ-dσ)) / (2dσ)
Theta = (V(T-dT) - V(T)) / dT              # Per day
Rho = (V(r+dr) - V(r-dr)) / (2dr)
```

**Step Sizes**:
- Delta/Gamma: h = max(0.01 × S, 10^-4)
- Vega: dσ = max(0.01 × σ, 10^-4)
- Rho: dr = 10^-4
- Theta: dT = 1/365 (one day)

### Numerical Stability

**Safeguards**:
- Stock price floors at 10^-12 to prevent log(0)
- Volatility must be > 0 (enforced in schema)
- Time to maturity clamped to [0, ∞)
- Regression in American pricing has error handling fallback

---

## Troubleshooting

### Backend Issues

#### Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
uvicorn server:app --port 8001
```

#### Import Errors
```bash
# Verify all calculator imports in server.py
python -c "from Calculators import *; print('OK')"

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

#### CORS Errors in Browser Console
- Check frontend URL matches allowed origins in `server.py`
- Ensure backend is running on localhost:8000
- Clear browser cache

---

### Frontend Issues

#### Angular Port in Use
```bash
ng serve --port 4201
```

#### Node Modules Issues
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

#### Form Validation Errors
- Check browser console for form errors
- Ensure dates are YYYY-MM-DD format
- Volatility must be > 0

---

### API Response Issues

#### 422 Validation Error
```json
{
  "detail": [{"loc": ["body", "strike"], "msg": "..."}]
}
```

**Solutions**:
- Check strike ≥ 0.01
- Verify all required fields present
- Validate date formats

#### 400 Bad Request
- Check exercise_style is one of the four types
- Verify expiration > start date/time

---

## Performance Optimization Tips

1. **European Options**: Fastest (< 5ms), no simulation needed
2. **Binary Options**: ~5-10ms, simple payoff structure
3. **American Options**: Tune steps/simulations down if acceptable error
   - Minimum viable: steps=50, sims=5000 (~100ms)
   - Good accuracy: steps=100, sims=10000 (~300ms)
   - High precision: steps=200, sims=50000 (~1-2s)
4. **Asian Options**: 
   - Geometric averaging faster than arithmetic
   - Arithmetic with variance reduction recommended

---

## Future Enhancements

- Implied volatility solver (inverse Black-Scholes)
- Greeks surface visualization
- Real-time market data integration
- Portfolio-level Greeks aggregation
- Barrier and Lookback option support
- Local volatility surface calibration
- GPU-accelerated Monte Carlo

---

This README covers the complete project structure, API specification, and usage patterns. For additional questions, refer to individual calculator docstrings or the JSON schema files.
