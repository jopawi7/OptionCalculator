# Binary Option Calculator - Documentation

## Overview

This calculator computes the theoretical price and Greeks for binary (cash-or-nothing) options using the Black-Scholes model.

## Data Flow

```
Input Data → Normalization → Black-Scholes Calculation → Output Dictionary
```

## 1. Input Data Sources

The calculator accepts input from two sources:

**JSON File Input**
- Reads from `../Input/input.json` (relative to script location)
- Structured JSON format with all parameters
- Automatically normalizes dividend data

**Manual CLI Input**
- User prompted for each parameter interactively
- Real-time input via terminal

### Input Data Structure

The calculator expects these parameters:

```
type: string ("call" or "put")
start_date: string (format: "YYYY-MM-DD")
expiration_date: string (format: "YYYY-MM-DD")
strike: number (strike price K)
stock_price: number (current spot price S)
volatility: number (annualized volatility σ, as decimal)
interest_rate: number (risk-free rate r, as decimal)
dividends: number or array (dividend yield q, as decimal)
```

### Data Normalization Process

**Dividends:**
- If array of objects: calculates average of all "amount" fields
- If single number: uses value directly
- If missing or invalid: defaults to 0.0

**Option Type:**
- Converts to lowercase
- Must be "call" or "put"

## 2. Calculation Process

### Black-Scholes Formula Application

**Parameters Used:**
- S = spot price
- K = strike price
- σ = volatility
- r = risk-free rate
- q = dividend yield
- T = time to maturity (in years)
- Q = payoff amount (fixed at 1.0)

**Time Calculation:**
```
T = (expiration_date - start_date) / 365.0
```

**d2 Parameter:**
```
d2 = [ln(S/K) + (r - q - 0.5σ²)T] / (σ√T)
```

**Greeks Calculated:**

For Binary Call:
- Theoretical Price: Q · e^(-rT) · N(d2)
- Delta: rate of change with respect to spot price
- Gamma: rate of change of delta
- Vega: sensitivity to volatility
- Rho: sensitivity to interest rate
- Theta: time decay

For Binary Put:
- Same formulas with N(-d2) and adjusted signs

All values rounded to 10 decimal places.

## 3. Output Format

### Return Type: Python Dictionary

The function returns a dictionary (key-value pairs):

```python
{
    "theoretical_price": 0.4512345678,
    "delta": 0.0123456789,
    "gamma": -0.0045678901,
    "rho": 0.1234567890,
    "theta": -0.0567890123,
    "vega": -0.0234567890
}
```

**Data Structure:** Dictionary (hash map)
**Keys:** 6 strings (metric names)
**Values:** floats (10 decimal precision)

### Console Output

When run as CLI, results print to terminal:
```
theoretical_price: 0.4512345678
delta: 0.0123456789
gamma: -0.0045678901
rho: 0.1234567890
theta: -0.0567890123
vega: -0.0234567890
```

## 4. Error Handling

### Input Errors

**Missing File**
```
When: JSON file path doesn't exist
Action: Raises FileNotFoundError
Result: Program exits with code 1
```

**Invalid Option Type**
```
When: type is not "call" or "put"
Action: Raises ValueError
Result: Calculation stops, error message displayed
```

**Invalid Date Range**
```
When: Expiration date is before or equal to start date
Action: Raises ValueError
Result: Calculation stops, error message displayed
```

### Data Processing Errors

**JSON Parsing Issues**
```
When: Malformed JSON or missing required fields
Action: Exception caught in main execution
Display: "Error loading JSON: {details}"
Result: Program exits with code 1
```

**Mathematical Errors**
```
When: Invalid mathematical operations (log of negative, division by zero)
Action: Exception caught in main execution
Display: "Calculation error: {details}"
Result: No output generated, error displayed
```

## 5. Execution Flow

```
Program Start
    ↓
User selects input method (JSON or Manual)
    ↓
Data loaded and normalized
    ↓
Option type validated
    ↓
Time to maturity computed
    ↓
d2 parameter calculated
    ↓
Price and Greeks computed
    ↓
Results rounded to 10 decimals
    ↓
Dictionary returned / printed to console
    ↓
Program End
```

## 6. Function Reference

### `N(x)` - Normal CDF
- Computes cumulative distribution function of standard normal
- Uses error function (erf) implementation
- Returns probability value between 0 and 1

### `n(x)` - Normal PDF
- Computes probability density function of standard normal
- Used in Greeks calculation
- Returns density value

### `read_inputs_from_file(filename=None)`
- Reads JSON from specified path or default location
- Normalizes dividend data structure
- Returns dictionary of input parameters

### `calculate_option_value(data)`
- Main calculation function
- Takes normalized input dictionary
- Returns dictionary with 6 metrics (price + 5 Greeks)

## 7. Platform Notes

- Works on Windows, macOS, and Linux
- Uses `os.path.normpath()` for cross-platform file paths
- Date format must be ISO standard: YYYY-MM-DD
- All rates and volatility in decimal format (0.05 = 5%)
