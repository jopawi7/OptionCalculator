import { Component, signal, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, FormGroup, FormArray, Validators } from '@angular/forms';
import { HttpClient, HttpClientModule, HttpErrorResponse } from '@angular/common/http';

@Component({
  selector: 'app-root',
  templateUrl: './app.html',
  styleUrls: ['./app.css'],
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule, HttpClientModule]
})
export class AppComponent {
  ready = signal(false);
  calculatorForm!: FormGroup;
  isLoading = false;

  // Add this signal for error state
  errorMessage = signal<string | null>(null);

  theoretical_price: number | null = null;
  delta: number | null = null;
  gamma: number | null = null;
  rho: number | null = null;
  theta: number | null = null;
  vega: number | null = null;

  constructor(
    private fb: FormBuilder,
    private http: HttpClient,
    private cdr: ChangeDetectorRef
  ) {
    this.calculatorForm = this.fb.group({
      type: ['CALL', [Validators.required, Validators.pattern(/^(call|put)$/i)]],
      style: ['EUROPEAN', [Validators.required, Validators.pattern(/^(american|european|asian|binary)$/i)]],
      startDate: [new Date().toISOString().split('T')[0], [Validators.required, Validators.pattern(/^\d{4}-\d{2}-\d{2}$/)]],
      startTime: [new Date().toTimeString().slice(0, 8), [Validators.required, Validators.pattern(/^([01]\d|2[0-3]):[0-5]\d:[0-5]\d$/)]],
      expirationDate: ['', [Validators.required, Validators.pattern(/^\d{4}-\d{2}-\d{2}$/)]],
      expirationTime: ['09:00:00', [Validators.required, Validators.pattern(/^([01]\d|2[0-3]):[0-5]\d:[0-5]\d$/)]],
      strike: [100.0, [Validators.required, Validators.min(0.01)]],
      stockPrice: [120, [Validators.required, Validators.min(0.01)]],
      volatility: [20, [Validators.required, Validators.min(0.0000001)]],
      interestRate: [5, [Validators.required]],
      number_of_steps: [500, [Validators.required, Validators.min(1), Validators.max(1000)]],
      number_of_simulations: [10000, [Validators.required, Validators.min(1), Validators.max(10000)]],
      average_type: ['arithmetic', [Validators.required, Validators.pattern(/^(arithmetic|geometric)$/i)]],
      binary_payoff_structure: ['cash', [Validators.required]],
      binary_payout: [1, [Validators.required, Validators.min(0.01)]],
      dividends: this.fb.array([])
    });

    setTimeout(() => this.ready.set(true), 200);
  }

  get dividends(): FormArray {
    return this.calculatorForm.get('dividends') as FormArray;
  }

  addDividend(): void {
    this.dividends.push(this.fb.group({
      date: ['', Validators.required],
      amount: [0, [Validators.required, Validators.min(0)]]
    }));
  }

  removeDividend(index: number): void {
    this.dividends.removeAt(index);
  }

  formValue(controlName: string): any {
    return this.calculatorForm.get(controlName)?.value;
  }

  setFormValue(controlName: string, value: any): void {
    this.calculatorForm.get(controlName)?.setValue(value);
  }

  onBinaryPayoffStructureChange(value: string): void {
    if (value === 'cash') {
      this.setFormValue('binary_payout', 1);
    } else if (value === 'asset') {
      // No change to binary_payout
    }
    // For 'custom', the value remains editable
    this.cdr.detectChanges();
  }

  // Add this method to clear errors
  clearError(): void {
    this.errorMessage.set(null);
  }

  onSubmit(): void {
    // Clear previous errors
    this.clearError();

    if (this.calculatorForm.invalid) {
      this.calculatorForm.markAllAsTouched();
      return;
    }

    this.isLoading = true;
    this.cdr.detectChanges();
    const v = this.calculatorForm.getRawValue();
    const payload: any = {
      type: v.type.toLowerCase(),
      exercise_style: v.style.toLowerCase(),
      start_date: v.startDate,
      start_time: v.startTime,
      expiration_date: v.expirationDate,
      expiration_time: v.expirationTime,
      strike: Number(v.strike),
      stock_price: Number(v.stockPrice),
      volatility: Number(v.volatility) / 100,
      interest_rate: Number(v.interestRate),
      average_type: v.average_type.toLowerCase(),
      number_of_steps: v.number_of_steps,
      number_of_simulations: v.number_of_simulations,
      binary_payoff_structure: v.binary_payoff_structure.toLowerCase(),
      binary_payout: Number(v.binary_payout),
      dividends: (v.dividends ?? []).map((d: any) => ({ date: d.date, amount: Number(d.amount) }))
    };

    this.http.post('http://127.0.0.1:8000/api/price', payload).subscribe({
      next: (res: any) => {
        this.theoretical_price = res.theoretical_price;
        this.delta = res.delta;
        this.gamma = res.gamma;
        this.rho = res.rho;
        this.theta = res.theta;
        this.vega = res.vega;
        this.isLoading = false;
        this.cdr.detectChanges();
      },
      error: (err: HttpErrorResponse) => {
        console.error('Error from backend:', err);

        // Extract error message from response
        let message = 'An error occurred. Please check your input.';

        if (err.error && typeof err.error === 'object') {
          if (err.error.detail) {
            // Handle FastAPI error format
            message = err.error.detail;
          } else if (err.error.message) {
            message = err.error.message;
          }
        } else if (err.statusText) {
          message = `${err.status}: ${err.statusText}`;
        }

        // Set the error message signal
        this.errorMessage.set(message);

        this.isLoading = false;
        this.cdr.detectChanges();
      }
    });
  }
}
