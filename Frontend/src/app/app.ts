import { Component, signal, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, FormGroup, FormArray, Validators, AbstractControl, ValidationErrors } from '@angular/forms';
import { HttpClient, HttpClientModule } from '@angular/common/http';

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
      expirationTime: ['', [Validators.required, Validators.pattern(/^([01]\d|2[0-3]):[0-5]\d:[0-5]\d$/)]],
      strike: [100.0, [Validators.required, Validators.min(0.01)]],
      stockPrice: [120, [Validators.required, Validators.min(0.01)]],
      volatility: [20, [Validators.required, Validators.min(0.0000001)]],
      interestRate: [1.5, [Validators.required]],
      number_of_steps: [100, [Validators.required, Validators.min(1), Validators.max(1000)]],
      number_of_simulations: [10000, [Validators.required, Validators.min(1), Validators.max(100000)]],
      average_type: ['arithmetic', [Validators.required, Validators.pattern(/^(arithmetic|geometric)$/i)]],
      binary_payoff_structure: ['cash', [Validators.required]],
      binary_payout: [1, [Validators.required, Validators.min(0.01)]],
      dividends: this.fb.array([])
    }, { validators: this.dateTimeRangeValidator });

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
      // Keine Änderung an binary_payout
    }
    // Bei 'custom' bleibt der Wert editierbar
    this.cdr.detectChanges();
  }

  onSubmit(): void {
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
      error: (err) => {
        console.error('Error from backend:', err);
        this.isLoading = false;
        this.cdr.detectChanges();
      }
    });
  }

  // Validator to ensure expiration datetime is not before start datetime
  dateTimeRangeValidator(group: AbstractControl): ValidationErrors | null {
    const startDate = group.get('startDate')?.value;
    const startTime = group.get('startTime')?.value;
    const expirationDate = group.get('expirationDate')?.value;
    const expirationTime = group.get('expirationTime')?.value;

    if (!startDate || !startTime || !expirationDate || !expirationTime) {
      return null; // Validation only when all values are present
    }

    const startDateTime = new Date(`${startDate}T${startTime}`);
    const expirationDateTime = new Date(`${expirationDate}T${expirationTime}`);

    if (expirationDateTime < startDateTime) {
      return { expirationBeforeStart: true };
    }
    return null;
  }

}
