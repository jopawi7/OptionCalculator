import { Component, signal, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, FormGroup, FormArray, Validators } from '@angular/forms';
import { HttpClient, HttpClientModule } from '@angular/common/http';

@Component({
  selector: 'app-root',
  standalone: true,
  templateUrl: './app.html',
  styleUrls: ['./app.css'],
  imports: [CommonModule, ReactiveFormsModule, HttpClientModule]
})
export class AppComponent {
  ready = signal(false);
  calculatorForm!: FormGroup;

  theoretical_price: number | null = null;
  delta: number | null = null;
  gamma: number | null = null;
  rho: number | null = null;
  theta: number | null = null;
  vega: number | null = null;

  constructor(
    private fb: FormBuilder,
    private http: HttpClient,
    private cdr: ChangeDetectorRef  // Inject ChangeDetectorRef
  ) {
    this.calculatorForm = this.fb.group({
      type: ['PUT'],
      style: ['European'],
      startDate: [new Date().toISOString().split('T')[0], Validators.required],
      startTime: [new Date().toTimeString().slice(0, 5), Validators.required],
      expirationDate: ['', Validators.required],
      expirationTime: ['AM', Validators.required],
      strike: [100.0, [Validators.required, Validators.min(0)]],
      stockPrice: [300, [Validators.required, Validators.min(0)]],
      volatility: [20.02, [Validators.required, Validators.min(0)]],
      interestRate: [1.5, [Validators.required]],
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

  onSubmit(): void {
    if (this.calculatorForm.invalid) {
      this.calculatorForm.markAllAsTouched();
      return;
    }

    const v = this.calculatorForm.getRawValue();

    const payload = {
      type: v.type.toLowerCase(),
      exercise_style: v.style.toLowerCase(),
      start_date: v.startDate,
      start_time: v.startTime,
      expiration_date: v.expirationDate,
      expiration_time: v.expirationTime,
      strike: Number(v.strike),
      stock_price: Number(v.stockPrice),
      volatility: Number(v.volatility),
      interest_rate: Number(v.interestRate),
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
        this.cdr.detectChanges();  // Manuell Change Detection triggern
      },
      error: (err) => {
        console.error('Error from backend:', err);
      }
    });
  }
}
