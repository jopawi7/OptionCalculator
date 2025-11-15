// app.ts (oder app.component.ts)
import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, FormGroup, FormArray, Validators } from '@angular/forms';
import { HttpClient, HttpClientModule } from '@angular/common/http';

@Component({
  selector: 'app-root',
  standalone: true,
  templateUrl: './app.html',       // <- wir nutzen dein bestehendes Template weiter
  styleUrls: ['./app.css'],        // <- plural, robust für alle Angular-Versionen
  imports: [
    CommonModule,
    ReactiveFormsModule,
    HttpClientModule
  ]
})
export class AppComponent {
  // Für dein Header-Animation-Binding [class.show]="ready()"
  ready = signal(false);

  // Hauptformular
  calculatorForm!: FormGroup;

  constructor(private fb: FormBuilder, private http: HttpClient) {
    this.calculatorForm = this.fb.group({
      type: ['PUT'],
      style: ['European'],
      startDate: [new Date().toISOString().split('T')[0], Validators.required], // yyyy-MM-dd
      startTime: [new Date().toTimeString().slice(0, 5), Validators.required],  // HH:mm
      expirationDate: ['', Validators.required],
      strike: [100.0, [Validators.required, Validators.min(0)]],
      stockPrice: [300, [Validators.required, Validators.min(0)]],
      volatility: [20.02, [Validators.required, Validators.min(0)]],            // % Eingabe
      interestRate: [1.5, [Validators.required]],                               // % Eingabe
      dividends: this.fb.array([])
    });

    setTimeout(() => this.ready.set(true), 200);
  }

  // --- Dividenden-Helpers ---
  get dividends(): FormArray {
    return this.calculatorForm.get('dividends') as FormArray;
  }

  addDividend(): void {
    this.dividends.push(
      this.fb.group({
        date: ['', Validators.required],
        amount: [0, [Validators.required, Validators.min(0)]]
      })
    );
  }

  removeDividend(index: number): void {
    this.dividends.removeAt(index);
  }

  // --- UI-Toggles (Buttons CALL/PUT etc.) ---
  formValue(controlName: string): any {
    return this.calculatorForm.get(controlName)?.value;
  }

  setFormValue(controlName: string, value: any): void {
    this.calculatorForm.get(controlName)?.setValue(value);
  }

  // --- Submit -> Backend ---
  onSubmit(): void {

    console.log('▶️ onSubmit() wurde getriggert');


     if (this.calculatorForm.invalid) {
    this.calculatorForm.markAllAsTouched();
    console.warn('❌ Formular ungültig:', this.calculatorForm.value);
    return;
  }


    if (this.calculatorForm.invalid) {
      this.calculatorForm.markAllAsTouched();
      return;
    }

    const v = this.calculatorForm.getRawValue();

    // Laufzeit in Jahren
    const start = new Date(`${v.startDate}T${v.startTime}:00`);
    const expiry = new Date(`${v.expirationDate}T00:00:00`);
    const T = Math.max(0, (expiry.getTime() - start.getTime()) / (365 * 24 * 60 * 60 * 1000));

    // % -> Dezimal
    const sigma = (v.volatility ?? 0) / 100;
    const r = (v.interestRate ?? 0) / 100;

    const payload = {
      type: v.type,
      style: v.style,
      startDateTime: start.toISOString(), // ggf. lokal senden, wenn Backend das erwartet
      expirationDate: v.expirationDate,
      strike: Number(v.strike),
      stockPrice: Number(v.stockPrice),
      volatility: sigma,
      interestRate: r,
      T,
      dividends: (v.dividends ?? []).map((d: any) => ({ date: d.date, amount: Number(d.amount) })),
    };

    this.http.post('http://127.0.0.1:8000/api/price', payload).subscribe({
      next: (res) => console.log('✅ Result from Python:', res),
      error: (err) => console.error('❌ Error from Python:', err),
    });
  }
}
