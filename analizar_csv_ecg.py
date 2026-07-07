# -*- coding: utf-8 -*-
"""
Análisis de señal ECG desde CSV local
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("ANÁLISIS DE SEÑAL ECG DESDE CSV LOCAL")
print("=" * 60)

# Cargar el CSV
csv_path = "ecg_live_capture_20260706_181344.csv"
print(f"\nCargando: {csv_path}")

try:
    df = pd.read_csv(csv_path)
    print(f"✓ Datos cargados exitosamente")
    print(f"  Muestras: {len(df)}")
    print(f"  Columnas: {list(df.columns)}")
except Exception as e:
    print(f"✗ Error cargando archivo: {e}")
    exit(1)

# Información de la señal
print("\n" + "=" * 60)
print("INFORMACIÓN DE LA SEÑAL")
print("=" * 60)

# Calcular parámetros de muestreo
tiempo = df['tiempo_s'].values
adc_raw = df['adc_raw'].values
ecg_filtrado = df['ecg_filtrado'].values

# Frecuencia de muestreo
dt = np.mean(np.diff(tiempo))
fs = 1.0 / dt
duracion = tiempo[-1] - tiempo[0]

print(f"Duración total: {duracion:.2f} segundos ({duracion/60:.2f} minutos)")
print(f"Número de muestras: {len(tiempo)}")
print(f"Frecuencia de muestreo: {fs:.2f} Hz")
print(f"Intervalo entre muestras: {dt*1000:.3f} ms")

# Estadísticas de la señal raw
print("\nEstadísticas de señal raw (ADC):")
print(f"  Mín: {np.min(adc_raw):.0f}")
print(f"  Máx: {np.max(adc_raw):.0f}")
print(f"  Media: {np.mean(adc_raw):.0f}")
print(f"  Desv. Est.: {np.std(adc_raw):.2f}")

# Estadísticas de la señal filtrada
print("\nEstadísticas de señal filtrada:")
print(f"  Mín: {np.min(ecg_filtrado):.2f}")
print(f"  Máx: {np.max(ecg_filtrado):.2f}")
print(f"  Media: {np.mean(ecg_filtrado):.2f}")
print(f"  Desv. Est.: {np.std(ecg_filtrado):.2f}")

# Análisis de frecuencia (FFT)
print("\n" + "=" * 60)
print("ANÁLISIS EN FRECUENCIA")
print("=" * 60)

# Calcular espectro de potencia
N = len(ecg_filtrado)
freqs = np.fft.rfftfreq(N, dt)
fft_vals = np.abs(np.fft.rfft(ecg_filtrado - np.mean(ecg_filtrado)))
fft_power = (fft_vals ** 2) / N

# Encontrar picos en el espectro
peak_indices = signal.find_peaks(fft_power, height=np.max(fft_power) * 0.1)[0]
peak_freqs = freqs[peak_indices]
peak_powers = fft_power[peak_indices]

# Ordenar por potencia descendente
sorted_idx = np.argsort(peak_powers)[::-1]
print("\nPicos principales en el espectro (frecuencias con mayor energía):")
for i in range(min(5, len(sorted_idx))):
    idx = sorted_idx[i]
    print(f"  {i+1}. {peak_freqs[idx]:.2f} Hz (potencia: {peak_powers[idx]:.2e})")

# Detección de picos (latidos)
print("\n" + "=" * 60)
print("DETECCIÓN DE LATIDOS")
print("=" * 60)

# Detectar picos R en la señal filtrada
signal_normalized = (ecg_filtrado - np.mean(ecg_filtrado)) / np.std(ecg_filtrado)
threshold = np.mean(signal_normalized) + 1.5 * np.std(signal_normalized)
peaks, properties = signal.find_peaks(signal_normalized, height=threshold, distance=int(fs * 0.3))

print(f"Latidos detectados: {len(peaks)}")

if len(peaks) > 0:
    # Calcular intervalos RR
    peak_times = tiempo[peaks]
    rr_intervals = np.diff(peak_times)
    
    print(f"\nIntervalo RR:")
    print(f"  Mín: {np.min(rr_intervals):.3f} s")
    print(f"  Máx: {np.max(rr_intervals):.3f} s")
    print(f"  Media: {np.mean(rr_intervals):.3f} s")
    print(f"  Desv. Est.: {np.std(rr_intervals):.3f} s")
    
    # BPM estimado
    bpm_values = 60.0 / rr_intervals
    print(f"\nFrecuencia cardíaca (BPM):")
    print(f"  Mín: {np.min(bpm_values):.1f} bpm")
    print(f"  Máx: {np.max(bpm_values):.1f} bpm")
    print(f"  Media: {np.mean(bpm_values):.1f} bpm")
    print(f"  Desv. Est.: {np.std(bpm_values):.1f} bpm")

# Análisis variabilidad de frecuencia cardíaca (HRV)
print("\n" + "=" * 60)
print("ANÁLISIS DE VARIABILIDAD DE FRECUENCIA CARDÍACA (HRV)")
print("=" * 60)

if len(peaks) > 2:
    # SDNN (Standard Deviation of NN intervals)
    sdnn = np.std(rr_intervals)
    print(f"SDNN (Desv. Est. de intervalos NN): {sdnn*1000:.1f} ms")
    
    # RMSSD (Root Mean Square of Successive Differences)
    successive_diffs = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(successive_diffs ** 2))
    print(f"RMSSD (Raíz media de dif. sucesivas): {rmssd*1000:.1f} ms")
    
    # pNN50 (Percentage of NN intervals differing by more than 50 ms)
    pnn50 = 100.0 * np.sum(np.abs(successive_diffs) > 0.05) / len(successive_diffs)
    print(f"pNN50 (% intervalos que difieren >50ms): {pnn50:.1f}%")

# Generar gráficos
print("\n" + "=" * 60)
print("GENERANDO GRÁFICOS")
print("=" * 60)

fig, axes = plt.subplots(4, 1, figsize=(14, 10))

# 1. Señal completa
ax = axes[0]
ax.plot(tiempo, adc_raw, label='Señal RAW (ADC)', alpha=0.7, linewidth=0.5)
ax.plot(tiempo, ecg_filtrado, label='Señal Filtrada', alpha=0.7, linewidth=0.5)
if len(peaks) > 0:
    ax.scatter(tiempo[peaks], ecg_filtrado[peaks], color='red', s=50, label='Latidos detectados', zorder=5)
ax.set_xlabel('Tiempo (s)')
ax.set_ylabel('Amplitud')
ax.set_title('Señal ECG Completa', fontweight='bold', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Zoom en los primeros 10 segundos
ax = axes[1]
mask = tiempo <= 10
ax.plot(tiempo[mask], ecg_filtrado[mask], linewidth=0.8, label='Señal filtrada')
peaks_mask = peaks[tiempo[peaks] <= 10]
if len(peaks_mask) > 0:
    ax.scatter(tiempo[peaks_mask], ecg_filtrado[peaks_mask], color='red', s=80, label='Picos R')
ax.set_xlabel('Tiempo (s)')
ax.set_ylabel('Amplitud')
ax.set_title('Primeros 10 segundos (Detalle)', fontweight='bold', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Espectro de potencia
ax = axes[2]
ax.semilogy(freqs, fft_power)
ax.set_xlabel('Frecuencia (Hz)')
ax.set_ylabel('Potencia (V²/Hz)')
ax.set_title('Espectro de Potencia', fontweight='bold', fontsize=12)
ax.set_xlim(0, min(50, fs/2))
ax.grid(True, alpha=0.3)

# 4. Variabilidad RR
ax = axes[3]
if len(peaks) > 1:
    ax.plot(range(len(rr_intervals)), rr_intervals * 1000, 'b.-', label='Intervalos RR')
    ax.axhline(np.mean(rr_intervals) * 1000, color='r', linestyle='--', label='Media')
    ax.fill_between(range(len(rr_intervals)), 
                     (np.mean(rr_intervals) - np.std(rr_intervals)) * 1000,
                     (np.mean(rr_intervals) + np.std(rr_intervals)) * 1000,
                     alpha=0.2, color='red', label='±1 Desv. Est.')
    ax.set_xlabel('Número de latido')
    ax.set_ylabel('Intervalo RR (ms)')
    ax.set_title('Variabilidad de Intervalos RR', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
output_fig = "analisis_ecg.png"
plt.savefig(output_fig, dpi=150, bbox_inches='tight')
print(f"Gráfico guardado: {output_fig}")

# Generar gráfico de distribución de BPM
if len(peaks) > 1:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(bpm_values, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    ax.axvline(np.mean(bpm_values), color='red', linestyle='--', linewidth=2, label=f'Media: {np.mean(bpm_values):.1f} bpm')
    ax.axvline(np.median(bpm_values), color='green', linestyle='--', linewidth=2, label=f'Mediana: {np.median(bpm_values):.1f} bpm')
    ax.set_xlabel('Frecuencia Cardiaca (BPM)')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Distribución de Frecuencia Cardiaca', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    output_fig2 = "distribucion_bpm.png"
    plt.savefig(output_fig2, dpi=150, bbox_inches='tight')
    print(f"Gráfico guardado: {output_fig2}")

print("\n" + "=" * 60)
print("ANALISIS COMPLETADO EXITOSAMENTE")
print("=" * 60)
