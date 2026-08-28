import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import stats
# =====================================================================
# 1. PARÁMETROS Y GENERACIÓN DE SEÑAL + RUIDO
# =====================================================================
fs = 1000          # Frecuencia de muestreo (Hz)
N = 1000           # Cantidad de muestras
t = np.arange(N) / fs

# Señal senoidal pura (Vmax = sqrt(2) V => Potencia = 1 W)
vmax = np.sqrt(2)
ff = 3             # 3 Hz
x = vmax * np.sin(2 * np.pi * ff * t)

# Ruido aditivo (SNR = 40 dB)
SNR = 40
Psen = (vmax**2) / 2
Pruido = Psen / (10**(SNR / 10))
ruido = np.random.normal(0, np.sqrt(Pruido), N)

# Señal ruidosa
noisy_x = x + ruido

# =====================================================================
# 2. CUANTIZACIÓN Y ERROR DE CUANTIZACIÓN (nq)
# =====================================================================
B = 8                           # ADC de 8 bits
Vref = 2 * vmax                 # Rango dinámico pico a pico
qq = Vref / (2**B)              # Tamaño del escalón del ADC (LSB)

# Proceso de cuantización
x_q = np.round(noisy_x / qq) * qq

# Error de cuantización
nq = x_q - noisy_x

# =====================================================================
# 3. VISUALIZACIÓN DE PROPIEDADES DE nq
# =====================================================================
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# A. Error de Cuantización en el Tiempo
axs[0, 0].plot(nq, label=r'$n_q$', color='tab:blue', alpha=0.7)
axs[0, 0].axhline(qq/2, color='r', linestyle='--', label=r'$+q/2$')
axs[0, 0].axhline(-qq/2, color='r', linestyle='--', label=r'$-q/2$')
axs[0, 0].set_title("1. Error de Cuantización $n_q[n]$ en el tiempo")
axs[0, 0].set_ylabel("Error [V]")
axs[0, 0].legend()
axs[0, 0].grid(True)

# B. Histograma (Demostración de Distribución Uniforme)
axs[0, 1].hist(nq, bins=30, density=True, color='tab:green', edgecolor='black', alpha=0.7)
axs[0, 1].axhline(1/qq, color='r', linestyle='--', linewidth=2, label=r'PDF Teórica ($1/q$)')
axs[0, 1].set_title("2. Histograma de $n_q$ (Distribución Uniforme)")
axs[0, 1].set_xlabel("Amplitud del Error [V]")
axs[0, 1].set_ylabel("Densidad de Probabilidad")
axs[0, 1].legend()
axs[0, 1].grid(True)

# C. Autocorrelación de nq (Demostración de Incorrelación)
autocorr = signal.correlate(nq, nq, mode='full')
lags = signal.correlation_lags(len(nq), len(nq))
# Normalización por la energía en el origen (lag 0)
autocorr_norm = autocorr / np.max(autocorr)

axs[1, 0].plot(lags, autocorr_norm, color='tab:purple')
axs[1, 0].set_title("3. Autocorrelación de $n_q$ (Delta de Dirac = Incorrelado)")
axs[1, 0].set_xlabel("Lag (Muestras de retardo)")
axs[1, 0].set_ylabel("Autocorrelación Normalizada")
axs[1, 0].set_xlim(-100, 100)  # Zoom al centro para ver el impulso
axs[1, 0].grid(True)

# D. Correlación cruzada entre la Señal x[n] y el Error nq[n]
crosscorr = signal.correlate(x, nq, mode='full') / (np.std(x) * np.std(nq) * N)

axs[1, 1].plot(lags, crosscorr, color='tab:orange')
axs[1, 1].set_title("4. Correlación cruzada entre $x[n]$ y $n_q[n]$ (~0)")
axs[1, 1].set_xlabel("Lag (Muestras de retardo)")
axs[1, 1].set_ylabel("Coeficiente de Correlación")
axs[1, 1].set_ylim(-0.2, 0.2)
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()

# =====================================================================
# VERIFICACIÓN ESTADÍSTICA DE nq (P-VALORES Y VARIANZA)
# =====================================================================

# 1. Comparación de Varianzas (Teórica vs Empírica)
var_teorica = (qq**2) / 12
var_empirica = np.var(nq)

print("=" * 65)
print("  ANÁLISIS ESTADÍSTICO DEL ERROR DE CUANTIZACIÓN (nq)")
print("=" * 65)
print(f"Varianza Teórica (q^2 / 12) : {var_teorica:.6e}")
print(f"Varianza Empírica (np.var)  : {var_empirica:.6e}")
print("-" * 65)

# 2. Test Kolmogorov-Smirnov (Distribución Uniforme)
loc_teorico = -qq / 2
scale_teorica = qq  # Ancho del intervalo: q/2 - (-q/2) = q

stat_ks, p_val_ks = stats.kstest(nq, 'uniform', args=(loc_teorico, scale_teorica))

print(f"1. Test KS (Uniformidad) -> Estadístico: {stat_ks:.4f} | p-valor: {p_val_ks:.4e}")
if p_val_ks > 0.05:
    print("   -> CONCLUSIÓN: p > 0.05. Aceptamos que nq sigue una distribución UNIFORME.")
else:
    print("   -> CONCLUSIÓN: p <= 0.05. No se ajusta a una uniforme.")

print("-" * 65)

# 3. Test de Incorrelación sobre las muestras adyacentes
# Calculamos la correlación entre nq[n] y nq[n+1] (lag 1)
r_lag1, p_val_corr = stats.pearsonr(nq[:-1], nq[1:])

print(f"2. Correlación Lag 1     -> Coeficiente: {r_lag1:.4f} | p-valor: {p_val_corr:.4e}")
if p_val_corr > 0.05:
    print("   -> CONCLUSIÓN: p > 0.05. No hay correlación entre muestras (INCORRELADO).")
else:
    print("   -> CONCLUSIÓN: Hay correlación lineal entre muestras.")
print("=" * 65)