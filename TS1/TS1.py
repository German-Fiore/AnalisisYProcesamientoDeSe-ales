# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 21:15:40 2026

@author: German
"""
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# =====================================================================
# 1. CONFIGURACIÓN Y PARÁMETROS GENERALES
# =====================================================================
N = 1000          # Número de muestras requerido por el enunciado
f0 = 2000         # Frecuencia fundamental de las señales (2 kHz)

# Nyquist y puntos por período:
# Para tener al menos 10 puntos por período de una señal de 2 kHz,
# la frecuencia de muestreo fs debe ser como mínimo: 10 * f0 = 20 kHz.
fs = 20000        # Frecuencia de muestreo elegida (20 kHz)
Ts = 1.0 / fs     # Período de muestreo (50 microsegundos)
t = np.arange(N) * Ts  # Vector de tiempo continuo muestreado (segundos)

# =====================================================================
# 2. SÍNTESIS DE LAS SEÑALES EN EL TIEMPO
# =====================================================================

# Señal 1: Senoidal de 2 kHz (Amplitud A=1V => Potencia media = A^2/2 = 0.5 W)
x1 = np.sin(2 * np.pi * f0 * t)

# Señal 2: Cosenoidal con 2 W de potencia media y desfasada en pi/2 (Coseno puro)
# P = A^2 / 2 = 2 W => A^2 = 4 => Amplitud A = 2 V.
x2 = 2 * np.cos(2 * np.pi * f0 * t)

# Señal 3: Ruido Blanco Gaussiano, DC = 0V, Varianza (Potencia) = 0.1 W
# Para ruido de media cero, la potencia media es la varianza (sigma^2 = 0.1).
# Desviación estándar (escala) = sqrt(0.1)
x3 = np.random.normal(loc=0.0, scale=np.sqrt(0.1), size=N)

# Señal 4: Ruido Blanco Uniforme, DC = 0V, Varianza (Potencia) = 0.1 W
# Para distribución uniforme en el intervalo [-b, b], la varianza es b^2 / 3.
# b^2 / 3 = 0.1 => b = sqrt(0.3) ≈ 0.5477 V
limite_b = np.sqrt(0.3)
x4 = np.random.uniform(low=-limite_b, high=limite_b, size=N)

# Señal 5: Pulso rectangular (onda cuadrada) de 2 kHz, 1 W de potencia y duty cycle 50%
# Para una onda cuadrada simétrica entre -A y A, la potencia es A^2.
# P = A^2 = 1 W => Amplitud A = 1 V.
x5 = signal.square(2 * np.pi * f0 * t, duty=0.5)

# BONUS 1: Señal extra de scipy.signal (Dientes de Sierra - Sawtooth)
# Amplitud A = 1 V => Potencia teórica media = A^2 / 3 ≈ 0.333 W
x_bonus = signal.sawtooth(2 * np.pi * f0 * t)

# Empaquetamos todo en una estructura para procesar en bucle de forma elegante
senales = [
    {"nombre": "Seno (2 kHz, A=1)", "datos": x1, "teorica": 0.5},
    {"nombre": "Coseno (2 kHz, A=2, P=2W)", "datos": x2, "teorica": 2.0},
    {"nombre": "Ruido Gaussiano (P=0.1W)", "datos": x3, "teorica": 0.1},
    {"nombre": "Ruido Uniforme (P=0.1W)", "datos": x4, "teorica": 0.1},
    {"nombre": "Pulso Cuadrado (2 kHz, P=1W)", "datos": x5, "teorica": 1.0},
    {"nombre": "Bonus: Dientes de Sierra (2 kHz)", "datos": x_bonus, "teorica": 0.333}
]

# =====================================================================
# 3. VERIFICACIÓN NUMÉRICA DEL TEOREMA DE PARSEVAL
# =====================================================================
print("=" * 85)
print(f"{'Señal Analizada':<32} | {'P_Tiempo (W)':<14} | {'P_Frecuencia (W)':<15} | {'Diferencia (W)':<12}")
print("=" * 85)

for s in senales:
    x = s["datos"]
    
    # Potencia media en el tiempo: promedio de las muestras al cuadrado
    pot_tiempo = np.mean(np.abs(x)**2)
    
    # DFT usando el algoritmo de la FFT (obtenemos los coeficientes complejos X[k])
    X = np.fft.fft(x)
    
    # Potencia media en frecuencia (Teorema de Parseval): suma(|X[k]|^2) / N^2
    pot_frecuencia = np.sum(np.abs(X)**2) / (N**2)
    
    diff = np.abs(pot_tiempo - pot_frecuencia)
    print(f"{s['nombre']:<32} | {pot_tiempo:.6f}       | {pot_frecuencia:.6f}          | {diff:.2e}")
print("=" * 85)

# =====================================================================
# 4. GRAFICACIÓN DE RESULTADOS (Tiempo vs Frecuencia)
# =====================================================================
fig, axs = plt.subplots(len(senales), 2, figsize=(14, 16))
freqs = np.fft.rfftfreq(N, d=Ts)  # Eje de frecuencias unilateral (Hz)

for i, s in enumerate(senales):
    x = s["datos"]
    
    # Cantidad de muestras a mostrar en el tiempo para que sea estético
    muestras_vis = 100 if "Ruido" in s["nombre"] else 30
    
    # A. Gráfico en el tiempo usando PLOT (Línea continua)
    ax_t = axs[i, 0]
    ax_t.plot(t[:muestras_vis] * 1000, x[:muestras_vis], color='C0', linewidth=1.8)
    
    # Forzamos una línea negra sólida cruzando el origen y=0 en todo el eje X
    ax_t.axhline(0, color='black', linewidth=1.2, zorder=3)
    
    ax_t.set_title(f"{s['nombre']} - Dominio del Tiempo", fontsize=10, fontweight='bold')
    ax_t.set_ylabel("Amplitud [V]")
    ax_t.grid(True, linestyle=":", alpha=0.6)
    
    # B. Gráfico en frecuencia usando PLOT
    ax_f = axs[i, 1]
    X_unilateral = np.fft.rfft(x) / N
    mag = np.abs(X_unilateral)
    mag[1:-1] = 2 * mag[1:-1]  # Conservar energía unilateral
    
    ax_f.plot(freqs / 1000, mag, color="crimson", linewidth=1.5)
    
    # Forzamos la línea negra de referencia horizontal en y=0 en frecuencia
    ax_f.axhline(0, color='black', linewidth=1.2, zorder=3)
    
    ax_f.set_title("Módulo de la FFT (Espectro Unilateral)", fontsize=10, fontweight='bold')
    ax_f.set_ylabel("Magnitud Normalizada")
    ax_f.grid(True, linestyle=":", alpha=0.6)
    ax_f.set_xlim(0, fs / (2 * 1000))  # Límite en la frecuencia de Nyquist (10 kHz)

# Configuración de etiquetas finales de los ejes X
axs[-1, 0].set_xlabel("Tiempo [ms]")
axs[-1, 1].set_xlabel("Frecuencia [kHz]")

plt.tight_layout()
plt.show()

