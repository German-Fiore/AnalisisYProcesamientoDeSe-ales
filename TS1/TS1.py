import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


# PARÁMETROS GENERALES 
N = 1000          # Número de muestras 
f0 = 2000         # Frecuencia de las señales (2 kHz)

# Elegimos fs = 20 kHz para cumplir al menos 10 puntos por período
# (fs / f0 = 20000 / 2000 = exactamente 10 puntos por ciclo)
fs = 20000        
t = np.arange(N) / fs


# SÍNTESIS DE LAS 5 SEÑALES


# Señal 1: Senoidal de 2 kHz (Amplitud de 1 V)
x1 = np.sin(2 * np.pi * f0 * t)

# Señal 2: Seno desfasado pi/2 con potencia media de 2W y Amplitud A = 2 V
x2 = 2 * np.sin(2 * np.pi * f0 * t + np.pi / 2)

# Señal 3: Ruido normalmente distribuido (Gaussiano), DC = 0V, Varianza = 0.1 W
# Para el ruido, Potencia = Varianza (sigma^2). La desviación estándar es sqrt(0.1)
std_gauss = np.sqrt(0.1)
x3 = np.random.normal(0, std_gauss, N)

# Señal 4: Ruido uniformemente distribuido, DC = 0V, Varianza = 0.1 W
# Para distribución uniforme en [-b, b], Varianza = b^2 / 3 = 0.1 -> b = sqrt(0.3)
limite_b = np.sqrt(0.3)
x4 = np.random.uniform(-limite_b, limite_b, N)

# Señal 5: Pulso rectangular (onda cuadrada) de 2 kHz, Potencia = 1 W, Ciclo de actividad = 50%
# Potencia = A^2 = 1W -> Amplitud = 1 V
x5 = signal.square(2 * np.pi * f0 * t, duty=0.5)

# Guardamos las señales para graficar
seniales = [
    {"nombre": "Seno (2 kHz)", "datos": x1},
    {"nombre": "Seno Desfasado (2 kHz, P = 2W)", "datos": x2},
    {"nombre": "Ruido Gaussiano (P = 0.1W)", "datos": x3},
    {"nombre": "Ruido Uniforme (P = 0.1W)", "datos": x4},
    {"nombre": "Onda Cuadrada (2 kHz, P = 1W)", "datos": x5}
]


# 3. GRAFICOS (Tiempo y Espectro FFT)

# Creamos una cuadrícula de 5 filas por 2 columnas
fig, axs = plt.subplots(5, 2, figsize=(12, 15))

for i, s in enumerate(seniales):
    x = s["datos"]
    
    # Columna 1: Dominio del Tiempo
    # Para ver la forma de las señales periódicas mostramos 30 muestras (3 ciclos).
    # Para los ruidos mostramos 100 muestras para que se aprecie la aleatoriedad.
    muestras_vis = 100 if "Ruido" in s["nombre"] else 30
    
    axs[i, 0].plot(t[:muestras_vis] * 1000, x[:muestras_vis], color="blue", linewidth=1.5)
    axs[i, 0].axhline(0, color="black", linewidth=1.0) # Eje X en y=0
    axs[i, 0].set_title(f"{s['nombre']} - Tiempo", fontsize=10)
    axs[i, 0].set_ylabel("Amplitud [V]")
    axs[i, 0].grid(True)
    
    # Columna 2: Dominio de la Frecuencia (FFT Unilateral)
    # Calculamos la FFT compleja y la normalizamos por N
    X_fft = np.fft.rfft(x) / N
    mag = np.abs(X_fft)
    
    # Multiplicamos por 2 los componentes (excepto continua) para recuperar la amplitud real
    mag[1:-1] = 2 * mag[1:-1]
    
    # Eje de frecuencias en Hz
    freqs = np.fft.rfftfreq(N, d=1/fs)
    
    axs[i, 1].plot(freqs / 1000, mag, color="red", linewidth=1.5)
    axs[i, 1].axhline(0, color="black", linewidth=1.0) # Eje X en y=0
    axs[i, 1].set_title(f"{s['nombre']} - Espectro (FFT)", fontsize=10)
    axs[i, 1].set_ylabel("Magnitud Normalizada")
    axs[i, 1].grid(True)
    axs[i, 1].set_xlim(0, fs / (2 * 1000)) # Acotamos el eje frecuencial hasta Nyquist (10 kHz)

# Configuramos las etiquetas finales en los ejes X de abajo de todo
axs[-1, 0].set_xlabel("Tiempo [ms]")
axs[-1, 1].set_xlabel("Frecuencia [kHz]")

plt.tight_layout()
plt.show()
