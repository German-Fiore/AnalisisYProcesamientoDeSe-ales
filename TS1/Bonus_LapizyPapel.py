import matplotlib.pyplot as plt
import numpy as np


# =====================================================================
# 1. CONVOLUCIÓN LINEAL (Inciso c)
# =====================================================================
# x[n] = u[n+1] - u[n-2] -> {1, 1, 1}
x = np.array([1, 1, 1])

# h[n] = delta[n] - delta[n-4] -> {1, 0, 0, 0, -1}
h = np.array([1, 0, 0, 0, -1])

# Convolución lineal y[n] = x[n] * h[n]
y = np.convolve(x, h)

# =====================================================================
# 2. CONFIGURACIÓN DE LA DFT DE 8 PUNTOS (Desplazada)
# =====================================================================
# Ubicamos la secuencia completa a partir de n=0 y rellenamos con ceros hasta N=8
y_desplazada = np.zeros(8)
y_desplazada[: len(y)] = y

# Cálculo de la FFT de 8 puntos
Y = np.fft.fft(y_desplazada)
k_vals = np.arange(8)


# =====================================================================
# 3. GRAFICACIÓN DE RESULTADOS
# =====================================================================
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# A. Secuencia en el tiempo y[n]
axs[0].stem(k_vals, y_desplazada, basefmt=" ")
axs[0].axhline(0, color="black", linewidth=1.2)
axs[0].set_title("Secuencia y[n] Desplazada a n=0")
axs[0].set_xlabel("Muestra (n)")
axs[0].set_ylabel("Amplitud")
axs[0].set_ylim(-1.5, 1.5)
axs[0].grid(True, linestyle=":", alpha=0.6)

# B. Magnitud de la DFT 
axs[1].stem(k_vals, np.abs(Y), linefmt="C1-", markerfmt="C1o", basefmt=" ")
axs[1].axhline(0, color="black", linewidth=1.2)
axs[1].set_title("Espectro de Magnitud Y[k]")
axs[1].set_xlabel("Frecuencia Discreta k")
axs[1].set_ylabel("Magnitud")
axs[1].set_ylim(0, 5.5)
axs[1].grid(True, linestyle=":", alpha=0.6)


plt.tight_layout()
plt.show()