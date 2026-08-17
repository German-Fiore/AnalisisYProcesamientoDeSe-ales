
# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import numpy as np
import matplotlib.pyplot as plt

# Parámetros globales
fs = 1000  # Frecuencia de muestreo (Hz)
N = 1000   # Cantidad de muestras

# Lista de frecuencias para el experimento (Bonus)
frecuencias = [3, 500, 999, 1001, 2001]

def mi_funcion_sen(vmax=1, dc=0, ff=1, ph=0, nn=N, fs=fs):
    ts = 1 / fs
    # Tiempo vector Nx1
    tt = np.arange(nn).reshape(-1, 1) * ts
    # Señal vectorial Nx1
    xx = vmax * np.sin(2 * np.pi * ff * tt + ph) + dc
    
    return tt, xx

# Visualización de los experimentos
plt.figure(figsize=(12, 10))

for i, ff in enumerate(frecuencias):
    tt, xx = mi_funcion_sen(vmax=1.5, dc=0, ff=ff, ph=0, nn=N, fs=fs)
    
    plt.subplot(len(frecuencias), 1, i + 1)
    plt.plot(tt, xx)
    plt.title(f"Experimento: ff = {ff} Hz (fs = {fs} Hz)")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [V]")
    plt.grid(True)

plt.tight_layout()
plt.show()





