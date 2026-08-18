# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 19:57:20 2026

@author: German
"""

import numpy as np
import matplotlib.pyplot as plt


# %% Ejercicio 1: Secuencias elementales y ventana de pulso
# =============================================================================
print("--- EJERCICIO 1 ---")

# --- DEFINICIÓN DE VARIABLES Y CÁLCULOS ---

# Índices y valores de x[n]
n_x = np.array([-1, 0, 1, 2])
x = np.array([2, -1, 0, 3])

# Rango para visualizar el escalón
n_u = np.arange(-5, 11)
# u[n] vale 1 si n >= 0, de lo contrario 0
u = np.where(n_u >= 0, 1, 0)

# Definimos un rango que incluya a x[n] para comparar
n_c = np.arange(-2, 5)

# 1. Definimos x[n] en este nuevo rango (rellenando con ceros fuera de sus índices)
x_ext = np.zeros_like(n_c)
indices_x = np.isin(n_c, [-1, 0, 1, 2])
x_ext[indices_x] = [2, -1, 0, 3]

# 2. Definimos el pulso p[n] = u[n] - u[n-3]
# Vale 1 para n=0, 1, 2 y 0 en el resto
p = np.where((n_c >= 0) & (n_c < 3), 1, 0)

# 3. Calculamos el producto g[n] = x[n] * p[n]
g = x_ext * p


# --- VISUALIZACIÓN ---

fig, axs = plt.subplots(5, 1, figsize=(9, 10), sharex=False)

# 1. Punto A: Secuencia x[n] original
axs[0].stem(n_x, x)
axs[0].set_title("a) Secuencia x[n] = 2δ[n+1] - δ[n] + 3δ[n-2]")
axs[0].set_ylabel("Amplitud")

# 2. Punto B: Escalón unitario u[n]
axs[1].stem(n_u, u)
axs[1].set_title("b) Escalón unitario u[n]")
axs[1].axhline(0, color='black', lw=1)
axs[1].set_ylabel("Amplitud")

# 3. Punto C: x[n] en rango extendido
axs[2].stem(n_c, x_ext)
axs[2].set_title("c.1) x[n] original (rango extendido)")

# 4. Punto C: Pulso p[n]
axs[3].stem(n_c, p, linefmt='C1-', markerfmt='C1o')
axs[3].set_title("c.2) Pulso p[n] = u[n] - u[n-3]")

# 5. Punto C: Resultado g[n] = x[n] * p[n]
axs[4].stem(n_c, g, linefmt='C2-', markerfmt='C2o')
axs[4].set_title("c.3) Resultado g[n] = x[n] · p[n]")
axs[4].set_xlabel("n")

# Formato general para todos los paneles
for ax in axs:
    ax.grid(True)

plt.tight_layout()
plt.show()



# %% Ejercicio 2: Manipulación de secuencias (Shift, Flip, Decimation)
# =============================================================================
print("\n--- EJERCICIO 2 ---")

# Definición de la secuencia original x[n] = {1, 3, 2, -1}
n_orig = np.array([0, 1, 2, 3])
x_orig = np.array([1, 3, 2, -1])

# --- Procesamiento de las transformaciones ---

# a) x[n - 3] : Desplazamiento (Shift)
n_a = n_orig + 3
x_a = x_orig

# b) x[-n] : Inversión temporal (Flip)
n_b = -n_orig
x_b = x_orig

# c) x[2n] : Diezmado (Decimation) con M=2
# Conservamos índices pares: 2n = 0 -> n=0; 2n = 2 -> n=1
n_c = np.array([0,1])
x_c = x_orig[::2] # Slicing para tomar cada 2 muestras

# d) x[-n + 2] : Inversión y desplazamiento
# Calculamos n tal que -n + 2 esté en [1-3]
n_d = 2 - n_orig
x_d = x_orig

# --- Visualización ---

fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=False)

# Secuencia Original
axs[0].stem(n_orig, x_orig, linefmt='C0-', markerfmt='C0o', label="Original")
axs[0].set_title("Secuencia Original x[n]")

# a) Desplazamiento
axs[1].stem(n_a, x_a, linefmt='C1-', markerfmt='C1o')
axs[1].set_title("a) Desplazamiento: y[n] = x[n - 3]")

# b) Inversión
axs[2].stem(n_b, x_b, linefmt='C2-', markerfmt='C2o')
axs[2].set_title("b) Inversión: y[n] = x[-n]")

# c) Diezmado
axs[3].stem(n_c, x_c, linefmt='C3-', markerfmt='C3o')
axs[3].set_title("c) Diezmado (M=2): y[n] = x[2n]")

# d) Inversión + Desplazamiento
axs[4].stem(n_d, x_d, linefmt='C4-', markerfmt='C4o')
axs[4].set_title("d) Flip-and-shift: y[n] = x[-n + 2]")

# Formato general
for ax in axs:
    ax.grid(True)
    ax.axhline(0, color='black', lw=1)
    ax.set_ylabel("Amplitud")
    ax.set_xlim(-4, 7) # Ajustamos rango para ver todos los cambios

axs[4].set_xlabel("n (índice)")
plt.tight_layout()
plt.show()



# %% Ejercicio 3: Periodicidad de senoidales discretas
# =============================================================================
print("\n--- EJERCICIO 3 ---")

# Definimos el rango de muestras (0 a 42 como sugiere la guía)
n = np.arange(0, 43)

# Definimos los casos de prueba: (frecuencia angular, título)
casos = [
    (np.pi/6, "a) w0 = pi/6 (Periódica, N=12)"),
    (3.0,     "b) w0 = 3 (No periódica)"),
    (4*np.pi/7, "c) w0 = 4pi/7 (Periódica, N=7)")
]

# Creamos la figura
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

for i, (w0, titulo) in enumerate(casos):
    # Graficamos la señal discreta
    axs[i].stem(n, np.cos(w0 * n), linefmt='C0-', markerfmt='C0o')
    
    # Graficamos la envolvente continua para ver la diferencia
    t_cont = np.linspace(0, 42, 500)
    axs[i].plot(t_cont, np.cos(w0 * t_cont), 'r--', alpha=0.3, label="Envolvente continua")
    
    axs[i].set_title(titulo)
    axs[i].grid(True)
    axs[i].set_ylabel("Amplitud")

axs[-1].set_xlabel("n (muestras)")
plt.tight_layout()
plt.show()



# %% Ejercicio 4: Energía o potencia
# =============================================================================
print("\n--- EJERCICIO 4 ---")

# --- Señal 1: Pulso rectangular ---
x1 = np.ones(10) # 10 muestras de valor 1
E1 = np.sum(x1**2)
P1 = 0 # Por definición para señales de energía finita

print(f"Señal 1 (Pulso): Energía = {E1}, Potencia = {P1} -> Señal de ENERGÍA")

# --- Señal 2: Senoidal infinita (Simulada con un ciclo para potencia) ---
N = 8 # Periodo fundamental
n_ciclo = np.arange(N)
x2_ciclo = np.cos(np.pi/4 * n_ciclo)
P2 = np.sum(x2_ciclo**2) / N

print(f"Señal 2 (Seno Infinito): Potencia Media = {P2} -> Señal de POTENCIA")

# --- Señal 2: Caso ADC (1000 muestras) ---
n_adc = np.arange(1000)
x2_adc = np.cos(np.pi/4 * n_adc)
E2_adc = np.sum(x2_adc**2)

print(f"Señal 2 (1000 muestras): Energía = {E2_adc:.2f}, Potencia = {P1} -> Pasa a ser señal de ENERGÍA")



# %% Ejercicio 5: Clasificación de sistemas (Verificación numérica)
# =============================================================================
print("\n--- EJERCICIO 5 ---")


# Definimos una señal de entrada: Escalón unitario de 20 muestras
n = np.arange(-5, 25)
x = np.where(n >= 0, 1, 0)

# Implementación de los 5 sistemas
def sistema_a(n, x): return x * np.cos(0.5 * np.pi * n)
def sistema_b(n, x): return x**2
def sistema_c(n, x): 
    x_futuro = np.roll(x, -1)
    x_futuro[-1] = 0  # Condición de borde
    return x_futuro + x
def sistema_d(n, x): return n * x
def sistema_e(n, x): return np.cumsum(x) # Acumulador

# Lista para graficar
sistemas = [
    (sistema_a(n, x), "a) y[n] = x[n]·cos(0.5πn)  [Lineal, Var. Tiempo, Causal, Estable]"),
    (sistema_b(n, x), "b) y[n] = x²[n]  [No Lineal, Invariante, Causal, Estable]"),
    (sistema_c(n, x), "c) y[n] = x[n+1] + x[n]  [Lineal, Invariante, NO Causal, Estable]"),
    (sistema_d(n, x), "d) y[n] = n·x[n]  [Lineal, Var. Tiempo, Causal, NO Estable]"),
    (sistema_e(n, x), "e) y[n] = ∑ x[k]  [Lineal, Invariante, Causal, NO Estable]")
]

fig, axs = plt.subplots(5, 1, figsize=(10, 12))

for i, (y, titulo) in enumerate(sistemas):
    axs[i].stem(n, y)
    axs[i].set_title(titulo)
    axs[i].grid(True)
    axs[i].set_ylabel("Amplitud")

axs[-1].set_xlabel("n")
plt.tight_layout()
plt.show()




# %% Ejercicio 6: Respuesta al impulso desde la ecuación de diferencias
# =============================================================================
print("\n--- EJERCICIO 6 ---")

# --- Punto B: Respuesta al impulso h[n] = {2, -1} para n = 0, 1 ---
n_h = np.array([0, 1])
h = np.array([2, -1])

# --- Punto C: Entrada x[n] = {1, 2, 3} para n = 0, 1, 2 ---
x = np.array([1, 2, 3])

# Convolución directa para obtener y[n]
y = np.convolve(x, h)
n_y = np.arange(0, len(y))  # Muestras n = 0, 1, 2, 3


# --- Visualización (2 subgráficos) ---

fig, axs = plt.subplots(2, 1, figsize=(9, 6))

# 1. Respuesta al impulso h[n]
axs[0].stem(n_h, h, linefmt='C0-', markerfmt='C0o')
axs[0].set_title("b) Respuesta al impulso h[n] = 2δ[n] - δ[n-1]")
axs[0].set_ylabel("Amplitud")
axs[0].set_xlim(-1, 4)
axs[0].grid(True)

# 2. Salida y[n]
axs[1].stem(n_y, y, linefmt='C1-', markerfmt='C1o')
axs[1].set_title("c) Salida y[n] para x[n] = {1, 2, 3}")
axs[1].set_ylabel("Amplitud")
axs[1].set_xlabel("n (Muestras)")
axs[1].set_xlim(-1, 4)
axs[1].grid(True)

plt.tight_layout()
plt.show()

# Impresión de resultados
print(f"Salida y[n]: {y}")
print(f"Cantidad de muestras no nulas: {np.count_nonzero(y)}")