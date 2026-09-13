import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. PARÁMETROS DEL EXPERIMENTO
# =============================================================================
cant_muestras = 1000                                 # N = 1000
frecuencia_muestreo = 1000.0                         # fs = 1000 Hz
delta_f = frecuencia_muestreo / cant_muestras        # df = 1 Hz

# Casos de k0 requeridos (desintonía respecto al bin de referencia 250)
casos_k0 = [cant_muestras/4, cant_muestras/4 + 0.25, cant_muestras/4 + 0.5]
colores = ['#1f77b4', '#2ca02c', '#ff7f0e']
etiquetas = ['k0 = N/4 (250.0 Hz)', 'k0 = N/4 + 0.25 (250.25 Hz)', 'k0 = N/4 + 0.5 (250.5 Hz)']

tiempo = np.arange(cant_muestras) / frecuencia_muestreo

# Generación de las tres senoidales con potencia unitaria (Amplitud = sqrt(2))
senales = []
for k0 in casos_k0:
    frecuencia_0 = k0 * delta_f
    # s(t) = sqrt(2) * sin(2 * pi * f0 * t) -> Potencia = A^2 / 2 = 2 / 2 = 1 W
    senoidal = np.sqrt(2) * np.sin(2 * np.pi * frecuencia_0 * tiempo)
    senales.append(senoidal)

# =============================================================================
# INCISO B: VERIFICACIÓN DE POTENCIA MEDIANTE PARSEVAL
# =============================================================================
print("=" * 65)
print(" VERIFICACIÓN DE PARSEVAL (INCISO B)")
print("=" * 65)

for i, senoidal in enumerate(senales):
    # Potencia en el dominio del tiempo
    potencia_tiempo = np.mean(senoidal**2)
    
    # FFT unilateral normalizada
    fft_unilateral = np.fft.rfft(senoidal) / cant_muestras
    psd_unilateral = np.abs(fft_unilateral)**2
    psd_unilateral[1:-1] *= 2  # Factor 2 para componentes de CA
    
    # Potencia total en el dominio frecuencial (Suma de la PSD)
    potencia_frecuencia = np.sum(psd_unilateral)
    
    print(f"Caso {etiquetas[i]}:")
    print(f"  - Potencia en tiempo : {potencia_tiempo:.6f} W")
    print(f"  - Potencia en freq.  : {potencia_frecuencia:.6f} W")
    print("-" * 65)

# =============================================================================
# GRAFICACIÓN (INCISOS A y C)
# =============================================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 9))

# --- Inciso A: Sin Zero-Padding ---
for i, senoidal in enumerate(senales):
    fft_unilateral = np.fft.rfft(senoidal) / cant_muestras
    psd_unilateral = np.abs(fft_unilateral)**2
    psd_unilateral[1:-1] *= 2
    
    frecuencias = np.fft.rfftfreq(cant_muestras, d=1/frecuencia_muestreo)
    axes[0].plot(frecuencias, 10 * np.log10(psd_unilateral + 1e-12), 
                 label=etiquetas[i], color=colores[i], marker='o', ms=4, alpha=0.8)

axes[0].set_title("a) PSD Sin Zero-Padding (N = 1000) - Muestreo Espectral Grueso", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Frecuencia [Hz]")
axes[0].set_ylabel("Potencia [dB]")
#axes[0].set_xlim(240, 260)
axes[0].set_ylim(-130, 5)
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend()

# --- Inciso C: Con Zero-Padding (10*N) ---
muestras_zp = 10 * cant_muestras  # N + 9*N ceros = 10000 muestras

for i, senoidal in enumerate(senales):
    # Se evalúa la FFT rellenando con ceros pero normalizando por N (las muestras reales)
    fft_zp = np.fft.rfft(senoidal, n=muestras_zp) / cant_muestras
    psd_zp = np.abs(fft_zp)**2
    psd_zp[1:-1] *= 2
    
    frecuencias_zp = np.fft.rfftfreq(muestras_zp, d=1/frecuencia_muestreo)
    axes[1].plot(frecuencias_zp, 10 * np.log10(psd_zp + 1e-12), 
                 label=etiquetas[i], color=colores[i], alpha=0.8)

axes[1].set_title("c) PSD Con Zero-Padding (N_total = 10000) - Interpolación Espectral", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Frecuencia [Hz]")
axes[1].set_ylabel("Potencia [dB]")
#axes[1].set_xlim(240, 260)
axes[1].set_ylim(-130, 5)
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].legend()

plt.tight_layout()
plt.show()