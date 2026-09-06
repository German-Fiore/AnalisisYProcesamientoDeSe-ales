import numpy as np
import matplotlib.pyplot as plt

def simular(bits_adc, escala_ruido):
    frecuencia_muestreo = 1000.0  # Frecuencia fs (1000 Hz)
    cant_muestras = 1000          # Muestras N
    v_rango = 2.0                 # Rango V_R (+/- 2.0 V)

    tiempo = np.arange(cant_muestras) / frecuencia_muestreo

    # Paso de cuantización (q = V_R / 2^B)
    paso_cuantizacion = v_rango / (2**bits_adc)

    # GENERACIÓN DE SEÑALES
    frec_fundamental = frecuencia_muestreo / cant_muestras
    senoidal_pura = np.sqrt(2) * np.sin(2 * np.pi * frec_fundamental * tiempo)

    # Potencias teóricas de ruido
    potencia_cuantizacion = (paso_cuantizacion**2) / 12
    potencia_ruido_analogico = escala_ruido * potencia_cuantizacion

    # Ruido analógico n(t)
    ruido_analogico = np.random.normal(0, np.sqrt(potencia_ruido_analogico), cant_muestras)

    # Señal s_R(t) = s(t) + n(t)
    senal_contaminada = senoidal_pura + ruido_analogico

    # CUANTIZACIÓN (ADC) Y ERROR
    senal_cuantizada = paso_cuantizacion * np.round(senal_contaminada / paso_cuantizacion)
    senal_cuantizada = np.clip(senal_cuantizada, -v_rango, v_rango - paso_cuantizacion)
    error_cuantizacion = senal_cuantizada - senal_contaminada

    # ANÁLISIS ESPECTRAL (PSD) Y PISOS DE RUIDO
    fft_pura = np.fft.rfft(senoidal_pura) / cant_muestras
    fft_contaminada = np.fft.rfft(senal_contaminada) / cant_muestras
    fft_cuantizada = np.fft.rfft(senal_cuantizada) / cant_muestras

    psd_pura = np.abs(fft_pura)**2
    psd_contaminada = np.abs(fft_contaminada)**2
    psd_cuantizada = np.abs(fft_cuantizada)**2

    psd_pura[1:-1] *= 2
    psd_contaminada[1:-1] *= 2
    psd_cuantizada[1:-1] *= 2

    psd_pura_db = 10 * np.log10(psd_pura + 1e-12)
    psd_contaminada_db = 10 * np.log10(psd_contaminada + 1e-12)
    psd_cuantizada_db = 10 * np.log10(psd_cuantizada + 1e-12)

    frecuencias = np.fft.rfftfreq(cant_muestras, d=1/frecuencia_muestreo)

    piso_analogico_db = 10 * np.log10(potencia_ruido_analogico / (cant_muestras / 2))
    piso_digital_db = 10 * np.log10(potencia_cuantizacion / (cant_muestras / 2))

    # VISUALIZACIÓN
    # --- Gráfico 1: Dominio del Tiempo ---
    plt.figure(figsize=(10, 4))
    plt.plot(tiempo, senal_cuantizada, label=r"$s_Q = Q_{B, V_R}\{s_R\}$ (ADC out)", color="tab:blue", lw=1.5)
    plt.plot(tiempo, senal_contaminada, color="green", linestyle=":", marker="o", markersize=2, label=r"$s_R = s + n$ (ADC in)", alpha=0.7)
    plt.plot(tiempo, senoidal_pura, color="orange", linestyle="--", label=r"$s$ (analog)", lw=1.2)
    plt.title(rf"Señal muestreada por un ADC de {bits_adc} bits - $\pm V_R = {v_rango}\text{{ V}} - q = {paso_cuantizacion}\text{{ V}}$")
    plt.xlabel("tiempo [segundos]")
    plt.ylabel("Amplitud [V]")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    # --- Gráfico 2: Densidad Espectral de Potencia (PSD) ---
    plt.figure(figsize=(10, 4))
    plt.plot(frecuencias, psd_cuantizada_db, color="tab:blue", lw=1.0, label=r"$s_Q = Q_{B, V_R}\{s_R\}$ (ADC out)")
    plt.plot(frecuencias, psd_pura_db, color="orange", linestyle="--", lw=1.0, label=r"$s$ (analog)")
    plt.plot(frecuencias, psd_contaminada_db, color="green", linestyle=":", alpha=0.6, label=r"$s_R = s + n$ (ADC in)")
    plt.axhline(piso_analogico_db, color="red", linestyle="--", label=rf"$\overline{{n}} = {piso_analogico_db:.1f}\text{{ dB}}$ (piso analog.)")
    plt.axhline(piso_digital_db, color="cyan", linestyle="--", label=rf"$\overline{{n}}_Q = {piso_digital_db:.1f}\text{{ dB}}$ (piso digital)")
    plt.title(rf"Señal muestreada por un ADC de {bits_adc} bits - $\pm V_R = {v_rango}\text{{ V}} - q = {paso_cuantizacion}\text{{ V}}$")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Densidad de Potencia [dB]")
    plt.ylim(-160 if bits_adc == 16 else -85, 10)  # Ajuste automático de escala si usás 16 bits
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    # --- Gráfico 3: Histograma del Error ---
    plt.figure(figsize=(10, 4))
    plt.hist(error_cuantizacion, bins=10, color="tab:blue", edgecolor="white")
    plt.plot([-paso_cuantizacion/2, -paso_cuantizacion/2, paso_cuantizacion/2, paso_cuantizacion/2], [0, 100, 100, 0], 'r--', lw=1.5)
    plt.xlabel("Error de cuantización $e_q$ [V]") 
    plt.ylabel("Cantidad de muestras")
    plt.title(rf"Ruido de cuantización para {bits_adc} bits - $\pm V_R = {v_rango}\text{{ V}} - q = {paso_cuantizacion}\text{{ V}}$")
    plt.xlim(-paso_cuantizacion, paso_cuantizacion)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()
    
simular(bits_adc=4, escala_ruido=1.0)