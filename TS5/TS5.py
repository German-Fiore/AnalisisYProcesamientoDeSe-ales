import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal as sig

# Configuración de gráficos
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 9
plt.rcParams['axes.grid'] = True

resultados_tabla = []


def estimar_ancho_de_banda(f, Pxx, porcentaje=0.99):
    """Calcula el ancho de banda acumulado al porcentaje especificado."""
    pot_acum = np.cumsum(Pxx)
    pot_acum_norm = pot_acum / pot_acum[-1]
    idx_corte = np.where(pot_acum_norm >= porcentaje)[0][0]
    f_corte = f[idx_corte]
    bw = f_corte - f[0]
    return bw, f[0], f_corte


def procesar_senal(nombre, x, fs, nperseg, color='red', xlim_temp=None):
    """Procesa una señal por Welch, remueve media (DC), normaliza y calcula métricas."""
    x = np.squeeze(x).astype(np.float64)

    # REMOVER COMPONENTE CONTINUA (DC / valor medio)
    x = x - np.mean(x)

    # NORMALIZACIÓN DE AMPLITUD (Escalado entre -1 y 1)
    max_val = np.max(np.abs(x))
    if max_val > 0:
        x = x / max_val

    t = np.arange(len(x)) / fs

    # Estimación de PSD mediante Welch (Ventana Hann)
    f, Pxx = sig.welch(
        x,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=nperseg // 2,
        scaling='density',
    )

    # Ancho de Banda al 99%
    bw, fmin, fmax = estimar_ancho_de_banda(f, Pxx, porcentaje=0.99)
    df = f[1] - f[0]

    # Guardar resultados
    resultados_tabla.append({
        'Señal': nombre,
        'fs [Hz]': fs,
        'Método': 'Welch (Hann)',
        'N (win)': nperseg,
        'Δf [Hz]': round(df, 3),
        'BW 99% [Hz]': round(bw, 2),
        'f_corte [Hz]': round(fmax, 2),
    })

    print(
        f'[{nombre}] | BW 99%: {bw:.2f} Hz (desde {fmin:.2f} Hz hasta {fmax:.2f} Hz)'
    )

    # Gráficos
    plt.figure(figsize=(10, 6))

    # Dominio Temporal
    plt.subplot(2, 1, 1)
    plt.plot(t, x, color=color)
    plt.title(f'Señal {nombre} - Dominio Temporal')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud Normalizada')
    if xlim_temp:
        plt.xlim(xlim_temp)
    else:
        plt.xlim(0, t[-1])
    plt.grid(True)

    # Dominio Frecuencial (PSD en semilogy)
    plt.subplot(2, 1, 2)
    plt.semilogy(f, Pxx, color=color)
    plt.title('Densidad Espectral de Potencia (Método de Welch)')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('PSD [V²/Hz]')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# =============================================================================
# 1. ELECTROCARDIOGRAMA (ECG)
# =============================================================================
fs_ecg = 1000
nperseg_ecg = 2048

if os.path.exists('ECG_TP4.mat'):
    mat_struct = sio.loadmat('ECG_TP4.mat')
    ecg_ruido = mat_struct['ecg_lead']
    procesar_senal('ECG (Con Ruido)', ecg_ruido, fs_ecg, nperseg_ecg, color='red')

if os.path.exists('ecg_sin_ruido.npy'):
    ecg_sin_ruido = np.load('ecg_sin_ruido.npy')
    procesar_senal(
        'ECG (Sin Ruido)', ecg_sin_ruido, fs_ecg, nperseg_ecg, color='darkred'
    )

# =============================================================================
# 2. PLETISMOGRAFÍA (PPG)
# =============================================================================
fs_ppg = 400
nperseg_ppg = 1024

if os.path.exists('PPG.csv'):
    ppg_ruido = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)
    procesar_senal(
        'PPG (Con Ruido)',
        ppg_ruido,
        fs_ppg,
        nperseg_ppg,
        color='hotpink',
        xlim_temp=(0, 70),
    )

if os.path.exists('ppg_sin_ruido.npy'):
    ppg_sin_ruido = np.load('ppg_sin_ruido.npy')
    procesar_senal(
        'PPG (Sin Ruido)',
        ppg_sin_ruido,
        fs_ppg,
        nperseg_ppg,
        color='deeppink',
        xlim_temp=(0, 70),
    )

# =============================================================================
# 3. AUDIOS (.WAV)
# =============================================================================
nperseg_audio = 4096
archivos_audio = [
    ('La Cucaracha', 'la cucaracha.wav', 'darkorange'),
    ('Prueba PSD', 'prueba psd.wav', 'limegreen'),
    ('Silbido', 'silbido.wav', 'royalblue'),
]

for tag, fname, col in archivos_audio:
    if os.path.exists(fname):
        fs_aud, wav_data = sio.wavfile.read(fname)
        if wav_data.ndim > 1:
            wav_data = wav_data[:, 0]
        procesar_senal(
            f'Audio - {tag}', wav_data, fs_aud, nperseg_audio, color=col
        )

# =============================================================================
# TABLA COMPARATIVA DE RESULTADOS
# =============================================================================
print('\n=====================================================================')
print(' TABLA COMPARATIVA DE ANCHO DE BANDA Y ESTIMACIÓN DE PSD')
print('=====================================================================')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df_resultados = pd.DataFrame(resultados_tabla)
print(df_resultados)