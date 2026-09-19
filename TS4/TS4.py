import matplotlib.pyplot as plt
import numpy as np
import scipy.signal.windows as win

np.random.seed(42)

# Parámetros generales
N = 1000  # Muestras por realización
M = 200  # Realizaciones
a0 = np.sqrt(2)  # Amplitud para P_s = 1 W
Omega0 = np.pi / 2  # Omega0 = pi/2 -> bin k0 = 250

# SNRs 
SNRs_dB = [3, 10]

windows = {
    'Rectangular': np.ones(N),
    'Flat-top': win.flattop(N, sym=False),
    'Blackman-Harris': win.blackmanharris(N, sym=False),
    'Hann': win.hann(N, sym=False),
}

results = {}

for snr_db in SNRs_dB:
    sigma = 10 ** (-snr_db / 20)
    fr = np.random.uniform(-2, 2, M)
    Omega1 = Omega0 + fr * (2 * np.pi / N)

    results[snr_db] = {}

    for w_name, w in windows.items():
        w_sum = np.sum(w)  # Suma de la ventana para normalización exacta
        n_pad = 16 * N

        a_estimates = []
        Omega_estimates = []

        for j in range(M):
            n = np.arange(N)
            noise = np.random.normal(0, sigma, N)
            x = a0 * np.sin(Omega1[j] * n) + noise

            #  Estimador de Amplitud evaluado en Omega0 = pi/2
            X_w_Omega0 = np.sum(x * w * np.exp(-1j * Omega0 * n))
            #  Normalización directa por la suma de la ventana
            a_hat = (2 * np.abs(X_w_Omega0)) / w_sum
            a_estimates.append(a_hat)

            #  Estimador de Frecuencia con interpolación parabólica (Log-FFT)
            X_fft = np.abs(np.fft.rfft(x * w, n=n_pad))
            idx_max = np.argmax(X_fft)

            if 0 < idx_max < len(X_fft) - 1:
                alpha_val = np.log(X_fft[idx_max - 1] + 1e-12)
                beta_val = np.log(X_fft[idx_max] + 1e-12)
                gamma_val = np.log(X_fft[idx_max + 1] + 1e-12)
                delta_val = (
                    0.5
                    * (alpha_val - gamma_val)
                    / (alpha_val - 2 * beta_val + gamma_val)
                )
                idx_fine = idx_max + delta_val
            else:
                idx_fine = idx_max

            Omega_hat = idx_fine * (2 * np.pi / n_pad)
            Omega_estimates.append(Omega_hat)

        a_arr = np.array(a_estimates)
        Omega_arr = np.array(Omega_estimates)

        # Cálculo de Sesgo y Varianza
        err_a = a_arr - a0
        err_w = Omega_arr - Omega1

        results[snr_db][w_name] = {
            'a_hat': a_arr,
            'sa_a': np.mean(err_a),
            'va_a': np.var(a_arr),
            'Omega_hat': Omega_arr,
            'sa_w': np.mean(err_w),
            'va_w': np.var(Omega_arr),
        }

# --- IMPRESIÓN DE TABLAS Y GRÁFICOS ---

for snr_db in SNRs_dB:
    print("\n" + "=" * 60)
    print(f" RESULTADOS PARA SNR = {snr_db} dB")
    print("=" * 60)

    #  Tabla de Estimación de Amplitud
    print("\nEstimación de Amplitud (a_hat):")
    print("-" * 65)
    print(
        f"{'Métrica':<10} | {'Rectangular':<12} | {'Flat-top':<12} | {'Blackman-H.':<12} | {'Hann':<12}"
    )
    print("-" * 65)

    sa_a_line = f"{'sa':<10} | " + " | ".join(
        [f"{results[snr_db][w]['sa_a']:12.6f}" for w in windows]
    )
    va_a_line = f"{'va':<10} | " + " | ".join(
        [f"{results[snr_db][w]['va_a']:12.6e}" for w in windows]
    )

    print(sa_a_line)
    print(va_a_line)

    #  Tabla de Estimación de Frecuencia
    print("\nEstimación de Frecuencia (Omega_hat):")
    print("-" * 65)
    print(
        f"{'Métrica':<10} | {'Rectangular':<12} | {'Flat-top':<12} | {'Blackman-H.':<12} | {'Hann':<12}"
    )
    print("-" * 65)

    sa_w_line = f"{'sa':<10} | " + " | ".join(
        [f"{results[snr_db][w]['sa_w']:12.6f}" for w in windows]
    )
    va_w_line = f"{'va':<10} | " + " | ".join(
        [f"{results[snr_db][w]['va_w']:12.6e}" for w in windows]
    )

    print(sa_w_line)
    print(va_w_line)

    # Histogramas Comparativos (4 Ventanas Juntas)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for w_name in windows:
        # Histograma Amplitud
        axes[0].hist(
            results[snr_db][w_name]["a_hat"],
            bins=20,
            alpha=0.5,
            label=w_name,
            density=True,
        )
        # Histograma Frecuencia
        axes[1].hist(
            results[snr_db][w_name]["Omega_hat"],
            bins=20,
            alpha=0.5,
            label=w_name,
            density=True,
        )

    axes[0].axvline(
        a0,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Valor Real a0 ({a0:.3f})",
    )
    axes[0].set_title(
        f"Histograma de Amplitud (SNR = {snr_db} dB)", fontweight="bold"
    )
    axes[0].set_xlabel(r"$\hat{a}_1$")
    axes[0].set_ylabel("Densidad")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    axes[1].axvline(
        Omega0,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=r"$\Omega_0$ Central",
    )
    axes[1].set_title(
        f"Histograma de Frecuencia (SNR = {snr_db} dB)", fontweight="bold"
    )
    axes[1].set_xlabel(r"$\hat{\Omega}_1$ [rad/muestra]")
    axes[1].set_ylabel("Densidad")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.show()