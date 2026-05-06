import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================
file_h5 = r"C:/Users/lenovo/Documents/Multi Channel DataManager/datos0010.h5"

# Canal del fotodiodo en Stream_1 (A4)
PHOTODIODE_CHANNEL = 3

# Downsample para detectar ciclos/flash
DOWNSAMPLE_FACTOR = 200

# Parámetros detección de ciclos
CYCLE_SMOOTH_S = 2.0
MIN_CYCLE_INTERVAL_S = 40.0

# Ventana donde buscar flash ON/OFF tras inicio de ciclo
SEARCH_WINDOW_START_S = 3.0
SEARCH_WINDOW_END_S = 7.0

# Parámetros PSD
#PSD_NPERSEG = 2048

PSD_NPERSEG = 8192
PSD_NFFT_FACTOR = 4
CANAL_PSD = 0
FREQ_XLIM_LOW = 200


# ============================================================
# 2. FUNCIONES UTILITARIAS
# ============================================================
def load_photodiode_and_shapes(file_h5, photodiode_channel):
    """Carga fotodiodo (ya convertido), tiempo y shape de electrodos."""
    with h5py.File(file_h5, "r") as f:
        elec_data = f["Data/Recording_0/AnalogStream/Stream_0/ChannelData"]

        anlg_data = f["Data/Recording_0/AnalogStream/Stream_1/ChannelData"]
        anlg_info = f["Data/Recording_0/AnalogStream/Stream_1/InfoChannel"]

        raw_pd = anlg_data[photodiode_channel, :].astype(float)

        adzero_pd = float(anlg_info[photodiode_channel]["ADZero"])
        conv_pd = float(anlg_info[photodiode_channel]["ConversionFactor"])
        exp_pd = float(anlg_info[photodiode_channel]["Exponent"])
        tick_us_pd = float(anlg_info[photodiode_channel]["Tick"])
        label_pd = anlg_info[photodiode_channel]["Label"].decode()

        fs = 1e6 / tick_us_pd
        t = np.arange(len(raw_pd)) / fs
        photodiode = (raw_pd - adzero_pd) * conv_pd * (10.0 ** exp_pd)

        n_channels, n_samples = elec_data.shape

    print("Fotodiodo:", label_pd)
    print("Sampling rate:", fs)
    print("Duración (s):", t[-1])
    print("Electrodos:", n_channels, "canales")
    print("Muestras:", n_samples)

    return photodiode, t, fs, n_channels, n_samples


def detect_cycle_onsets(photodiode_ds, fs_ds, cycle_smooth_s, min_cycle_interval_s):
    """Detecta inicios de ciclo con envolvente RMS."""
    signal_centered = photodiode_ds - np.median(photodiode_ds)

    smooth_samples = max(int(fs_ds * cycle_smooth_s), 1)
    kernel = np.ones(smooth_samples) / smooth_samples
    mean_power = np.convolve(signal_centered ** 2, kernel, mode="same")
    envelope_rms = np.sqrt(mean_power)

    env_low = np.percentile(envelope_rms, 10)
    env_high = np.percentile(envelope_rms, 90)
    envelope_threshold = env_low + 0.5 * (env_high - env_low)

    envelope_state = envelope_rms > envelope_threshold
    candidates = np.where((~envelope_state[:-1]) & (envelope_state[1:]))[0] + 1

    min_interval_samples = int(min_cycle_interval_s * fs_ds)

    cycle_onsets = []
    if len(candidates) > 0:
        cycle_onsets.append(candidates[0])
        for idx in candidates[1:]:
            if idx - cycle_onsets[-1] > min_interval_samples:
                cycle_onsets.append(idx)

    return np.array(cycle_onsets, dtype=int), envelope_rms, envelope_threshold


def detect_flashes_in_cycles(
    t_ds,
    photodiode_ds,
    cycle_onsets,
    search_window_start_s,
    search_window_end_s,
    drop_first_cycle=True
):
    """Detecta ON y OFF del flash en cada ciclo, dentro de una ventana temporal."""
    pd_low = np.percentile(photodiode_ds, 10)
    pd_high = np.percentile(photodiode_ds, 90)
    pd_threshold = (pd_low + pd_high) / 2

    flash_on_times = []
    flash_off_times = []

    for cycle_idx in cycle_onsets:
        cycle_start_time = t_ds[cycle_idx]
        w_start = cycle_start_time + search_window_start_s
        w_end = cycle_start_time + search_window_end_s

        mask = (t_ds >= w_start) & (t_ds < w_end)
        window_time = t_ds[mask]
        window_signal = photodiode_ds[mask]

        if len(window_signal) < 2:
            continue

        window_state = window_signal > pd_threshold

        # OFF -> ON
        on_candidates = np.where((~window_state[:-1]) & (window_state[1:]))[0] + 1
        if len(on_candidates) == 0:
            continue

        local_on_idx = on_candidates[0]
        flash_on_time = window_time[local_on_idx]

        # ON -> OFF (después de ON)
        post_on_state = window_state[local_on_idx:]
        off_candidates = np.where((post_on_state[:-1]) & (~post_on_state[1:]))[0] + 1
        if len(off_candidates) == 0:
            continue

        local_off_idx = local_on_idx + off_candidates[0]
        flash_off_time = window_time[local_off_idx]

        flash_on_times.append(flash_on_time)
        flash_off_times.append(flash_off_time)

    flash_on_times = np.array(flash_on_times)
    flash_off_times = np.array(flash_off_times)

    if drop_first_cycle and len(flash_on_times) > 1 and len(flash_off_times) > 1:
        flash_on_times = flash_on_times[1:]
        flash_off_times = flash_off_times[1:]

    return flash_on_times, flash_off_times, pd_threshold


def load_electrode_conversion(file_h5):
    """Carga factores para convertir Stream_0 a µV."""
    with h5py.File(file_h5, "r") as f:
        elec_data = f["Data/Recording_0/AnalogStream/Stream_0/ChannelData"]
        elec_info = f["Data/Recording_0/AnalogStream/Stream_0/InfoChannel"]

        adzero_all = np.array([float(elec_info[ch]["ADZero"]) for ch in range(elec_data.shape[0])])
        conv_all = np.array([float(elec_info[ch]["ConversionFactor"]) for ch in range(elec_data.shape[0])])
        exp_all = np.array([float(elec_info[ch]["Exponent"]) for ch in range(elec_data.shape[0])])

        scale_all = conv_all * (10.0 ** exp_all) * 1e6
        n_total_samples = elec_data.shape[1]

    return adzero_all, scale_all, n_total_samples


def extract_segments_indices(file_h5, indices_ini, indices_fin, adzero_all, scale_all):
    """Extrae segmentos [ini:fin] para todos los canales, en µV."""
    segmentos = []

    with h5py.File(file_h5, "r") as f:
        elec_data = f["Data/Recording_0/AnalogStream/Stream_0/ChannelData"]

        for i, (ini, fin) in enumerate(zip(indices_ini, indices_fin)):
            if ini < 0 or fin <= ini or fin > elec_data.shape[1]:
                continue

            raw_block = elec_data[:, ini:fin].astype(float)
            signal_uV = (raw_block - adzero_all[:, None]) * scale_all[:, None]
            segmentos.append(signal_uV)

            print(f"Segmento {i + 1}/{len(indices_ini)}: {ini} -> {fin}", flush=True)

    if len(segmentos) == 0:
        return None

    min_len = min(seg.shape[1] for seg in segmentos)
    segmentos_recortados = [seg[:, :min_len] for seg in segmentos]
    return np.stack(segmentos_recortados, axis=0)


##def psd_promedio(segmentos, canal, fs, nperseg=2048):
##    """Promedia PSD sobre repeticiones para un canal."""
##    psd_list = []
##    freqs = None
##
##    for i in range(segmentos.shape[0]):
##        signal = segmentos[i, canal, :]
##        f, pxx = welch(signal, fs=fs, nperseg=nperseg)
##        freqs = f
##        psd_list.append(pxx)
##
##    psd_array = np.array(psd_list)
##    psd_mean = np.mean(psd_array, axis=0)
##    psd_std = np.std(psd_array, axis=0)
##
##    return freqs, psd_mean, psd_std


def psd_promedio(segmentos, canal, fs, nperseg=8192):
    """Promedia PSD sobre repeticiones para un canal."""
    psd_list = []
    freqs = None

    for i in range(segmentos.shape[0]):
        signal = segmentos[i, canal, :]
        nperseg_eff = min(nperseg, signal.shape[0])

        f, pxx = welch(
            signal,
            fs=fs,
            nperseg=nperseg_eff,
            noverlap=nperseg_eff // 2,
            nfft=PSD_NFFT_FACTOR * nperseg_eff,
            detrend="constant",
            scaling="density"
        )

        freqs = f
        psd_list.append(pxx)

    psd_array = np.array(psd_list)
    psd_mean = np.mean(psd_array, axis=0)
    psd_std = np.std(psd_array, axis=0)

    return freqs, psd_mean, psd_std


# ============================================================
# 3. FLUJO PRINCIPAL
# ============================================================
photodiode, t, fs, n_channels, n_samples = load_photodiode_and_shapes(
    file_h5=file_h5,
    photodiode_channel=PHOTODIODE_CHANNEL
)

# Downsample fotodiodo
t_ds = t[::DOWNSAMPLE_FACTOR]
photodiode_ds = photodiode[::DOWNSAMPLE_FACTOR]
fs_ds = fs / DOWNSAMPLE_FACTOR

# Detectar ciclos
cycle_onsets, envelope_rms, envelope_threshold = detect_cycle_onsets(
    photodiode_ds=photodiode_ds,
    fs_ds=fs_ds,
    cycle_smooth_s=CYCLE_SMOOTH_S,
    min_cycle_interval_s=MIN_CYCLE_INTERVAL_S
)

print("Número de ciclos detectados:", len(cycle_onsets))
if len(cycle_onsets) > 0:
    print("Tiempos de inicio de ciclo (s):", t_ds[cycle_onsets])

# Detectar flashes ON/OFF
flash_on_times, flash_off_times, pd_threshold = detect_flashes_in_cycles(
    t_ds=t_ds,
    photodiode_ds=photodiode_ds,
    cycle_onsets=cycle_onsets,
    search_window_start_s=SEARCH_WINDOW_START_S,
    search_window_end_s=SEARCH_WINDOW_END_S,
    drop_first_cycle=True
)

print("Número de flashes ON:", len(flash_on_times))
print("Número de flashes OFF:", len(flash_off_times))
print("Tiempos ON (s):", flash_on_times)
print("Tiempos OFF (s):", flash_off_times)

if len(flash_on_times) == 0 or len(flash_off_times) == 0:
    raise RuntimeError("No se detectaron suficientes flashes ON/OFF para continuar.")

# Pasar a índices originales
flash_on_indices = (flash_on_times * fs).astype(int)
flash_off_indices = (flash_off_times * fs).astype(int)
print("Índices ON:", flash_on_indices)
print("Índices OFF:", flash_off_indices)

# ============================================================
# 4. GRÁFICA DETECCIÓN FOTODIODO
# ============================================================
plt.figure(figsize=(12, 5))
plt.plot(t_ds, photodiode_ds, label="Fotodiodo downsampled")
plt.axhline(pd_threshold, color="red", linestyle="--", label="Threshold ON/OFF")

plt.plot(
    flash_on_times,
    np.interp(flash_on_times, t_ds, photodiode_ds),
    "go",
    markersize=8,
    label="Flash ON"
)
plt.plot(
    flash_off_times,
    np.interp(flash_off_times, t_ds, photodiode_ds),
    "ro",
    markersize=8,
    label="Flash OFF"
)

plt.xlabel("Tiempo (s)")
plt.ylabel("Señal")
plt.title("Detección de flash ON/OFF en A4")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 5. EXTRAER SEGMENTOS ON / BASAL / OFF
# ============================================================
adzero_all, scale_all, n_total_samples = load_electrode_conversion(file_h5)

print("Extrayendo segmentos ON...", flush=True)
segmentos_on = extract_segments_indices(
    file_h5=file_h5,
    indices_ini=flash_on_indices,
    indices_fin=flash_off_indices,
    adzero_all=adzero_all,
    scale_all=scale_all
)

if segmentos_on is None:
    raise RuntimeError("No se pudieron extraer segmentos ON.")

print("Shape segmentos ON:", segmentos_on.shape)

# Basal: justo antes de ON, misma duración que ON
dur = segmentos_on.shape[2]
indices_ini_basal = flash_on_indices - dur
indices_fin_basal = flash_on_indices

print("Extrayendo segmentos basales...", flush=True)
segmentos_basales = extract_segments_indices(
    file_h5=file_h5,
    indices_ini=indices_ini_basal,
    indices_fin=indices_fin_basal,
    adzero_all=adzero_all,
    scale_all=scale_all
)

if segmentos_basales is None:
    raise RuntimeError("No se pudieron extraer segmentos basales.")
print("Shape segmentos basales:", segmentos_basales.shape)

# OFF: inmediatamente después de OFF detectado, misma duración
indices_ini_off = flash_off_indices
indices_fin_off = flash_off_indices + dur

print("Extrayendo segmentos OFF...", flush=True)
segmentos_off = extract_segments_indices(
    file_h5=file_h5,
    indices_ini=indices_ini_off,
    indices_fin=indices_fin_off,
    adzero_all=adzero_all,
    scale_all=scale_all
)

if segmentos_off is None:
    raise RuntimeError("No se pudieron extraer segmentos OFF.")
print("Shape segmentos OFF:", segmentos_off.shape)

# Igualar cantidad de repeticiones entre ON/Basal/OFF (por seguridad)
n_rep = min(segmentos_on.shape[0], segmentos_basales.shape[0], segmentos_off.shape[0])
segmentos_on = segmentos_on[:n_rep]
segmentos_basales = segmentos_basales[:n_rep]
segmentos_off = segmentos_off[:n_rep]

print("Repeticiones usadas (ON/Basal/OFF):", n_rep)

# ============================================================
# 6. ENERGÍA POR CANAL
# ============================================================
energia_on = np.mean(segmentos_on ** 2, axis=(0, 2))
energia_basal = np.mean(segmentos_basales ** 2, axis=(0, 2))
energia_off = np.mean(segmentos_off ** 2, axis=(0, 2))

##plt.figure(figsize=(8, 4))
##plt.plot(energia_on, label="ON")
##plt.plot(energia_off, label="OFF")
##plt.plot(energia_basal, label="Basal")
##plt.legend()
##plt.title("Energía por canal: ON vs OFF vs Basal")
##plt.xlabel("Canal")
##plt.ylabel("Energía")
##plt.tight_layout()
##plt.show()

# ============================================================
# 7. EJEMPLO DE TRAZAS EN UN CANAL
# ============================================================
canal = CANAL_PSD

plt.figure(figsize=(10, 4))
for i in range(segmentos_on.shape[0]):
    plt.plot(segmentos_on[i, canal, :], alpha=0.5)
plt.xlabel("Tiempo (muestras)")
plt.ylabel("Señal (µV)")
plt.title(f"Canal {canal} durante ON")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
for i in range(segmentos_off.shape[0]):
    plt.plot(segmentos_off[i, canal, :], alpha=0.5)
plt.xlabel("Tiempo (muestras)")
plt.ylabel("Señal (µV)")
plt.title(f"Canal {canal} durante OFF")
plt.tight_layout()
plt.show()

# ============================================================
# 8. PSD PROMEDIO ± STD
# ============================================================
f_on, psd_on_mean, psd_on_std = psd_promedio(segmentos_on, canal, fs, PSD_NPERSEG)
f_off, psd_off_mean, psd_off_std = psd_promedio(segmentos_off, canal, fs, PSD_NPERSEG)
f_basal, psd_basal_mean, psd_basal_std = psd_promedio(segmentos_basales, canal, fs, PSD_NPERSEG)

plt.figure(figsize=(10, 5))

plt.semilogy(f_on, psd_on_mean, label="ON")
plt.fill_between(f_on, psd_on_mean - psd_on_std, psd_on_mean + psd_on_std, alpha=0.2)

plt.semilogy(f_off, psd_off_mean, label="OFF")
plt.fill_between(f_off, psd_off_mean - psd_off_std, psd_off_mean + psd_off_std, alpha=0.2)

plt.semilogy(f_basal, psd_basal_mean, label="Basal")
plt.fill_between(f_basal, psd_basal_mean - psd_basal_std, psd_basal_mean + psd_basal_std, alpha=0.2)

plt.xlabel("Frecuencia (Hz)")
plt.ylabel("PSD")
plt.title(f"PSD promedio ± std, canal {canal}")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.semilogy(f_on, psd_on_mean, label="ON")
plt.semilogy(f_off, psd_off_mean, label="OFF")
plt.semilogy(f_basal, psd_basal_mean, label="Basal")
plt.xlim(0, FREQ_XLIM_LOW)
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("PSD")
plt.title(f"PSD baja frecuencia (0-{FREQ_XLIM_LOW} Hz), canal {canal}")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 9. PSD PARA TODOS LOS CANALES + ENERGÍA ESPECTRAL 1–50 Hz
# ============================================================

from scipy.integrate import trapezoid


# Parámetros banda fisiológica
FREQ_BAND_MIN = 1
FREQ_BAND_MAX = 50


def psd_todos_canales(segmentos, fs, nperseg=2048):
    """
    Calcula PSD con Welch para todos los canales y repeticiones.

    Parámetros
    ----------
    segmentos : array
        Shape: (n_reps, n_canales, T)
    fs : float
        Frecuencia de muestreo.
    nperseg : int
        Tamaño de ventana para Welch.

    Retorna
    -------
    freqs : array
        Frecuencias.
    psd_mean : array
        PSD promedio sobre repeticiones.
        Shape: (n_canales, n_freqs)
    psd_std : array
        Desviación estándar entre repeticiones.
        Shape: (n_canales, n_freqs)
    psd_reps : array
        PSD por repetición.
        Shape: (n_reps, n_canales, n_freqs)
    """

    n_reps, n_canales, _ = segmentos.shape
    psd_reps = []
    freqs = None

    for rep in range(n_reps):
        psd_canales = []

        for ch in range(n_canales):
            signal = segmentos[rep, ch, :]

            f, pxx = welch(
                signal,
                fs=fs,
                nperseg=min(nperseg, signal.shape[0])
            )

            freqs = f
            psd_canales.append(pxx)

        psd_reps.append(psd_canales)

        print(f"PSD repetición {rep + 1}/{n_reps}", flush=True)

    psd_reps = np.array(psd_reps)  # (n_reps, n_canales, n_freqs)

    psd_mean = np.mean(psd_reps, axis=0)
    psd_std = np.std(psd_reps, axis=0)

    return freqs, psd_mean, psd_std, psd_reps


def energia_espectral_banda(freqs, psd, fmin=1, fmax=50):
    """
    Integra la PSD dentro de una banda de frecuencia.

    Parámetros
    ----------
    freqs : array
        Vector de frecuencias.
    psd : array
        Shape: (n_canales, n_freqs)
    fmin, fmax : float
        Límites de la banda.

    Retorna
    -------
    energia : array
        Energía espectral por canal.
        Shape: (n_canales,)
    """

    mask = (freqs >= fmin) & (freqs <= fmax)

    if np.sum(mask) < 2:
        raise ValueError("La banda seleccionada tiene menos de 2 puntos de frecuencia.")

    energia = trapezoid(psd[:, mask], freqs[mask], axis=1)

    return energia


print("Calculando PSD todos los canales - ON...", flush=True)
f_on_all, psd_on_all, psd_on_std_all, psd_on_reps = psd_todos_canales(
    segmentos_on,
    fs,
    nperseg=PSD_NPERSEG
)

print("Calculando PSD todos los canales - OFF...", flush=True)
f_off_all, psd_off_all, psd_off_std_all, psd_off_reps = psd_todos_canales(
    segmentos_off,
    fs,
    nperseg=PSD_NPERSEG
)

print("Calculando PSD todos los canales - Basal...", flush=True)
f_basal_all, psd_basal_all, psd_basal_std_all, psd_basal_reps = psd_todos_canales(
    segmentos_basales,
    fs,
    nperseg=PSD_NPERSEG
)


# Energía espectral por canal en banda 1–50 Hz
energia_on_1_50 = energia_espectral_banda(
    f_on_all,
    psd_on_all,
    fmin=FREQ_BAND_MIN,
    fmax=FREQ_BAND_MAX
)

energia_off_1_50 = energia_espectral_banda(
    f_off_all,
    psd_off_all,
    fmin=FREQ_BAND_MIN,
    fmax=FREQ_BAND_MAX
)

energia_basal_1_50 = energia_espectral_banda(
    f_basal_all,
    psd_basal_all,
    fmin=FREQ_BAND_MIN,
    fmax=FREQ_BAND_MAX
)

delta_on_1_50 = energia_on_1_50 - energia_basal_1_50
delta_off_1_50 = energia_off_1_50 - energia_basal_1_50


# ============================================================
# 10. GRÁFICAS: ENERGÍA ESPECTRAL 1–50 Hz POR CANAL
# ============================================================

plt.figure(figsize=(9, 4))
plt.plot(energia_on_1_50, label="ON")
plt.plot(energia_off_1_50, label="OFF")
plt.plot(energia_basal_1_50, label="Basal")
plt.xlabel("Canal")
plt.ylabel("Energía espectral integrada")
plt.title("Energía espectral 1–50 Hz por canal")
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(9, 4))
plt.plot(delta_on_1_50, label="ON - Basal")
plt.plot(delta_off_1_50, label="OFF - Basal")
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Canal")
plt.ylabel("Δ energía espectral")
plt.title("Cambio de energía espectral 1–50 Hz respecto a basal")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 11. PSD PROMEDIO GLOBAL SOBRE CANALES
# ============================================================

psd_on_global = np.mean(psd_on_all, axis=0)
psd_off_global = np.mean(psd_off_all, axis=0)
psd_basal_global = np.mean(psd_basal_all, axis=0)

plt.figure(figsize=(10, 5))
plt.semilogy(f_on_all, psd_on_global, label="ON")
plt.semilogy(f_off_all, psd_off_global, label="OFF")
plt.semilogy(f_basal_all, psd_basal_global, label="Basal")
plt.xlim(0, 100)
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("PSD promedio sobre canales")
plt.title("PSD global promedio sobre canales")
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))
plt.semilogy(f_on_all, psd_on_global, label="ON")
plt.semilogy(f_off_all, psd_off_global, label="OFF")
plt.semilogy(f_basal_all, psd_basal_global, label="Basal")
plt.xlim(FREQ_BAND_MIN, FREQ_BAND_MAX)
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("PSD promedio sobre canales")
plt.title("PSD global en banda 1–50 Hz")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 12. CANALES CON MAYOR CAMBIO ESPECTRAL
# ============================================================

n_top = 10

top_on = np.argsort(delta_on_1_50)[-n_top:][::-1]
top_off = np.argsort(delta_off_1_50)[-n_top:][::-1]

print("\nCanales con mayor aumento ON - Basal en 1–50 Hz:")
for ch in top_on:
    print(f"Canal {ch}: Δ = {delta_on_1_50[ch]:.4e}")

print("\nCanales con mayor aumento OFF - Basal en 1–50 Hz:")
for ch in top_off:
    print(f"Canal {ch}: Δ = {delta_off_1_50[ch]:.4e}")


# ============================================================
# 13. COMPONENTES LENTAS POR REPETICIÓN
# ============================================================

def energia_banda_por_repeticion(freqs, psd_reps, fmin, fmax):
    """
    psd_reps shape: (n_reps, n_canales, n_freqs)
    retorna:
        energia shape: (n_reps, n_canales)
    """
    mask = (freqs >= fmin) & (freqs <= fmax)

    if np.sum(mask) < 2:
        raise ValueError("La banda tiene menos de 2 puntos de frecuencia.")

    energia = trapezoid(psd_reps[:, :, mask], freqs[mask], axis=2)   #cálculo de energía
    return energia


# Bandas lentas de interés
bandas_lentas = {
    "1-5 Hz": (1, 5),
    "5-10 Hz": (5, 10),
    "10-20 Hz": (10, 20),
    "1-20 Hz": (1, 20)
}


energia_lenta = {}

for nombre, (fmin, fmax) in bandas_lentas.items():
    energia_lenta[nombre] = {
        "ON": energia_banda_por_repeticion(f_on_all, psd_on_reps, fmin, fmax),
        "OFF": energia_banda_por_repeticion(f_off_all, psd_off_reps, fmin, fmax),
        "Basal": energia_banda_por_repeticion(f_basal_all, psd_basal_reps, fmin, fmax)
    }

    print(f"Banda {nombre}")
    print("  ON:", energia_lenta[nombre]["ON"].shape)
    print("  OFF:", energia_lenta[nombre]["OFF"].shape)
    print("  Basal:", energia_lenta[nombre]["Basal"].shape)


# ============================================================
# 14. COMPARACIÓN DE REPETICIONES EN UN CANAL
# ============================================================

canal = CANAL_PSD
banda = "1-5 Hz"

on_vals = energia_lenta[banda]["ON"][:, canal]
off_vals = energia_lenta[banda]["OFF"][:, canal]
basal_vals = energia_lenta[banda]["Basal"][:, canal]

plt.figure(figsize=(8, 4))
plt.plot(on_vals, "o-", label="ON")
plt.plot(off_vals, "o-", label="OFF")
plt.plot(basal_vals, "o-", label="Basal")
plt.xlabel("Repetición")
plt.ylabel(f"Energía espectral {banda}")
plt.title(f"Componentes lentas por repetición - canal {canal}")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 15. ENERGÍA LENTA POBLACIONAL POR REPETICIÓN
# ============================================================

banda = "1-5 Hz"

on_pop = np.mean(energia_lenta[banda]["ON"], axis=1)
off_pop = np.mean(energia_lenta[banda]["OFF"], axis=1)
basal_pop = np.mean(energia_lenta[banda]["Basal"], axis=1)

plt.figure(figsize=(8, 4))
plt.plot(on_pop, "o-", label="ON")
plt.plot(off_pop, "o-", label="OFF")
plt.plot(basal_pop, "o-", label="Basal")
plt.xlabel("Repetición")
plt.ylabel(f"Energía poblacional {banda}")
plt.title(f"Componentes lentas poblacionales por repetición")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))

plt.boxplot(
    [basal_pop, on_pop, off_pop],
    tick_labels=["Basal", "ON", "OFF"]
)
plt.ylabel(f"Energía poblacional {banda}")
plt.title(f"Distribución poblacional por repetición")
plt.tight_layout()
plt.show()

# ============================================================
# 16. ANÁLISIS DE COMPONENTES LENTAS: REPETICIONES, POBLACIÓN Y AUC
# ============================================================

from sklearn.metrics import roc_auc_score


def calcular_auc_basal_vs_condicion(basal_vals, condicion_vals):
    """
    Calcula AUC entre basal y una condición evocada.
    basal_vals: array (n_reps,)
    condicion_vals: array (n_reps,)
    """
    y_true = np.concatenate([
        np.zeros_like(basal_vals),
        np.ones_like(condicion_vals)
    ])

    scores = np.concatenate([
        basal_vals,
        condicion_vals
    ])

    return roc_auc_score(y_true, scores)


def resumen_banda_poblacional(energia_lenta, banda):
    """
    Calcula energía poblacional por repetición y AUC para una banda.
    """

    basal_pop = np.mean(energia_lenta[banda]["Basal"], axis=1)
    on_pop = np.mean(energia_lenta[banda]["ON"], axis=1)
    off_pop = np.mean(energia_lenta[banda]["OFF"], axis=1)

    auc_on = calcular_auc_basal_vs_condicion(basal_pop, on_pop)
    auc_off = calcular_auc_basal_vs_condicion(basal_pop, off_pop)

    print("\n====================================")
    print(f"Banda: {banda}")
    print("------------------------------------")
    print(f"Basal media ± std: {np.mean(basal_pop):.3f} ± {np.std(basal_pop):.3f}")
    print(f"ON    media ± std: {np.mean(on_pop):.3f} ± {np.std(on_pop):.3f}")
    print(f"OFF   media ± std: {np.mean(off_pop):.3f} ± {np.std(off_pop):.3f}")
    print(f"AUC ON vs Basal : {auc_on:.3f}")
    print(f"AUC OFF vs Basal: {auc_off:.3f}")

    return basal_pop, on_pop, off_pop, auc_on, auc_off


# ============================================================
# 17. RESUMEN POBLACIONAL POR BANDA
# ============================================================

resultados_auc_bandas = {}

for banda in bandas_lentas.keys():
    basal_pop, on_pop, off_pop, auc_on, auc_off = resumen_banda_poblacional(
        energia_lenta,
        banda
    )

    resultados_auc_bandas[banda] = {
        "basal_pop": basal_pop,
        "on_pop": on_pop,
        "off_pop": off_pop,
        "auc_on": auc_on,
        "auc_off": auc_off
    }


# ============================================================
# 18. GRÁFICA AUC POR BANDA
# ============================================================

bandas = list(resultados_auc_bandas.keys())
auc_on_vals = [resultados_auc_bandas[b]["auc_on"] for b in bandas]
auc_off_vals = [resultados_auc_bandas[b]["auc_off"] for b in bandas]

x = np.arange(len(bandas))
width = 0.35

plt.figure(figsize=(8, 4))
plt.bar(x - width/2, auc_on_vals, width, label="ON vs Basal")
plt.bar(x + width/2, auc_off_vals, width, label="OFF vs Basal")
plt.axhline(0.5, linestyle="--", linewidth=1, color="black")
plt.xticks(x, bandas)
plt.ylim(0.45, 1.05)
plt.ylabel("AUC")
plt.title("Detectabilidad poblacional por banda lenta")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 19. EVOLUCIÓN POR REPETICIÓN EN CADA BANDA
# ============================================================

for banda in bandas_lentas.keys():

    basal_pop = resultados_auc_bandas[banda]["basal_pop"]
    on_pop = resultados_auc_bandas[banda]["on_pop"]
    off_pop = resultados_auc_bandas[banda]["off_pop"]

    plt.figure(figsize=(8, 4))
    plt.plot(on_pop, "o-", label="ON")
    plt.plot(off_pop, "o-", label="OFF")
    plt.plot(basal_pop, "o-", label="Basal")
    plt.xlabel("Repetición")
    plt.ylabel(f"Energía poblacional {banda}")
    plt.title(f"Componentes lentas poblacionales por repetición - {banda}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 20. DISTRIBUCIONES POBLACIONALES POR BANDA
# ============================================================

for banda in bandas_lentas.keys():

    basal_pop = resultados_auc_bandas[banda]["basal_pop"]
    on_pop = resultados_auc_bandas[banda]["on_pop"]
    off_pop = resultados_auc_bandas[banda]["off_pop"]

    plt.figure(figsize=(6, 4))

    plt.boxplot(
    [basal_pop, on_pop, off_pop],
    tick_labels=["Basal", "ON", "OFF"]
    )
    plt.ylabel(f"Energía poblacional {banda}")
    plt.title(f"Distribución poblacional - {banda}")
    plt.tight_layout()
    plt.show()


# ============================================================
# 21. AUC POR CANAL EN UNA BANDA SELECCIONADA
# ============================================================

banda_auc = "1-5 Hz"

energia_basal = energia_lenta[banda_auc]["Basal"]  # (n_reps, n_canales)
energia_on = energia_lenta[banda_auc]["ON"]
energia_off = energia_lenta[banda_auc]["OFF"]

n_canales = energia_basal.shape[1]

auc_on_por_canal = np.zeros(n_canales)
auc_off_por_canal = np.zeros(n_canales)

for ch in range(n_canales):
    auc_on_por_canal[ch] = calcular_auc_basal_vs_condicion(
        energia_basal[:, ch],
        energia_on[:, ch]
    )

    auc_off_por_canal[ch] = calcular_auc_basal_vs_condicion(
        energia_basal[:, ch],
        energia_off[:, ch]
    )


plt.figure(figsize=(9, 4))
plt.plot(auc_on_por_canal, label="ON vs Basal")
plt.plot(auc_off_por_canal, label="OFF vs Basal")
plt.axhline(0.5, linestyle="--", linewidth=1, color="black")
plt.xlabel("Canal")
plt.ylabel("AUC")
plt.title(f"AUC por canal - banda {banda_auc}")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 22. CANALES MÁS DISCRIMINATIVOS
# ============================================================

n_top = 10

top_on_auc = np.argsort(auc_on_por_canal)[-n_top:][::-1]
top_off_auc = np.argsort(auc_off_por_canal)[-n_top:][::-1]

print(f"\nTop {n_top} canales ON vs Basal - {banda_auc}")
for ch in top_on_auc:
    print(f"Canal {ch}: AUC = {auc_on_por_canal[ch]:.3f}")

print(f"\nTop {n_top} canales OFF vs Basal - {banda_auc}")
for ch in top_off_auc:
    print(f"Canal {ch}: AUC = {auc_off_por_canal[ch]:.3f}")

# ============================================================
# 23. VALIDACIÓN: SHUFFLE, CONTROL BASAL-BASAL Y ROBUSTEZ
# ============================================================

from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt


def auc_basal_vs_condicion(basal_vals, condicion_vals):
    y_true = np.concatenate([
        np.zeros_like(basal_vals),
        np.ones_like(condicion_vals)
    ])

    scores = np.concatenate([
        basal_vals,
        condicion_vals
    ])

    return roc_auc_score(y_true, scores)


def shuffle_auc_test(basal_vals, condicion_vals, n_shuffle=1000, random_state=0):
    """
    Test por permutación de etiquetas.
    Si la señal es real, AUC real debe quedar muy por encima de la distribución shuffle.
    """
    rng = np.random.default_rng(random_state)

    scores = np.concatenate([basal_vals, condicion_vals])
    labels = np.concatenate([
        np.zeros_like(basal_vals),
        np.ones_like(condicion_vals)
    ])

    auc_real = roc_auc_score(labels, scores)

    auc_shuffle = np.zeros(n_shuffle)

    for i in range(n_shuffle):
        labels_perm = rng.permutation(labels)
        auc_shuffle[i] = roc_auc_score(labels_perm, scores)

    p_value = np.mean(auc_shuffle >= auc_real)

    return auc_real, auc_shuffle, p_value


def basal_vs_basal_control(basal_matrix, n_iter=1000, random_state=1):
    """
    Control negativo: divide repeticiones basales en dos grupos aleatorios.
    El AUC esperado debe estar cerca de 0.5.
    
    basal_matrix shape: (n_reps, n_canales)
    """
    rng = np.random.default_rng(random_state)

    basal_pop = np.mean(basal_matrix, axis=1)
    n = len(basal_pop)

    aucs = []

    for _ in range(n_iter):
        idx = rng.permutation(n)

        mitad = n // 2
        grupo_1 = basal_pop[idx[:mitad]]
        grupo_2 = basal_pop[idx[mitad:]]

        if len(grupo_1) < 2 or len(grupo_2) < 2:
            continue

        aucs.append(auc_basal_vs_condicion(grupo_1, grupo_2))

    return np.array(aucs)


# ============================================================
# 24. APLICAR VALIDACIÓN A BANDA PRINCIPAL
# ============================================================

banda_validacion = "1-5 Hz"

basal_matrix = energia_lenta[banda_validacion]["Basal"]
on_matrix = energia_lenta[banda_validacion]["ON"]
off_matrix = energia_lenta[banda_validacion]["OFF"]

# Promedio poblacional por repetición
basal_pop = np.mean(basal_matrix, axis=1)
on_pop = np.mean(on_matrix, axis=1)
off_pop = np.mean(off_matrix, axis=1)

# AUC real + shuffle
auc_on_real, auc_on_shuffle, p_on = shuffle_auc_test(
    basal_pop,
    on_pop,
    n_shuffle=1000,
    random_state=10
)

auc_off_real, auc_off_shuffle, p_off = shuffle_auc_test(
    basal_pop,
    off_pop,
    n_shuffle=1000,
    random_state=11
)

# Control basal vs basal
auc_basal_control = basal_vs_basal_control(
    basal_matrix,
    n_iter=1000,
    random_state=12
)

print("\n====================================")
print(f"VALIDACIÓN BANDA {banda_validacion}")
print("------------------------------------")
print(f"AUC real ON vs Basal : {auc_on_real:.3f}")
print(f"p shuffle ON         : {p_on:.4f}")
print(f"AUC real OFF vs Basal: {auc_off_real:.3f}")
print(f"p shuffle OFF        : {p_off:.4f}")
print("------------------------------------")
print(f"Control basal-basal AUC media ± std: "
      f"{np.mean(auc_basal_control):.3f} ± {np.std(auc_basal_control):.3f}")


# ============================================================
# 25. GRAFICAR DISTRIBUCIONES SHUFFLE
# ============================================================

plt.figure(figsize=(7, 4))
plt.hist(auc_on_shuffle, bins=30, alpha=0.7, label="Shuffle ON")
plt.axvline(auc_on_real, color="black", linestyle="--", linewidth=2, label="AUC real ON")
plt.axvline(0.5, color="gray", linestyle=":", linewidth=2, label="Azar")
plt.xlabel("AUC")
plt.ylabel("Frecuencia")
plt.title(f"Shuffle test ON vs Basal - {banda_validacion}")
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(7, 4))
plt.hist(auc_off_shuffle, bins=30, alpha=0.7, label="Shuffle OFF")
plt.axvline(auc_off_real, color="black", linestyle="--", linewidth=2, label="AUC real OFF")
plt.axvline(0.5, color="gray", linestyle=":", linewidth=2, label="Azar")
plt.xlabel("AUC")
plt.ylabel("Frecuencia")
plt.title(f"Shuffle test OFF vs Basal - {banda_validacion}")
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(7, 4))
plt.hist(auc_basal_control, bins=30, alpha=0.8)
plt.axvline(0.5, color="gray", linestyle=":", linewidth=2, label="Azar")
plt.xlabel("AUC")
plt.ylabel("Frecuencia")
plt.title(f"Control negativo Basal vs Basal - {banda_validacion}")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 23. d' (d-prime) - DEFINICIÓN
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

def d_prime(x0, x1, eps=1e-12):
    """
    x0: basal (array)
    x1: condición (ON u OFF)
    """
    mu0, mu1 = np.mean(x0), np.mean(x1)
    var0, var1 = np.var(x0, ddof=1), np.var(x1, ddof=1)

    pooled_std = np.sqrt(0.5 * (var0 + var1) + eps)

    return (mu1 - mu0) / pooled_std

# ============================================================
# 24. d' POBLACIONAL POR BANDA
# ============================================================

resultados_dprime = {}

for banda in bandas_lentas.keys():

    basal_pop = np.mean(energia_lenta[banda]["Basal"], axis=1)
    on_pop = np.mean(energia_lenta[banda]["ON"], axis=1)
    off_pop = np.mean(energia_lenta[banda]["OFF"], axis=1)

    d_on = d_prime(basal_pop, on_pop)
    d_off = d_prime(basal_pop, off_pop)

    resultados_dprime[banda] = {
        "d_on": d_on,
        "d_off": d_off
    }

    print("\n====================================")
    print(f"Banda: {banda}")
    print(f"d' ON  vs Basal: {d_on:.3f}")
    print(f"d' OFF vs Basal: {d_off:.3f}")

# ============================================================
# 25. VISUALIZACIÓN d' POR BANDA
# ============================================================

bandas = list(resultados_dprime.keys())
d_on_vals = [resultados_dprime[b]["d_on"] for b in bandas]
d_off_vals = [resultados_dprime[b]["d_off"] for b in bandas]

x = np.arange(len(bandas))
width = 0.35

plt.figure(figsize=(8,4))
plt.bar(x - width/2, d_on_vals, width, label="ON vs Basal")
plt.bar(x + width/2, d_off_vals, width, label="OFF vs Basal")

plt.xticks(x, bandas)
plt.ylabel("d'")
plt.title("Separación (d') por banda lenta")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 26. d' POR CANAL
# ============================================================

banda_dprime = "1-5 Hz"

basal = energia_lenta[banda_dprime]["Basal"]   # (n_reps, n_canales)
on = energia_lenta[banda_dprime]["ON"]
off = energia_lenta[banda_dprime]["OFF"]

n_canales = basal.shape[1]

d_on_ch = np.zeros(n_canales)
d_off_ch = np.zeros(n_canales)

for ch in range(n_canales):
    d_on_ch[ch] = d_prime(basal[:, ch], on[:, ch])
    d_off_ch[ch] = d_prime(basal[:, ch], off[:, ch])

plt.figure(figsize=(9,4))
plt.plot(d_on_ch, label="ON vs Basal")
plt.plot(d_off_ch, label="OFF vs Basal")
plt.axhline(0, linestyle="--", color="black")
plt.xlabel("Canal")
plt.ylabel("d'")
plt.title(f"d' por canal - {banda_dprime}")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 27. TOP CANALES
# ============================================================

n_top = 10

top_on = np.argsort(d_on_ch)[-n_top:][::-1]
top_off = np.argsort(d_off_ch)[-n_top:][::-1]

print(f"\nTop {n_top} canales ON - {banda_dprime}")
for ch in top_on:
    print(f"Canal {ch}: d' = {d_on_ch[ch]:.3f}")

print(f"\nTop {n_top} canales OFF - {banda_dprime}")
for ch in top_off:
    print(f"Canal {ch}: d' = {d_off_ch[ch]:.3f}")
