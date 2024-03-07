import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, fftfreq
from scipy import signal

# Задані параметри сигнала
n = 500
Fs = 1000
F_max = 25
F_filter = 32

# Списки для збереження результатів
discrete_signals = []
discrete_spectrums = []
restored_signals = []
variances = []
snr_ratios = []

# Розміри фігури та графіка
width_cm = 21
height_cm = 14

# Графік для відображення сигналу
fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54))
t = np.arange(n) / Fs
x = np.sin(2 * np.pi * F_max * t)
ax.plot(t, x, linewidth=1)
ax.set_xlabel('Час (с)', fontsize=14)
ax.set_ylabel('Амплітуда', fontsize=14)
plt.title('Згенерований сигнал', fontsize=14)

# Збереження зображення сигналу
fig.savefig('./figures/signal.png', dpi=600)

# Розрахунок та відображення спектра сигналу
fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54))
spectrum = fftshift(fft(x))
freqs = fftshift(fftfreq(n, 1 / Fs))
ax.plot(freqs, np.abs(spectrum), linewidth=1)
ax.set_xlabel('Частота (Гц)', fontsize=14)
ax.set_ylabel('Амплітуда', fontsize=14)
plt.title('Спектр сигналу', fontsize=14)

# Збереження зображення спектра
fig.savefig('./figures/spectrum.png', dpi=600)

# Розрахунок параметрів ФНЧ
w = F_max / (Fs / 2)
order = 3
sos = signal.butter(order, w, 'low', output='sos')

# Фільтрація сигналу
filtered_signal = signal.sosfiltfilt(sos, x)

# Відображення результатів фільтрації
fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54))
ax.plot(t, filtered_signal, linewidth=1)
ax.set_xlabel('Час (с)', fontsize=14)
ax.set_ylabel('Амплітуда', fontsize=14)
plt.title('Фільтрований сигнал', fontsize=14)

plt.show()

# Цикл дискретизації з різними кроками
for Dt in [2, 4, 8, 16]:
    t = np.arange(n) / Fs
    x = np.sin(2 * np.pi * F_max * t)

    # Дискретизація сигналу
    discrete_signal = x[::Dt]
    discrete_signals.append(list(discrete_signal))

    # Resampling the signal
    resampled_signal = np.interp(np.arange(int(n / Dt)) * Dt / n, np.arange(n), x)
    discrete_spectrums.append(np.abs(fftshift(fft(resampled_signal, n))))

    # Відновлення сигналу за допомогою фільтрації
    w = F_filter / (Fs / 2)
    sos = signal.butter(3, w, 'low', output='sos')
    restored_signal = signal.sosfiltfilt(sos, resampled_signal)
    restored_signals.append(list(restored_signal))

    # Розрахунок різниці між початковим та відновленим сигналом
    E1 = restored_signal[:len(restored_signal)] - x[:len(restored_signal)]

    # Розрахунок дисперсії та відношення сигнал-шум
    var_signal = np.var(x)
    var_diff = np.var(E1)
    variances.append(var_diff)
    snr_ratio = var_signal / var_diff
    snr_ratios.append(snr_ratio)

# Побудова графіків
plt.figure(figsize=(10, 6))
plt.plot([2, 4, 8, 16], variances, marker='o')
plt.xlabel('Крок дискретизації (Dt)', fontsize=14)
plt.ylabel('Дисперсія різниці', fontsize=14)
plt.title('Залежність дисперсії різниці відновленого сигналу від кроку дискретизації', fontsize=14)
plt.grid(True)
plt.savefig('./figures/dispersion_vs_dt.png', dpi=600)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot([2, 4, 8, 16], snr_ratios, marker='o')
plt.xlabel('Крок дискретизації (Dt)', fontsize=14)
plt.ylabel('Співвідношення сигнал-шум', fontsize=14)
plt.title('Залежність співвідношення сигнал-шум від кроку дискретизації', fontsize=14)
plt.grid(True)
plt.savefig('./figures/snr_vs_dt.png', dpi=600)
plt.show()

# Побудова графіків спектрів дискретизованих сигналів
for i, spectrum in enumerate(discrete_spectrums):
    fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54))
    ax.plot(freqs, np.abs(spectrum), linewidth=1)
    ax.set_xlabel('Частота (Гц)', fontsize=14)
    ax.set_ylabel('Амплітуда', fontsize=14)
    plt.title(f'Спектр дискретизованого сигналу (Dt = {2**(i+1)})', fontsize=14)
    plt.grid(True)
    plt.savefig(f'./figures/discrete_spectrum_{2**(i+1)}.png', dpi=600)
    plt.show()


# Побудова графіків відновлених сигналів
for i, restored_signal in enumerate(restored_signals):
    fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54))
    ax.plot(t[:len(restored_signal)], restored_signal, linewidth=1)
    ax.set_xlabel('Час (с)', fontsize=14)
    ax.set_ylabel('Амплітуда', fontsize=14)
    plt.title(f'Відновлений сигнал (Dt = {2**(i+1)})', fontsize=14)
    plt.grid(True)
    plt.savefig(f'./figures/restored_signal_{2**(i+1)}.png', dpi=600)
    plt.show()

# Побудова графіків дискретизованих сигналів
for i, discrete_signal in enumerate(discrete_signals):
    fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54))
    ax.plot(np.arange(len(discrete_signal)) / Fs * (2 ** (i + 1)), discrete_signal, linewidth=1)
    ax.set_xlabel('Час (с)', fontsize=14)
    ax.set_ylabel('Амплітуда', fontsize=14)
    plt.title(f'Дискретизований сигнал (Dt = {2**(i+1)})', fontsize=14)
    plt.grid(True)
    plt.savefig(f'./figures/discrete_signal_{2**(i+1)}.png', dpi=600)
    plt.show()
