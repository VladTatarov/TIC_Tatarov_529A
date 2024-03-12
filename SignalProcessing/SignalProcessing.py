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
discrete_variances = []
discrete_snr_ratios = []
quantization_variances = []
quantization_snr_ratios = []
quantized_signals = []
bits_sequences = []

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
    discrete_variances.append(var_diff)
    snr_ratio = var_signal / var_diff
    discrete_snr_ratios.append(snr_ratio)

# Побудова графіків
plt.figure(figsize=(10, 6))
plt.plot([2, 4, 8, 16], discrete_variances, marker='o')
plt.xlabel('Крок дискретизації (Dt)', fontsize=14)
plt.ylabel('Дисперсія різниці', fontsize=14)
plt.title('Залежність дисперсії різниці відновленого сигналу від кроку дискретизації', fontsize=14)
plt.grid(True)
plt.savefig('./figures/dispersion_vs_dt.png', dpi=600)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot([2, 4, 8, 16], discrete_snr_ratios, marker='o')
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
    plt.title(f'Спектр дискретизованого сигналу (Dt = {2 ** (i + 1)})', fontsize=14)
    plt.grid(True)
    plt.savefig(f'./figures/discrete_spectrum_{2 ** (i + 1)}.png', dpi=600)
    plt.show()

# Побудова графіків відновлених сигналів
for i, restored_signal in enumerate(restored_signals):
    fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54))
    ax.plot(t[:len(restored_signal)], restored_signal, linewidth=1)
    ax.set_xlabel('Час (с)', fontsize=14)
    ax.set_ylabel('Амплітуда', fontsize=14)
    plt.title(f'Відновлений сигнал (Dt = {2 ** (i + 1)})', fontsize=14)
    plt.grid(True)
    plt.savefig(f'./figures/restored_signal_{2 ** (i + 1)}.png', dpi=600)
    plt.show()

# Побудова графіків дискретизованих сигналів
for i, discrete_signal in enumerate(discrete_signals):
    fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54))
    ax.plot(np.arange(len(discrete_signal)) / Fs * (2 ** (i + 1)), discrete_signal, linewidth=1)
    ax.set_xlabel('Час (с)', fontsize=14)
    ax.set_ylabel('Амплітуда', fontsize=14)
    plt.title(f'Дискретизований сигнал (Dt = {2 ** (i + 1)})', fontsize=14)
    plt.grid(True)
    plt.savefig(f'./figures/discrete_signal_{2 ** (i + 1)}.png', dpi=600)
    plt.show()

# Квантування сигналу на M рівнях
for M in [4, 16, 64, 256]:
    # Генерація сигналу
    t = np.arange(n) / Fs
    x = np.sin(2 * np.pi * F_max * t)

    # Обчислення кроку квантування
    delta = (np.max(x) - np.min(x)) / (M - 1)

    # Квантування сигналу
    quantized_signal = delta * np.round(x / delta)
    quantized_signals.append(quantized_signal)

    # Розрахунок дисперсії та співвідношення сигнал-шум
    var_signal = np.var(x)
    var_diff = np.var(quantized_signal - x[:len(quantized_signal)])
    snr_ratio = var_signal / var_diff

    # Збереження значень дисперсії та співвідношення сигнал-шум
    quantization_variances.append(var_diff)
    quantization_snr_ratios.append(snr_ratio)

    # Побудова бітової послідовності
    bits = []
    quantize_levels = np.arange(np.min(quantized_signal), np.max(quantized_signal) + 1, delta)
    quantize_bit = np.arange(0, M)
    for signal_value in quantized_signal:
        for index, value in enumerate(quantize_levels[:M]):
            if np.round(np.abs(signal_value - value), 0) == 0:
                bits.append(quantize_bit[index])
                break
    bits = [int(item) for item in list(''.join([format(bits, '0' + str(int(np.log2(M))) + 'b') for bits in bits]))]
    bits_sequences.append(bits)

# Побудова таблиць квантування та рисунка з різними рівнями квантування
for i in range(len(quantized_signals)):
    plt.figure(figsize=(8, 6))
    plt.plot(t, quantized_signals[i], label=f'M = {len(quantized_signals[i])}', linewidth=1)
    plt.xlabel('Час (с)', fontsize=14)
    plt.ylabel('Амплітуда', fontsize=14)
    plt.title('Квантований сигнал', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./figures/quantized_signal_M_{len(quantized_signals[i])}.png', dpi=600)
    plt.show()

    print(f"Таблиця квантування для M = {len(quantized_signals[i])}:")
    print("----------------------------------------------------")
    print("|  Значення сигналу  |  Квантоване значення  |")
    print("----------------------------------------------------")
    for j in range(len(quantized_signals[i])):
        print(f"|        {x[j]:.3f}         |           {quantized_signals[i][j]:.3f}          |")
    print("----------------------------------------------------")

# Побудова графіка бітової послідовності
plt.figure(figsize=(10, 6))
plt.step(np.arange(0, len(bits)), bits, linewidth=0.1)
plt.xlabel('Час (відліки)', fontsize=14)
plt.ylabel('Біти', fontsize=14)
plt.title(f'Бітова послідовність при квантуванні на {M} рівнях', fontsize=14)
plt.grid(True)
plt.savefig(f'./figures/bit_sequence_{M}_levels.png', dpi=600)
plt.show()

# Розрахунок дисперсії та співвідношення сигнал-шум
var_signal = np.var(x)
var_diff = np.var(quantized_signal - x[:len(quantized_signal)])
snr_ratio = var_signal / var_diff

# Побудова графіка залежності дисперсії від кількості рівнів квантування
quantization_variances.append(var_diff)
quantization_snr_ratios.append(snr_ratio)

# Видалення останнього елемента з кожного списку
quantization_variances.pop()
quantization_snr_ratios.pop()

# Побудова рисунка залежності дисперсії від кількості рівнів квантування
plt.figure(figsize=(10, 6))
plt.plot([4, 16, 64, 256], quantization_variances, marker='o')
plt.xlabel('Кількість рівнів квантування (M)', fontsize=14)
plt.ylabel('Дисперсія різниці', fontsize=14)
plt.title('Залежність дисперсії різниці від кількості рівнів квантування', fontsize=14)
plt.grid(True)
plt.savefig('./figures/dispersion_vs_quantization_levels.png', dpi=600)
plt.show()

# Побудова рисунка залежності співвідношення сигнал-шум від кількості рівнів квантування
plt.figure(figsize=(10, 6))
plt.plot([4, 16, 64, 256], quantization_snr_ratios, marker='o')
plt.xlabel('Кількість рівнів квантування (M)', fontsize=14)
plt.ylabel('Співвідношення сигнал-шум', fontsize=14)
plt.title('Залежність співвідношення сигнал-шум від кількості рівнів квантування', fontsize=14)
plt.grid(True)
plt.savefig('./figures/snr_vs_quantization_levels.png', dpi=600)
plt.show()
