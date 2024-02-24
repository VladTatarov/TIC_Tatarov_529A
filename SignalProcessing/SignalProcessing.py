import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, fftfreq
from scipy import signal

# Заданные параметры сигнала
n = 500
Fs = 1000
F_max = 25
F_filter = 32

# Списки для сохранения результатов
discrete_signals = []
discrete_spectrums = []
restored_signals = []
variances = []
snr_ratios = []

# Размеры фигуры и графика
width_cm = 21
height_cm = 14

# График для отображения сигнала
fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54))
t = np.arange(n) / Fs
x = np.sin(2 * np.pi * F_max * t)
ax.plot(t, x, linewidth=1)
ax.set_xlabel('Час (с)', fontsize=14)
ax.set_ylabel('Амплітуда', fontsize=14)
plt.title('Згенерований сигнал', fontsize=14)

# Сохранение изображения сигнала
fig.savefig('./figures/signal.png', dpi=600)

# Расчет и отображение спектра сигнала
fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54))
spectrum = fftshift(fft(x))
freqs = fftshift(fftfreq(n, 1 / Fs))
ax.plot(freqs, np.abs(spectrum), linewidth=1)
ax.set_xlabel('Частота (Гц)', fontsize=14)
ax.set_ylabel('Амплітуда', fontsize=14)
plt.title('Спектр сигналу', fontsize=14)

# Сохранение изображения спектра
fig.savefig('./figures/spectrum.png', dpi=600)

# Расчет параметров ФНЧ
w = F_max / (Fs / 2)
order = 3
sos = signal.butter(order, w, 'low', output='sos')

# Фильтрация сигнала
filtered_signal = signal.sosfiltfilt(sos, x)

# Отображение результатов фильтрации
fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54))
ax.plot(t, filtered_signal, linewidth=1)
ax.set_xlabel('Час (с)', fontsize=14)
ax.set_ylabel('Амплітуда', fontsize=14)
plt.title('Фільтрований сигнал', fontsize=14)

plt.show()

# Цикл дискретизации с разными шагами
for Dt in [2, 4, 8, 16]:
    t = np.arange(n) / Fs
    x = np.sin(2 * np.pi * F_max * t)

    # Дискретизация сигнала
    discrete_signal = np.pad(x[::Dt], (0, len(x) - len(x[::Dt])), mode='constant')
    discrete_signals.append(list(discrete_signal))

    # Расчет спектра дискретизированного сигнала
    spectrum = fftshift(fft(discrete_signal))
    freqs = fftshift(fftfreq(n, 1 / Fs))
    discrete_spectrums.append(list(np.abs(spectrum)))

    # Восстановление сигнала с помощью фильтрации
    w = F_filter / (Fs / 2)
    sos = signal.butter(3, w, 'low', output='sos')
    restored_signal = signal.sosfiltfilt(sos, discrete_signal)
    restored_signals.append(list(restored_signal))

    # Расчет разницы между начальным и восстановленным сигналом
    E1 = restored_signal[:len(restored_signal)] - x[:len(restored_signal)]

    # Расчет дисперсии и отношения сигнал-шум
    var_signal = np.var(x)
    var_diff = np.var(E1)
    variances.append(var_diff)
    snr_ratio = var_signal / var_diff
    snr_ratios.append(snr_ratio)

# Построение графиков
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

# Построение графиков спектров дискретизированных сигналов
for i, spectrum in enumerate(discrete_spectrums):
    fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54))
    ax.plot(freqs, np.abs(spectrum), linewidth=1)
    ax.set_xlabel('Частота (Гц)', fontsize=14)
    ax.set_ylabel('Амплітуда', fontsize=14)
    plt.title(f'Спектр дискретизованого сигналу (Dt = {2**(i+1)})', fontsize=14)
    plt.grid(True)
    plt.savefig(f'./figures/discrete_spectrum_{2**(i+1)}.png', dpi=600)
    plt.show()


# Построение графиков восстановленных сигналов
for i, restored_signal in enumerate(restored_signals):
    fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54))
    ax.plot(t, restored_signal, linewidth=1)
    ax.set_xlabel('Час (с)', fontsize=14)
    ax.set_ylabel('Амплітуда', fontsize=14)
    plt.title(f'Відновлений сигнал (Dt = {2**(i+1)})', fontsize=14)
    plt.grid(True)
    plt.savefig(f'./figures/restored_signal_{2**(i+1)}.png', dpi=600)
    plt.show()
