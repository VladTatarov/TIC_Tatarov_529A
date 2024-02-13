import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, fftfreq
from scipy import signal

# Задані параметри сигналу
n = 500
Fs = 1000
F_max = 25

# Розміри фігури та графіку
width_cm = 21
height_cm = 14
fig, ax = plt.subplots(figsize=(width_cm/2.54, height_cm/2.54))

# Генерація сигналу
t = np.arange(n) / Fs
x = np.sin(2 * np.pi * F_max * t)

# Відображення сигналу
ax.plot(t, x, linewidth=1)
ax.set_xlabel('Час (с)', fontsize=14)
ax.set_ylabel('Амплітуда', fontsize=14)
plt.title('Згенерований сигнал', fontsize=14)

# Збереження зображення сигналу
fig.savefig('./figures/signal.png', dpi=600)

# Обчислення та відображення спектру сигналу
spectrum = fftshift(fft(x))
freqs = fftshift(fftfreq(n, 1/Fs))
fig, ax = plt.subplots(figsize=(width_cm/2.54, height_cm/2.54))

ax.plot(freqs, np.abs(spectrum), linewidth=1)
ax.set_xlabel('Частота (Гц)', fontsize=14)
ax.set_ylabel('Амплітуда', fontsize=14)
plt.title('Спектр сигналу', fontsize=14)

# Збереження зображення спектру
fig.savefig('./figures/spectrum.png', dpi=600)

# Розрахунок параметрів ФНЧ
w = F_max / (Fs / 2)
order = 3  # Порядок фільтру
sos = signal.butter(order, w, 'low', output='sos')

# Фільтрація сигналу
filtered_signal = signal.sosfiltfilt(sos, x)

# Відображення результатів фільтрації
fig, ax = plt.subplots(figsize=(width_cm/2.54, height_cm/2.54))
ax.plot(t, filtered_signal, linewidth=1)
ax.set_xlabel('Час (с)', fontsize=14)
ax.set_ylabel('Амплітуда', fontsize=14)
plt.title('Фільтрований сигнал', fontsize=14)

plt.show()
