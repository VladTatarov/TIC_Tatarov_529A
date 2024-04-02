import collections
import math
import random
import string
import matplotlib.pyplot as plt

open("results_sequence.txt", "w")
open("sequence.txt", "w")

# Загальні змінні
student_number = 12
group_number = "529"
N_sequence = 100

# Генерація послідовностей
sequences = []

# Перша послідовність
sequence_1 = ['1'] * student_number + ['0'] * (N_sequence - student_number)
random.shuffle(sequence_1)
sequences.append(sequence_1)

# Друга послідовність
surname = "Татаров"
sequence_2 = list(surname) + ['0'] * (N_sequence - len(surname))
sequences.append(sequence_2)

# Третя послідовність
sequence_3 = list(surname) + ['0'] * (N_sequence - len(surname))
random.shuffle(sequence_3)
sequences.append(sequence_3)

# Четверта послідовність
letters = list(surname) + list(group_number)
n_letters = len(letters)
n_repeats = N_sequence // n_letters
remainder = N_sequence % n_letters
sequence_4 = letters * n_repeats + letters[:remainder]
random.shuffle(sequence_4)
sequences.append(sequence_4)

# П'ята послідовність
letters_5 = list(surname[:2]) + list(group_number)
probability = 0.2
sequence_5 = letters_5 * 20
random.shuffle(sequence_5)
sequences.append(sequence_5)

# Шоста послідовність
letters_6 = list(surname[:2]) + list(group_number)
digits_6 = [str(i) for i in range(10)]
total_probability_letters = 0.7
total_probability_digits = 0.3
n_letters_6 = int(total_probability_letters * N_sequence)
n_digits_6 = int(total_probability_digits * N_sequence)
sequence_6 = random.choices(letters_6, k=n_letters_6) + random.choices(digits_6, k=n_digits_6)
random.shuffle(sequence_6)
sequences.append(sequence_6)

# Сьома послідовність
elements_7 = string.ascii_lowercase + string.digits
sequence_7 = [random.choice(elements_7) for _ in range(N_sequence)]
sequences.append(sequence_7)

# Восьма послідовність
original_sequence_8 = '1' * N_sequence
sequences.append(list(original_sequence_8))

# Обчислення характеристик
results = []
with open("results_sequence.txt", "a", encoding="utf-8") as file:
    for i, sequence in enumerate(sequences, start=1):
        sequence_str = ''.join(sequence)
        counts = collections.Counter(sequence)
        N_sequence = len(sequence)
        probability = {symbol: count / N_sequence for symbol, count in counts.items()}
        mean_probability = sum(probability.values()) / len(probability)
        equal = all(abs(prob - mean_probability) < 0.05 * mean_probability for prob in probability.values())
        uniformity = "рівна" if equal else "нерівна"
        entropy = -sum(p * math.log2(p) for p in probability.values())
        sequence_alphabet_size = len(set(sequence))
        if sequence_alphabet_size > 1:
            source_excess = 1 - entropy / math.log2(sequence_alphabet_size)
        else:
            source_excess = 1
        results.append([
            sequence_alphabet_size,
            round(entropy, 2),
            round(source_excess, 2),
            uniformity
        ])
        file.write(f"Послідовність: {sequence_str}\n")
        file.write(f"Розмір послідовності: {len(sequence_str)} byte\n")
        file.write(f"Розмір алфавіту: {sequence_alphabet_size}\n")
        file.write("Ймовірності появи символів: ")
        probs_str = ", ".join([f"{symbol}={prob:.4f}" for symbol, prob in probability.items()])
        file.write(probs_str + "\n")
        file.write(f"Середнє арифметичне ймовірностей: {mean_probability:.2f}\n")
        file.write(f"Ймовірність розподілу символів: {uniformity}\n")
        file.write(f"Ентропія: {entropy:.4f}\n")
        file.write(f"Надмірність джерела: {source_excess:.2f}\n\n")

# Створення таблиці
headers = ['Розмір алфавіту', 'Ентропія', 'Надмірність', 'Ймовірність']
row_labels = [f"Послідовність {i}" for i in range(1, len(results) + 1)]

fig, ax = plt.subplots(figsize=(10, len(results) * 0.6))
ax.axis('off')
table = ax.table(cellText=results, colLabels=headers, rowLabels=row_labels, loc='center', cellLoc='center')
table.set_fontsize(12)
table.scale(1.2, 1.2)
fig.savefig("Характеристики_послідовностей.png")
