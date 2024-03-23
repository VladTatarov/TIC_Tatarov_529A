import collections
import math
import random
import string
import matplotlib.pyplot as plt

# Мій порядковий номер
student_number = 12

# Номер моєї групи
group_number = "529-А"

# Розмірність послідовності
N_sequence = 100

# Кількість елементів "1"
N1 = student_number

# Заповнення списків "1" та "0" для першої послідовності
list1 = ['1'] * N1
list0 = ['0'] * (N_sequence - N1)

# Об'єднання та перемішування списків для першої послідовності
sequence_1 = list1 + list0
random.shuffle(sequence_1)

# Представлення першої послідовності у вигляді рядка
original_sequence_1 = ''.join(sequence_1)

# Збереження результатів першої послідовності у файл results_sequence.txt
with open("D:/PycharmProjects/TIC/LosslessСompression/results_sequence.txt", "a", encoding="utf-8") as file:
    file.write("Test Sequence 1:\n")
    file.write(f"Original Sequence: {original_sequence_1}\n")
    file.write(f"Sequence Alphabet Size: 2\n")  # "0" та "1"
    file.write(f"Original Sequence Size: {len(original_sequence_1)} [bits]\n\n")

# Моє прізвище
surname = "Татаров"

# Елементи алфавіту (букви мого прізвища та "0")
alphabet = set(surname)  # отримуємо унікальні символи з прізвища
alphabet.add('0')  # додаємо символ "0"

# Кількість елементів прізвища у другій послідовності
N2 = len(surname)

# Заповнення списків символами прізвища та "0" для другої послідовності
list1 = list(surname)
list0 = ['0'] * (N_sequence - N2)

# Об'єднання списків для другої послідовності
sequence_2 = list1 + list0

# Представлення другої послідовності у вигляді рядка
original_sequence_2 = ''.join(sequence_2)

# Збереження результатів другої послідовності у файл results_sequence.txt
with open("D:/PycharmProjects/TIC/LosslessСompression/results_sequence.txt", "a", encoding="utf-8") as file:
    file.write("Test Sequence 2:\n")
    file.write(f"Original Sequence: {original_sequence_2}\n")
    file.write(f"Sequence Alphabet Size: {len(alphabet)}\n")
    file.write(f"Original Sequence Size: {len(original_sequence_2)} [bits]\n\n")

# Генерація третьої послідовності
# Заповнення списків символами прізвища та "0" для третьої послідовності
list1 = list(surname)
list0 = ['0'] * (N_sequence - len(list1))

# Об'єднання списків для третьої послідовності
sequence_3 = list1 + list0

# Перемішування елементів третьої послідовності
random.shuffle(sequence_3)

# Представлення третьої послідовності у вигляді рядка
original_sequence_3 = ''.join(sequence_3)

# Збереження результатів третьої послідовності у файл results_sequence.txt
with open("D:/PycharmProjects/TIC/LosslessСompression/results_sequence.txt", "a", encoding="utf-8") as file:
    file.write("Test Sequence 3:\n")
    file.write(f"Original Sequence: {original_sequence_3}\n")
    file.write(f"Sequence Alphabet Size: {len(alphabet)}\n")
    file.write(f"Original Sequence Size: {len(original_sequence_3)} [bits]\n\n")

# Створення списку з букв вашого прізвища та цифр номера групи
letters = list(surname) + list(group_number)

# Визначення довжини списку букв та цифр
n_letters = len(letters)

# Визначення кількості повторів всього списку букв та цифр у межах розміру формованої послідовності
n_repeats = N_sequence // n_letters

# Визначення залишку елементів списку букв та цифр
remainder = N_sequence % n_letters

# Створення списку з буквами та цифрами шляхом множення елементів списку на кількість повторів
sequence_4 = letters * n_repeats

# Додавання залишку елементів до списку
sequence_4 += letters[:remainder]

# Перемішування елементів послідовності
random.shuffle(sequence_4)

# Представлення послідовності у вигляді рядка
original_sequence_4 = ''.join(sequence_4)

# Збереження результатів у файл results_sequence.txt
with open("D:/PycharmProjects/TIC/LosslessСompression/results_sequence.txt", "a", encoding="utf-8") as file:
    file.write("Test Sequence 4:\n")
    file.write(f"Original Sequence: {original_sequence_4}\n")
    file.write(f"Sequence Alphabet Size: {len(set(letters))}\n")
    file.write(f"Original Sequence Size: {len(original_sequence_4)} [bits]\n\n")

# Генерація п'ятої послідовності
# Список з буквами прізвища та цифрами номера групи
letters_5 = list(surname[:2]) + list(group_number)

# Ймовірність появи будь-якого елементу
probability = 0.2

# Згенерувати послідовність розмірності N_sequence
sequence_5 = random.choices(letters_5, weights=[probability] * len(letters_5), k=N_sequence)

# Обчислення ймовірностей кожного символу у послідовності
counts_5 = collections.Counter(sequence_5)
probability_5 = {symbol: count / N_sequence for symbol, count in counts_5.items()}

# Перемішати послідовність
random.shuffle(sequence_5)

# Представлення послідовності у вигляді рядка
original_sequence_5 = ''.join(sequence_5)

# Збереження результатів у файл results_sequence.txt
with open("D:/PycharmProjects/TIC/LosslessСompression/results_sequence.txt", "a", encoding="utf-8") as file:
    file.write("Test Sequence 5:\n")
    file.write(f"Original Sequence: {original_sequence_5}\n")
    file.write(f"Sequence Alphabet Size: {len(set(letters_5))}\n")
    file.write(f"Original Sequence Size: {len(original_sequence_5)} [bits]\n")
    file.write("Symbol Probabilities:\n")
    for symbol, prob in probability_5.items():
        file.write(f"{symbol}: {prob}\n")
    file.write("\n")

# Тестова послідовність №6
# Список букв та цифр
letters_6 = list(surname[:2]) + list(group_number)
digits_6 = [str(i) for i in range(10)]

# Сумарна ймовірність для букв та цифр
total_probability_letters = 0.7
total_probability_digits = 0.3

# Визначення кількості елементів для букв та цифр
n_letters_6 = int(total_probability_letters * N_sequence)
n_digits_6 = int(total_probability_digits * N_sequence)

# Створення списку з буквами та цифрами
sequence_6 = []
for i in range(n_letters_6):
    sequence_6.append(random.choice(letters_6))
for i in range(n_digits_6):
    sequence_6.append(random.choice(digits_6))

# Перемішування елементів
random.shuffle(sequence_6)

# Представлення послідовності у вигляді рядка
original_sequence_6 = ''.join(sequence_6)

# Збереження результатів у файл results_sequence.txt
with open("D:/PycharmProjects/TIC/LosslessСompression/results_sequence.txt", "a", encoding="utf-8") as file:
    file.write("Test Sequence 6:\n")
    file.write(f"Original Sequence: {original_sequence_6}\n")
    file.write(f"Sequence Alphabet Size: {len(set(letters_6))}\n")
    file.write(f"Original Sequence Size: {len(original_sequence_6)} [bits]\n\n")

# Генерація сьомої послідовності
# Список з символами англійського алфавіту та цифрами 0..9
elements_7 = string.ascii_lowercase + string.digits

# Створення списку з випадково обраними елементами
sequence_7 = [random.choice(elements_7) for _ in range(N_sequence)]

# Представлення послідовності у вигляді рядка
original_sequence_7 = ''.join(sequence_7)

# Збереження результатів у файл results_sequence.txt
with open("D:/PycharmProjects/TIC/LosslessСompression/results_sequence.txt", "a", encoding="utf-8") as file:
    file.write("Test Sequence 7:\n")
    file.write(f"Original Sequence: {original_sequence_7}\n")
    file.write(f"Sequence Alphabet Size: {len(elements_7)}\n")
    file.write(f"Original Sequence Size: {len(original_sequence_7)} [bits]\n\n")

# Генерація восьмої послідовності
# Заповнення рядка символами '1' до розміру N_sequence
original_sequence_8 = '1' * N_sequence

# Збереження результатів у файл results_sequence.txt
with open("D:/PycharmProjects/TIC/LosslessСompression/results_sequence.txt", "a", encoding="utf-8") as file:
    file.write("Test Sequence 8:\n")
    file.write(f"Original Sequence: {original_sequence_8}\n")
    file.write(f"Sequence Alphabet Size: 1\n")  # Тільки символ '1'
    file.write(f"Original Sequence Size: {len(original_sequence_8)} [bits]\n\n")

original_sequences = [
    original_sequence_1,
    original_sequence_2,
    original_sequence_3,
    original_sequence_4,
    original_sequence_5,
    original_sequence_6,
    original_sequence_7,
    original_sequence_8
]

# Список для зберігання характеристик кожної послідовності
results = []

# Цикл для обчислення характеристик кожної послідовності
for i, sequence in enumerate(original_sequences, start=1):
    # Знаходження унікальних символів та їх кількості в послідовності
    counts = collections.Counter(sequence)
    N_sequence = len(sequence)

    # Обчислення ймовірності появи кожного символу
    probability = {symbol: count / N_sequence for symbol, count in counts.items()}

    # Обчислення середньої ймовірності
    mean_probability = sum(probability.values()) / len(probability)

    # Визначення типу ймовірності появи символів
    equal = all(abs(prob - mean_probability) < 0.05 * mean_probability for prob in probability.values())
    uniformity = "рівна" if equal else "нерівна"

    # Обчислення ентропії
    entropy = -sum(p * math.log2(p) for p in probability.values())

    # Обчислення надмірності джерела
    sequence_alphabet_size = len(set(sequence))
    if sequence_alphabet_size > 1:
        source_excess = 1 - entropy / math.log2(sequence_alphabet_size)
    else:
        source_excess = 1

    # Збереження результатів в список
    results.append([
        sequence_alphabet_size,
        round(entropy, 2),
        round(source_excess, 2),
        uniformity
    ])

    # Збереження згенерованих послідовностей у файл sequence.txt
    with open("D:/PycharmProjects/TIC/LosslessСompression/sequence.txt", "a", encoding="utf-8") as file:
        file.write(f"Original Sequence {i}:\n")
        file.write(sequence + "\n\n")

# Виведення результатів у файл results_sequence.txt
with open("D:/PycharmProjects/TIC/LosslessСompression/results_sequence.txt", "a", encoding="utf-8") as file:
    file.write("Characteristics of Generated Sequences:\n")
    for i, sequence_result in enumerate(results, start=1):
        file.write(f"Sequence {i}:\n")
        file.write(f"Sequence Alphabet Size: {sequence_result[0]}\n")
        file.write(f"Entropy: {sequence_result[1]}\n")
        file.write(f"Source Excess: {sequence_result[2]}\n")
        file.write(f"Probability Uniformity: {sequence_result[3]}\n\n")

# Заголовки стовбців та рядків таблиці
headers = ['Розмір алфавіту', 'Ентропія', 'Надмірність', 'Ймовірність']
row_labels = [f"Послідовність {i}" for i in range(1, len(results) + 1)]

# Побудова таблиці
fig, ax = plt.subplots(figsize=(10, len(results) * 0.6))
ax.axis('off')  # Вимкнення осей графіку

# Створення таблиці
table = ax.table(cellText=results, colLabels=headers, rowLabels=row_labels, loc='center', cellLoc='center')

# Задання розміру шрифту та масштабу таблиці
table.set_fontsize(12)
table.scale(1.2, 1.2)

# Збереження рисунку у файл
fig.savefig("Характеристики_послідовностей.png")
