import ast
import collections
import math
import matplotlib.pyplot as plt


def main():
    with open("sequence.txt", "r") as file:
        orig_seqs = ast.literal_eval(file.read())
        orig_seqs = [seq.strip("[]").strip("'") for seq in orig_seqs]

    with open("results_rle_lzw.txt", "w") as file:
        for seq in orig_seqs:
            entropy, enc_RLE, dec_RLE, comp_ratio_RLE = process_sequence(seq)
            encoded, size, dec_LZW, comp_ratio_LZW = encode_lzw(seq)

            file.write("/-/" * 20 + "\n")
            file.write(f"original_sequence: {seq}\nLen(sequence): {len(seq) * 16} [bits]\n")
            file.write(f"entropy: {entropy}\n\nRLE coding\n")
            file.write(f"coded RLE: {enc_RLE}\nlen(coded): {len(enc_RLE) * 16} [bits]\n")
            file.write(f"compression_ratio_RLE: {comp_ratio_RLE}\ndecoded_RLE: {dec_RLE}\n")
            file.write(f"len(decoded_RLE): {len(dec_RLE) * 16} [bits]\n\n")

            file.write(f"compression_ratio_LZW: {comp_ratio_LZW}\nDecoded LZW: {dec_LZW}\n")
            file.write(f"len(Decoded LZW): {len(dec_LZW) * 16} [bits]\n")

    generate_table(orig_seqs)


def process_sequence(seq):
    counts = collections.Counter(seq)
    probability = {symbol: count / 100 for symbol, count in counts.items()}
    entropy = -sum(p * math.log2(p) for p in probability.values())
    enc_RLE, encoded = encode_rle(seq)
    dec_RLE = decode_rle(encoded)
    comp_ratio_RLE = round(len(seq) / len(enc_RLE), 2) if len(enc_RLE) > 0 else '-'
    return round(entropy, 2), enc_RLE, dec_RLE, comp_ratio_RLE


def encode_rle(seq):
    result = []
    count = 1
    for i, item in enumerate(seq):
        if i == 0:
            continue
        elif item == seq[i - 1]:
            count += 1
        else:
            result.append((seq[i - 1], count))
            count = 1
    result.append((seq[-1], count))
    encoded = "".join(f"{item[1]}{item[0]}" for item in result)
    return encoded, result


def decode_rle(seq):
    return "".join(item[0] * item[1] for item in seq)


def encode_lzw(seq):
    size = 0
    dictionary = {chr(i): i for i in range(65536)}
    result = []
    current = ""
    for c in seq:
        new_str = current + c
        if new_str in dictionary:
            current = new_str
        else:
            result.append(dictionary[current])
            dictionary[new_str] = len(dictionary)
            element_bits = 16 if dictionary[current] < 65536 else math.ceil(math.log2(len(dictionary)))
            size += element_bits
            current = c
    last = 16 if dictionary[current] < 65536 else math.ceil(math.log2(len(dictionary)))
    size += last
    result.append(dictionary[current])
    return result, size, decode_lzw(result), round(len(seq) * 16 / size, 2)


def decode_lzw(seq):
    dictionary = {i: chr(i) for i in range(65536)}
    result = ""
    previous = None
    for code in seq:
        if code in dictionary:
            current = dictionary[code]
            result += current
            if previous is not None:
                dictionary[len(dictionary)] = previous + current[0]
            previous = current
        else:
            current = previous + previous[0]
            result += current
            dictionary[len(dictionary)] = current
            previous = current
    return result


def generate_table(orig_seqs):
    results = []
    headers = ['Entropy', 'CR RLE', 'CR LZW']
    for seq in orig_seqs:
        entropy, _, _, comp_ratio_RLE = process_sequence(seq)
        _, _, _, comp_ratio_LZW = encode_lzw(seq)
        results.append([entropy, comp_ratio_RLE, comp_ratio_LZW])
    fig, ax = plt.subplots(figsize=(14 / 1.54, 14 / 1.54))
    row_labels = [f'Sequence {i + 1}' for i in range(len(orig_seqs))]
    ax.axis('off')
    table = ax.table(cellText=results, colLabels=headers, rowLabels=row_labels,
                     loc='center', cellLoc='center')
    table.set_fontsize(14)
    table.scale(0.8, 2)
    plt.savefig("Compression Results RLE and LZW", dpi=600)


if __name__ == "__main__":
    main()
