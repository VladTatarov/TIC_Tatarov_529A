import random
import ast


class HammingCode:
    def __init__(self, chunk_length=8):
        self.chunk_length = chunk_length
        self.check_bits = [i for i in range(1, chunk_length + 1) if not i & (i - 1)]
        self.check_bits_count = len(self.check_bits)
        self.total_length = self.chunk_length + self.check_bits_count
        self.redundancy = self.check_bits_count / self.chunk_length

    def encode(self, data):
        data_bin = ''.join(format(ord(char), '08b') for char in data)
        encoded = ''
        for i in range(0, len(data_bin), self.chunk_length):
            chunk = data_bin[i:i + self.chunk_length]
            chunk = self._set_check_bits(chunk)
            encoded += chunk
        return data_bin, encoded

    def decode(self, encoded, fix_errors=True):
        decoded_data = ''
        for i in range(0, len(encoded), self.total_length):
            chunk = encoded[i:i + self.total_length]
            if fix_errors:
                chunk = self._fix_errors(chunk)
            decoded_data += ''.join(self._safe_decode_chunk(chunk))
        return decoded_data

    def _set_check_bits(self, chunk):
        chunk = '0' * self.check_bits_count + chunk
        for bit in self.check_bits:
            value = sum(int(chunk[j]) for j in range(bit - 1, len(chunk), bit)) % 2
            chunk = chunk[:bit - 1] + str(value) + chunk[bit:]
        return chunk

    def _fix_errors(self, chunk):
        error_bits = []
        for bit in self.check_bits:
            value = sum(int(chunk[j]) for j in range(bit - 1, len(chunk), bit)) % 2
            if value != int(chunk[bit - 1]):
                error_bits.append(bit)
        if error_bits:
            error_bit = sum(error_bits)
            chunk = chunk[:error_bit - 1] + str(int(chunk[error_bit - 1]) ^ 1) + chunk[error_bit:]
        return chunk

    def _introduce_errors(self, encoded):
        result = ''
        for i in range(0, len(encoded), self.total_length):
            chunk = encoded[i:i + self.total_length]
            bit_index = random.randint(0, self.total_length - 1)
            chunk = chunk[:bit_index] + str(int(chunk[bit_index]) ^ 1) + chunk[bit_index + 1:]
            result += chunk
        return result

    def _safe_decode_chunk(self, chunk):
        for j in range(self.check_bits_count, len(chunk), 8):
            try:
                yield chr(int(chunk[j:j + 8], 2))
            except ValueError:
                yield '�'


def read_sequences(file_path):
    with open(file_path, "r") as file:
        sequences = ast.literal_eval(file.read())
        sequences = [seq.strip("[]").strip("'") for seq in sequences]
    return sequences


def process_sequence(hamming_code, sequence):
    source = sequence[:10]
    source_bin, encoded = hamming_code.encode(source)
    decoded = hamming_code.decode(encoded)
    encoded_with_error = hamming_code._introduce_errors(encoded)
    decoded_with_error = hamming_code.decode(encoded_with_error, fix_errors=False)
    decoded_without_error = hamming_code.decode(encoded_with_error)

    return {
        "source": source,
        "source_bin": source_bin,
        "encoded": encoded,
        "decoded": decoded,
        "encoded_with_error": encoded_with_error,
        "decoded_with_error": decoded_with_error,
        "decoded_without_error": decoded_without_error,
    }


def write_results(file_path, hamming_code, sequences):
    with open(file_path, "w", encoding="utf-8") as file:
        for sequence in sequences:
            result = process_sequence(hamming_code, sequence)
            write_sequence_result(file, hamming_code, result)


def write_sequence_result(file, hamming_code, result):
    file.write("///////////////////////////////////////////////////////////////////\n")
    file.write(f"оригінальна послідовність в байтовому форматі: {result['source']} byte\n")
    file.write(f"оригінальна послідовність в бітовому форматі: {result['source_bin']} bit\n")
    file.write(f"розмір оригінальної послідовності в бітах: {len(result['source_bin'])} bit\n")
    file.write(f"довжина блоку кодування: {hamming_code.chunk_length}\n")
    file.write(f"позиція контрольних біт: {hamming_code.check_bits}\n")
    file.write(f"відносна надмірність коду: {hamming_code.redundancy}\n")
    file.write("---------Кодування---------\n")
    file.write(f"закодовані дані: {result['encoded']}\n")
    file.write(f"розмір закодованих даних: {len(result['encoded'])}\n")
    file.write("---------Декодування---------\n")
    file.write(f"декодовані дані: {result['decoded']}\n")
    file.write(f"розмір декодованих даних у бітах: {len(result['decoded']) * 8} bit\n")
    file.write("---------Внесення помилки---------\n")
    file.write(f"послідовність з помилками: {result['encoded_with_error']}\n")
    diff_index_list = [i for i, (a, b) in enumerate(zip(result['encoded'], result['encoded_with_error']), start=1) if
                       a != b]
    file.write(f"кількість помилок: {len(diff_index_list)}\n")
    file.write(f"індекси помилок: {diff_index_list}\n")
    file.write(f"декодовані дані без виправлення помилки: {result['decoded_with_error']}\n")
    file.write("---------Виправлення помилки---------\n")
    file.write(f"декодовані дані з виправлення помилки: {result['decoded_without_error']}\n")
    file.write("///////////////////////////////////////////////////////////////////\n")


def main():
    input_file = "sequence.txt"
    output_file = "results_hamming.txt"
    original_sequences = read_sequences(input_file)
    hamming_code = HammingCode()
    write_results(output_file, hamming_code, original_sequences)


if __name__ == '__main__':
    main()
