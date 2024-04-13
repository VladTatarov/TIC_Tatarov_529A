import os
from tornado import concurrent
from huffman import HuffmanTree
import math
import numpy as np
from scipy import fftpack
from PIL import Image

BLOCK_SIZE = 8


def dct_2d(image):
    return fftpack.dct(fftpack.dct(image.T, norm='ortho').T, norm='ortho')


def quantize(block, component):
    q = load_quantization_table(component)
    return (block / q).round().astype(np.int32)


def load_quantization_table(component):
    if component == 'lum':
        q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                      [12, 12, 14, 19, 26, 48, 60, 55],
                      [14, 13, 16, 24, 40, 57, 69, 56],
                      [14, 17, 22, 29, 51, 87, 80, 62],
                      [18, 22, 37, 56, 68, 109, 103, 77],
                      [24, 35, 55, 64, 81, 104, 113, 92],
                      [49, 64, 78, 87, 103, 121, 120, 101],
                      [72, 92, 95, 98, 112, 100, 103, 99]])
    elif component == 'chrom':
        q = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                      [18, 21, 26, 66, 99, 99, 99, 99],
                      [24, 26, 56, 99, 99, 99, 99, 99],
                      [47, 66, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99]])
    else:
        raise ValueError(f"Invalid component: {component}. Should be 'lum' or 'chrom'.")
    return q


def zigzag_points(rows, cols):
    # Define constants for directions
    UP, DOWN, RIGHT, LEFT, UP_RIGHT, DOWN_LEFT = range(6)

    # Move point in different directions
    def move(direction, current_points):
        return {
            UP: lambda p: (p[0] - 1, p[1]),
            DOWN: lambda p: (p[0] + 1, p[1]),
            LEFT: lambda p: (p[0], p[1] - 1),
            RIGHT: lambda p: (p[0], p[1] + 1),
            UP_RIGHT: lambda p: move(UP, move(RIGHT, p)),
            DOWN_LEFT: lambda p: move(DOWN, move(LEFT, p))
        }[direction](current_points)

    # Check if point is within block bounds
    def inbounds(current_points):
        return 0 <= current_points[0] < rows and 0 <= current_points[1] < cols

    # Initialize starting point at top-left cell
    current_point = (0, 0)
    move_up = True
    for _ in range(rows * cols):
        yield current_point
        if move_up:
            if inbounds(move(UP_RIGHT, current_point)):
                current_point = move(UP_RIGHT, current_point)
            else:
                move_up = False
                if inbounds(move(RIGHT, current_point)):
                    current_point = move(RIGHT, current_point)
                else:
                    current_point = move(DOWN, current_point)
        else:
            if inbounds(move(DOWN_LEFT, current_point)):
                current_point = move(DOWN_LEFT, current_point)
            else:
                move_up = True
                if inbounds(move(DOWN, current_point)):
                    current_point = move(DOWN, current_point)
                else:
                    current_point = move(RIGHT, current_point)


def block_to_zigzag(block):
    return np.array([block[point] for point in zigzag_points(*block.shape)])


def bits_required(n):
    n = abs(n)
    result = 0
    while n > 0:
        n >>= 1
        result += 1
    return result


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def run_length_encode(arr):
    last_nonzero = -1
    for i, elem in enumerate(arr):
        if elem != 0:
            last_nonzero = i
    symbols = []
    values = []
    run_length = 0
    for i, elem in enumerate(arr):
        if i > last_nonzero:
            symbols.append((0, 0))
            values.append(int_to_binstr(0))
            break
        elif elem == 0 and run_length < 15:
            run_length += 1
        else:
            size = bits_required(elem)
            symbols.append((run_length, size))
            values.append(int_to_binstr(elem))
            run_length = 0

    return symbols, values


def binstr_flip(binstr):
    # перевіряємо, чи є binstr двійковим рядком
    if not set(binstr).issubset('01'):
        raise ValueError("binstr повинен мати лише '0' та '1'")
    return ''.join(map(lambda c: '0' if c == '1' else '1', binstr))


def uint_to_binstr(number, size):
    return bin(number)[2:][-size:].zfill(size)


def int_to_binstr(n):
    if n == 0:
        return ''
    binstr = bin(abs(n))[2:]
    return binstr if n > 0 else binstr.translate(str.maketrans('01', '10'))


def write_to_file(filepath, dc, ac, blocks_count, tables):
    with open(filepath, 'w') as f:
        for table_name in ['dc_y', 'ac_y', 'dc_c', 'ac_c']:
            f.write(uint_to_binstr(len(tables[table_name]), 16))
            for key, value in tables[table_name].items():
                if table_name.startswith('dc'):
                    f.write(uint_to_binstr(key, 4))
                    f.write(uint_to_binstr(len(value), 4))
                else:
                    f.write(uint_to_binstr(key[0], 4))
                    f.write(uint_to_binstr(key[1], 4))
                    f.write(uint_to_binstr(len(value), 8))
                f.write(value)

        f.write(uint_to_binstr(blocks_count, 32))
        for b in range(blocks_count):
            for c in range(3):
                category = bits_required(dc[b, c])
                symbols, values = run_length_encode(ac[b, :, c])
                dc_table = tables['dc_y'] if c == 0 else tables['dc_c']
                ac_table = tables['ac_y'] if c == 0 else tables['ac_c']
                f.write(dc_table[category])
                f.write(int_to_binstr(dc[b, c]))
                for i in range(len(symbols)):
                    f.write(ac_table[tuple(symbols[i])])
                    f.write(values[i])


def encode(name_media):
    input_file = f"OurMedia/{name_media}.bmp"
    output_file = f"OurMedia/{name_media}_2.asf"
    image = Image.open(input_file)
    ycbcr = image.convert('YCbCr')
    npmat = np.array(ycbcr, dtype=np.uint8)
    rows, cols = npmat.shape[0], npmat.shape[1]

    if rows % BLOCK_SIZE == cols % BLOCK_SIZE == 0:
        blocks_count = rows // BLOCK_SIZE * cols // BLOCK_SIZE
    else:
        raise ValueError("Image width and height must be multiples of 8")

    dc = np.empty((blocks_count, 3), dtype=np.int32)
    ac = np.empty((blocks_count, 63, 3), dtype=np.int32)

    for i in range(0, rows, BLOCK_SIZE):
        for j in range(0, cols, BLOCK_SIZE):
            block_index = (i // BLOCK_SIZE) * (cols // BLOCK_SIZE) + j // BLOCK_SIZE
            for k in range(3):
                block = npmat[i:i + BLOCK_SIZE, j:j + BLOCK_SIZE, k] - 128
                dct_matrix = dct_2d(block)
                quant_matrix = quantize(dct_matrix, 'lum' if k == 0 else 'chrom')
                zigzag = block_to_zigzag(quant_matrix)
                dc[block_index, k] = zigzag[0]
                ac[block_index, :, k] = zigzag[1:]

    tables = generate_huffman_tables(dc, ac)
    write_to_file(output_file, dc, ac, blocks_count, tables)
    size_vyhsdnogo = os.path.getsize(input_file)
    with open("results_jpeg.txt", "a") as file:
        print('Size output file: {} byte'.format(size_vyhsdnogo), file=file)


def generate_huffman_tables(dc, ac):
    H_DC_Y = HuffmanTree(np.vectorize(bits_required)(dc[:, 0]))
    H_DC_C = HuffmanTree(np.vectorize(bits_required)(dc[:, 1:].flat))

    H_AC_Y = HuffmanTree(flatten(run_length_encode(ac[i, :, 0])[0] for i in range(ac.shape[0])))
    H_AC_C = HuffmanTree(flatten(run_length_encode(ac[i, :, j])[0] for i in range(ac.shape[0]) for j in [1, 2]))

    return {'dc_y': H_DC_Y.value_to_bitstring_table(),
            'ac_y': H_AC_Y.value_to_bitstring_table(),
            'dc_c': H_DC_C.value_to_bitstring_table(),
            'ac_c': H_AC_C.value_to_bitstring_table()}


class JPEGFileReader:
    TABLE_SIZE_BITS = 16
    BLOCKS_COUNT_BITS = 32
    DC_CODE_LENGTH_BITS = 4
    CATEGORY_BITS = 4
    AC_CODE_LENGTH_BITS = 8
    RUN_LENGTH_BITS = 4
    SIZE_BITS = 4

    def __init__(self, filepath):
        self.__file = open(filepath, 'r')

    def read_int(self, size):
        if size == 0:
            return 0

        # старший біт вказує на знак числа
        bin_num = self.__read_str(size)
        if bin_num[0] == '1':
            return self.__int2(bin_num)
        else:
            return self.__int2(binstr_flip(bin_num)) * -1

    def read_dc_table(self):
        table = dict()

        table_size = self.__read_uint(self.TABLE_SIZE_BITS)
        for _ in range(table_size):
            category = self.__read_uint(self.CATEGORY_BITS)
            code_length = self.__read_uint(self.DC_CODE_LENGTH_BITS)
            code = self.__read_str(code_length)
            table[code] = category
        return table

    def read_ac_table(self):
        table = dict()

        table_size = self.__read_uint(self.TABLE_SIZE_BITS)
        for _ in range(table_size):
            run_length = self.__read_uint(self.RUN_LENGTH_BITS)
            size = self.__read_uint(self.SIZE_BITS)
            code_length = self.__read_uint(self.AC_CODE_LENGTH_BITS)
            code = self.__read_str(code_length)
            table[code] = (run_length, size)
        return table

    def read_blocks_count(self):
        return self.__read_uint(self.BLOCKS_COUNT_BITS)

    def read_huffman_code(self, table):
        prefix = ''

        while prefix not in table:
            prefix += self.__read_char()
        return table[prefix]

    def __read_uint(self, size):
        if size <= 0:
            raise ValueError("розмір повинен бути більшим за 0")

        return self.__int2(self.__read_str(size))

    def __read_str(self, length):
        return self.__file.read(length)

    def __read_char(self):
        return self.__read_str(1)

    @staticmethod
    def __int2(bin_num):
        return int(bin_num, 2)


def read_image_file(filepath):
    reader = JPEGFileReader(filepath)
    tables = dict()
    for table_name in ['dc_y', 'ac_y', 'dc_c', 'ac_c']:
        if 'dc' in table_name:
            tables[table_name] = reader.read_dc_table()
        else:
            tables[table_name] = reader.read_ac_table()
    blocks_count = reader.read_blocks_count()

    dc = np.empty((blocks_count, 3), dtype=np.int32)
    ac = np.empty((blocks_count, 63, 3), dtype=np.int32)

    for block_index in range(blocks_count):
        for component in range(3):
            dc_table = tables['dc_y'] if component == 0 else tables['dc_c']
            ac_table = tables['ac_y'] if component == 0 else tables['ac_c']
            category = reader.read_huffman_code(dc_table)
            dc[block_index, component] = reader.read_int(category)
            cells_count = 0
            # TODO: спроба зробити читання AC коефіцієнтів краще
            while cells_count < 63:
                run_length, size = reader.read_huffman_code(ac_table)
                if (run_length, size) == (0, 0):
                    while cells_count < 63:
                        ac[block_index, cells_count, component] = 0
                        cells_count += 1
                else:
                    for i in range(run_length):
                        ac[block_index, cells_count, component] = 0
                        cells_count += 1
                    if size == 0:
                        ac[block_index, cells_count, component] = 0
                    else:
                        value = reader.read_int(size)
                        ac[block_index, cells_count, component] = value
                    cells_count += 1
    return dc, ac, tables, blocks_count


def zigzag_to_block(zigzag):
    rows = cols = int(math.sqrt(len(zigzag)))
    if rows * cols != len(zigzag):
        raise ValueError("Довжина зіг-зага повинна бути ідеальним квадратом ")
    block = np.empty((rows, cols), np.int32)
    for i, point in enumerate(zigzag_points(rows, cols)):
        block[point] = zigzag[i]
    return block


def dequantize(block, component):
    q = load_quantization_table(component)
    return block * q


def idct_2d(image):
    return fftpack.idct(fftpack.idct(image.T, norm='ortho').T, norm='ortho')


def decoder(name_media):
    text = "OurMedia/" + name_media + "_2.asf"

    dc, ac, tables, blocks_count = read_image_file(text)
    # блок є квадратом 8x8
    block_side = 8
    # висота і ширина зображення рівні
    image_side = int(math.sqrt(blocks_count)) * block_side
    blocks_per_line = image_side // block_side

    npmat = np.empty((image_side, image_side, 3), dtype=np.uint8)

    for block_index in range(blocks_count):
        i = block_index // blocks_per_line * block_side
        j = block_index % blocks_per_line * block_side
        for c in range(3):
            zigzag = [dc[block_index, c]] + list(ac[block_index, :, c])
            quant_matrix = zigzag_to_block(zigzag)
            dct_matrix = dequantize(quant_matrix, 'lum' if c == 0 else 'chrom')
            block = idct_2d(dct_matrix)
            npmat[i:i + 8, j:j + 8, c] = block + 128

    image = Image.fromarray(npmat, 'YCbCr')
    image = image.convert('RGB')
    filename = "NewPhotoResult" + name_media + "_2.jpg"
    image.save(filename)

    size_jpeg = os.path.getsize(filename)
    size_raw = os.path.getsize("OurMedia/" + name + ".bmp")
    width, height = image.size
    ratio = size_raw / size_jpeg
    with open("results_jpeg.txt", "a", encoding='utf-8') as file:
        print('Розмір файла JPEG: {} байт'.format(size_jpeg), '\n',
              f'Розмір зображення JPEG: {width}x{height}', file=file)
        print('Коефіцієнт стиснення= {:.2f}'.format(ratio), '\n', file=file)


if __name__ == "__main__":
    names = ["12_1", "12_2", "12_3"]
    for name in names:
        print(f"Начало обработки изображения: {name}")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Кодирование изображения
            print(f"Начало кодирования изображения: {name}")
            futures_encode = [executor.submit(encode, name)]
            concurrent.futures.wait(futures_encode)
            print(f"Завершено кодирование изображения: {name}")

            # Декодирование изображения
            print(f"Начало декодирования изображения: {name}")
            futures_decode = [executor.submit(decoder, name)]
            concurrent.futures.wait(futures_decode)
            print(f"Завершено декодирование изображения: {name}")

        print(f"Завершена обработка изображения: {name}")

    print("Обработка всех изображений завершена")
