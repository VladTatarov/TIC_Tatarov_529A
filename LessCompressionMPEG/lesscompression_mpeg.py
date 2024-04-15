import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


def segment_image(frame):
    """Функція для розділення кадру на сегменти"""
    block_size = 16
    height, width, _ = frame.shape
    segments = []
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            segment = frame[y:y + block_size, x:x + block_size]
            segments.append(segment)
    return segments


def get_center(segment):
    """Функція для розрахунку центру блоку зображення"""
    return np.array([segment.shape[1] // 2, segment.shape[0] // 2])


def get_anchor_search_area(segment, search_range):
    """Функція для розрахунку зони кадру, в якій відбуватиметься пошук схожого блоку"""
    center = get_center(segment)
    x1 = max(0, center[0] - search_range)
    x2 = min(segment.shape[1], center[0] + search_range + 1)
    y1 = max(0, center[1] - search_range)
    y2 = min(segment.shape[0], center[1] + search_range + 1)
    return x1, y1, x2, y2


def get_block_zone(frame, segment, x, y):
    """Функція для вибору зони кадру для блоку"""
    block_size = segment.shape[0]
    return frame[y:y + block_size, x:x + block_size]


def get_mad(segment, candidate):
    """Функція для розрахунку параметру, який відображає ступінь схожості блоків"""
    return np.mean(np.abs(segment.astype(int) - candidate.astype(int)))


def get_best_match(frame, segment, search_range):
    """Функція для пошуку максимально схожого блоку зображення"""
    x1, y1, x2, y2 = get_anchor_search_area(segment, search_range)
    min_mad = float('inf')
    best_match = None
    for y in range(y1, y2):
        for x in range(x1, x2):
            candidate = get_block_zone(frame, segment, x, y)
            mad = get_mad(segment, candidate)
            if mad < min_mad:
                min_mad = mad
                best_match = candidate
    return best_match


def block_search_body(frame, segments, search_range):
    """Функція, яка об'єднує усі процедури пошуку блоків (кодування)"""
    predicted_frames = []
    for segment in segments:
        best_match = get_best_match(frame, segment, search_range)
        predicted_frames.append(best_match)
    return predicted_frames


def get_residual(frame, predicted_frames):
    """Функція для пошуку загальної різниці між кадрами"""
    residual = np.zeros_like(frame)
    for i, predicted_frame in enumerate(predicted_frames):
        segment = predicted_frame
        x = (i % (frame.shape[1] // segment.shape[1])) * segment.shape[1]
        y = (i // (frame.shape[1] // segment.shape[1])) * segment.shape[0]
        residual[y:y + segment.shape[0], x:x + segment.shape[1]] += frame[y:y + segment.shape[0],
                                                                    x:x + segment.shape[1]] - segment
    return residual


def get_reconstruct_target(frame, predicted_frames, residual):
    """Функція для відновлення кадру (декодування)"""
    reconstructed_frame = np.zeros_like(frame)
    for i, predicted_frame in enumerate(predicted_frames):
        segment = predicted_frame
        x = (i % (frame.shape[1] // segment.shape[1])) * segment.shape[1]
        y = (i // (frame.shape[1] // segment.shape[1])) * segment.shape[0]
        reconstructed_frame[y:y + segment.shape[0], x:x + segment.shape[1]] = segment + residual[y:y + segment.shape[0],
                                                                                        x:x + segment.shape[1]]
    return reconstructed_frame


def get_bits_per_pixel(frame, compressed_size):
    """Функція для розрахунку кількості біт на піксель"""
    return compressed_size / (frame.shape[0] * frame.shape[1])


def get_frames(video_path):
    """Функція для зчитування сусідніх кадрів з відео"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))  # Змінюємо розмір кадру до стандартного
        frames.append(frame)
    cap.release()
    return frames


def main(video_path):
    frames = get_frames(video_path)
    if not frames:
        print("Не вдалося завантажити відеофайл.")
        return

    search_range = 16

    results_dir = "Results"
    os.makedirs(results_dir, exist_ok=True)

    for i in range(1, min(6, len(frames))):
        current_frame = frames[i]
        previous_frame = frames[i - 1]

        cv2.imwrite(os.path.join(results_dir, f"Frame_{i - 1}.png"), previous_frame)
        cv2.imwrite(os.path.join(results_dir, f"Frame_{i}.png"), current_frame)

        segments = segment_image(previous_frame)
        predicted_frames = block_search_body(previous_frame, segments, search_range)
        residual = get_residual(current_frame, predicted_frames)
        reconstructed_frame = get_reconstruct_target(current_frame, predicted_frames, residual)

        if i == 1:
            cv2.imwrite(os.path.join(results_dir, f"Difference_{i}.png"), residual)
            cv2.imwrite(os.path.join(results_dir, f"Predicted_{i}.png"), np.concatenate(predicted_frames, axis=1))
            cv2.imwrite(os.path.join(results_dir, f"Reconstructed_{i}.png"), reconstructed_frame)

        original_bpp = get_bits_per_pixel(current_frame, len(current_frame.tobytes()))
        difference_bpp = get_bits_per_pixel(residual, len(residual.tobytes()))
        reconstructed_bpp = get_bits_per_pixel(reconstructed_frame, len(reconstructed_frame.tobytes()))

        print(f"Bits per pixel (original): {original_bpp:.2f}")
        print(f"Bits per pixel (difference): {difference_bpp:.2f}")
        print(f"Bits per pixel (reconstructed): {reconstructed_bpp:.2f}")

        original_bpp_r = get_bits_per_pixel(current_frame[:, :, 0], len(current_frame[:, :, 0].tobytes()))
        original_bpp_g = get_bits_per_pixel(current_frame[:, :, 1], len(current_frame[:, :, 1].tobytes()))

        difference_bpp_r = get_bits_per_pixel(residual[:, :, 0], len(residual[:, :, 0].tobytes()))
        difference_bpp_g = get_bits_per_pixel(residual[:, :, 1], len(residual[:, :, 1].tobytes()))

        reconstructed_bpp_r = get_bits_per_pixel(reconstructed_frame[:, :, 0],
                                                 len(reconstructed_frame[:, :, 0].tobytes()))
        reconstructed_bpp_g = get_bits_per_pixel(reconstructed_frame[:, :, 1],
                                                 len(reconstructed_frame[:, :, 1].tobytes()))

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar([0, 1, 2], [original_bpp, original_bpp_r, original_bpp_g], label="Оригінальний кадр")
        ax.bar([3, 4, 5], [difference_bpp, difference_bpp_r, difference_bpp_g], label="Різниця між кадрами")
        ax.bar([6, 7, 8], [reconstructed_bpp, reconstructed_bpp_r, reconstructed_bpp_g], label="Відновлений кадр")
        ax.set_xticks([1, 4, 7])
        ax.set_xticklabels(["Біт/Піксель RGB", "Біт/Піксель R", "Біт/Піксель G"])
        ax.legend()
        ax.set_title("Гістограма кількості біт на піксель для різних варіантів кодування")
        plt.savefig(os.path.join(results_dir, "Histogram.png"))

    print("Завершено.")


if __name__ == "__main__":
    video_path = "Video/sample4.avi"
    if not os.path.exists(video_path):
        print(f"Відеофайл {video_path} не знайдено.")
        exit()
    main(video_path)
