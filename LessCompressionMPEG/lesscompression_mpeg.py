import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np


def segment_image(anchor, block_size=16):
    h, w = anchor.shape
    h_segments = h // block_size
    w_segments = w // block_size
    return h_segments, w_segments


def get_center(x, y, block_size):
    return x + block_size // 2, y + block_size // 2


def get_anchor_search_area(x, y, anchor, block_size, search_area):
    h, w = anchor.shape
    cx, cy = get_center(x, y, block_size)
    sx = max(0, cx - block_size // 2 - search_area)
    sy = max(0, cy - block_size // 2 - search_area)
    anchor_search = anchor[
        sy:min(sy + search_area * 2 + block_size, h),
        sx:min(sx + search_area * 2 + block_size, w)
    ]
    return anchor_search


def get_block_zone(p, a_search, t_block, block_size):
    px, py = p
    px, py = px - block_size // 2, py - block_size // 2
    px, py = max(0, px), max(0, py)
    a_block = a_search[py:py + block_size, px:px + block_size]
    assert a_block.shape == t_block.shape
    return a_block


def get_mad(t_block, a_block):
    return np.sum(np.abs(np.subtract(t_block, a_block))) / (t_block.shape[0] * t_block.shape[1])


def get_best_match(t_block, a_search, block_size):
    step = 4
    ah, aw = a_search.shape
    acy, acx = ah // 2, aw // 2
    min_mad = float("inf")
    min_p = None
    while step >= 1:
        points = [(acx, acy), (acx + step, acy), (acx, acy + step), (acx + step, acy + step),
                  (acx - step, acy), (acx, acy - step), (acx - step, acy - step), (acx + step, acy - step),
                  (acx - step, acy + step)]
        for p in points:
            a_block = get_block_zone(p, a_search, t_block, block_size)
            mad = get_mad(t_block, a_block)
            if mad < min_mad:
                min_mad = mad
                min_p = p
        step //= 2
    px, py = min_p
    px, py = px - block_size // 2, py - block_size // 2
    px, py = max(0, px), max(0, py)
    match_block = a_search[py:py + block_size, px:px + block_size]
    return match_block


def block_search_body(anchor, target, block_size, search_area=7):
    h, w = anchor.shape
    h_segments, w_segments = segment_image(anchor, block_size)
    predicted = np.ones((h, w)) * 255
    for y in range(0, h_segments * block_size, block_size):
        for x in range(0, w_segments * block_size, block_size):
            target_block = target[y:y + block_size, x:x + block_size]
            anchor_search_area = get_anchor_search_area(x, y, anchor, block_size, search_area)
            anchor_block = get_best_match(target_block, anchor_search_area, block_size)
            predicted[y:y + block_size, x:x + block_size] = anchor_block
    return predicted


def get_residual(target, predicted):
    return np.subtract(target, predicted)


def get_reconstructed_target(residual, predicted):
    return np.add(residual, predicted)


def get_bits_per_pixel(im):
    h, w = im.shape
    epsilon = 1e-10
    bits = np.sum(np.log2(np.abs(im) + 1 + epsilon))
    return bits / (h * w)


def get_frames(filename, first_frame, second_frame):
    cap = cv2.VideoCapture(filename)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame - 1)
    _, fr1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, second_frame - 1)
    _, fr2 = cap.read()
    cap.release()
    return fr1, fr2


def calculate_bits_per_pixel(im):
    h, w = im.shape
    epsilon = 1e-10
    bits = np.sum(np.log2(np.abs(im) + 1 + epsilon))
    return bits / (h * w)


def save_images(output_dir, anchor_frame, target_frame, diff_frame_rgb, predicted_frame_rgb, residual_frame_rgb,
                restore_frame_rgb):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    cv2.imwrite(os.path.join(output_dir, "First_frame.png"), anchor_frame)
    cv2.imwrite(os.path.join(output_dir, "Second_frame.png"), target_frame)
    cv2.imwrite(os.path.join(output_dir, "Difference_between_frame.png"), diff_frame_rgb)
    cv2.imwrite(os.path.join(output_dir, "Prediction_frame.png"), predicted_frame_rgb)
    cv2.imwrite(os.path.join(output_dir, "Residual_frame.png"), residual_frame_rgb)
    cv2.imwrite(os.path.join(output_dir, "Restore_frame.png"), restore_frame_rgb)


def plot_histogram(output_dir, bits_anchor, bits_diff, bits_predicted):
    bar_width = 0.25
    plt.subplots(figsize=(12, 8))
    p1 = [sum(bits_anchor), *bits_anchor]
    diff = [sum(bits_diff), *bits_diff]
    mpeg = [sum(bits_predicted), *bits_predicted]
    br1 = np.arange(len(p1))
    br2 = [x + bar_width for x in br1]
    br3 = [x + bar_width for x in br2]
    plt.bar(br1, p1, color="r", width=bar_width, edgecolor="grey", label="Bits per pixel for anchor frame")
    plt.bar(br2, diff, color="g", width=bar_width, edgecolor="grey", label="Bits per pixel for frame difference")
    plt.bar(br3, mpeg, color="b", width=bar_width, edgecolor="grey",
            label="Bits per pixel for motion compensated difference")
    plt.title(f"Compression ratio = {round(sum(bits_anchor) / sum(bits_predicted), 2)}", fontweight="bold", fontsize=15)
    plt.ylabel("Bits per pixel", fontweight="bold", fontsize=15)
    plt.xticks([r + bar_width for r in range(len(p1))],
               ["Bits/Pixel RGB", "Bits/Pixel R", "Bits/Pixel G", "Bits/Pixel B"])
    plt.legend()
    plt.savefig(os.path.join(output_dir, "Histogram_bits_per_pixel.png"), dpi=600)


def process_frames(anchor_frame, target_frame, block_size):
    bits_anchor, bits_diff, bits_predicted = [], [], []
    h, w, ch = anchor_frame.shape
    diff_frame_rgb, predicted_frame_rgb, residual_frame_rgb, restore_frame_rgb = (
        np.zeros((h, w, ch)) for _ in range(4)
    )

    for i in range(ch):
        anchor_frame_c, target_frame_c = anchor_frame[:, :, i], target_frame[:, :, i]
        diff_frame = cv2.absdiff(anchor_frame_c, target_frame_c)
        predicted_frame = block_search_body(anchor_frame_c, target_frame_c, block_size)
        residual_frame = get_residual(target_frame_c, predicted_frame)
        reconstruct_target_frame = get_reconstructed_target(residual_frame, predicted_frame)

        bits_anchor.append(calculate_bits_per_pixel(anchor_frame_c))
        bits_diff.append(calculate_bits_per_pixel(diff_frame))
        bits_predicted.append(calculate_bits_per_pixel(residual_frame))

        diff_frame_rgb[:, :, i] = diff_frame
        predicted_frame_rgb[:, :, i] = predicted_frame
        residual_frame_rgb[:, :, i] = residual_frame
        restore_frame_rgb[:, :, i] = reconstruct_target_frame

    return (bits_anchor, bits_diff, bits_predicted, diff_frame_rgb, predicted_frame_rgb,
            residual_frame_rgb, restore_frame_rgb)


def main(anchor_frame, target_frame, save_output=False, output_dir="Results", block_size=16):
    (bits_anchor, bits_diff, bits_predicted, diff_frame_rgb, predicted_frame_rgb, residual_frame_rgb,
     restore_frame_rgb) = (
        process_frames(anchor_frame, target_frame, block_size)
    )

    if save_output:
        save_images(output_dir, anchor_frame, target_frame, diff_frame_rgb, predicted_frame_rgb, residual_frame_rgb,
                    restore_frame_rgb)

    plot_histogram(output_dir, bits_anchor, bits_diff, bits_predicted)


if __name__ == "__main__":
    fr = random.randint(0, 3000)
    frame1, frame2 = get_frames("Video/sample4.avi", fr, fr + 1)
    main(frame1, frame2, save_output=True)
