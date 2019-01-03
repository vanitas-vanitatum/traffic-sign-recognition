import csv
import random
from pathlib import Path
from typing import *

import cv2
import imutils
import numpy as np
import tqdm

SEED = 0xCAFFE
IOU_THRESHOLD = 0.3
py_rng = random.Random(SEED)
np_rng = np.random.RandomState(SEED)
MAX_TRIES = 50
MINIMAL_WIDTH = 5
MINIMAL_HEIGHT = 5
WIDTH_RANGE = (MINIMAL_WIDTH, 40)
HEIGHT_RANGE = (MINIMAL_HEIGHT, 40)


def load_annotation(p: Path) -> np.ndarray:
    with open(p.as_posix(), 'r') as f:
        reader = csv.reader(f)
        all_boxes = []
        for line in reader:
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(int, line[:8]))
            left = min(x1, x2, x3, x4)
            right = max(x1, x2, x3, x4)
            top = min(y1, y2, y3, y4)
            bottom = max(y1, y2, y3, y4)
            all_boxes.append([left, top, right, bottom])
    return np.asarray(all_boxes, dtype=np.float32)


def pre_process_image_and_annotation(img: np.ndarray, annotation: np.ndarray,
                                     max_side_len: int) -> Tuple[np.ndarray, np.ndarray, float, float]:
    o_h, o_w = img.shape[:2]
    if o_h > o_w:
        img = imutils.resize(img, height=max_side_len)
    else:
        img = imutils.resize(img, width=max_side_len)

    n_h, n_w = img.shape[:2]
    h_ratio = n_h / o_h
    w_ratio = n_w / o_w

    annotation = annotation.astype(np.float32)
    annotation[:, 0] *= w_ratio
    annotation[:, 2] *= w_ratio
    annotation[:, 1] *= h_ratio
    annotation[:, 3] *= h_ratio
    return img, annotation.astype(np.int32), h_ratio, w_ratio


def create_probability_map(image: np.ndarray, window_size: int) -> np.ndarray:
    output_h = image.shape[0] - window_size
    output_w = image.shape[1] - window_size
    output_map = np.zeros((output_h, output_w), dtype=np.float32)
    for y in range(0, image.shape[0] - window_size, 1):
        for x in range(0, image.shape[1] - window_size, 1):
            sample = image[y:y + window_size, x:x + window_size]
            output_map[y, x] = np.var(sample)
    output_map = (output_map - output_map.min()) / (output_map.max() - output_map.min())
    output_map = np.pad(output_map, [[window_size // 2, window_size // 2], [window_size // 2, window_size // 2]],
                        mode="constant", constant_values=0)
    return output_map


def sample_box_parameters_from_image(image: np.ndarray, annotation: np.ndarray, coordinates: np.ndarray,
                                     probability_map: np.ndarray,
                                     num_samples: int) -> np.ndarray:
    result = []
    h, w = image.shape[:2]
    f_proba_map = np.reshape(probability_map, (-1,))
    f_proba_map = normalize_probabilities(f_proba_map)
    f_coordinates = np.reshape(coordinates, (-1, 2))
    indices = np.arange(0, len(f_proba_map))
    selected_indices = np_rng.choice(indices, size=(MAX_TRIES,), replace=False, p=f_proba_map)
    widths = np_rng.uniform(WIDTH_RANGE[0], WIDTH_RANGE[1], size=(MAX_TRIES,))
    heights = np_rng.uniform(HEIGHT_RANGE[0], HEIGHT_RANGE[1], size=(MAX_TRIES,))
    for i, index in enumerate(selected_indices):
        x, y = f_coordinates[index]
        width = widths[i]
        height = heights[i]
        x1 = x - width // 2
        x2 = x + width // 2
        y1 = y - height // 2
        y2 = y + height // 2

        x1 = max(0, x1)
        x2 = min(w - 1, x2)
        y1 = max(0, y1)
        y2 = min(h - 1, y2)

        box = np.asarray([x1, y1, x2, y2])
        if x2 - x1 >= MINIMAL_WIDTH and y2 - y1 >= MINIMAL_HEIGHT and validate_box_with_annotations(box, annotation):
            result.append(box)
        if len(result) >= num_samples:
            break
    return np.asarray(result, dtype=np.int32)


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return probabilities / probabilities.sum()


def validate_box_with_annotations(box: np.ndarray, annotations: np.ndarray) -> bool:
    for ann in annotations:
        iou = get_iou(box, ann)
        if iou > IOU_THRESHOLD:
            return False
    return True


def get_coordinates(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    xs, ys = np.meshgrid(np.arange(0, w), np.arange(0, h))
    coords = np.stack([xs, ys], axis=-1)
    return coords


def get_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x11, y11, x21, y21 = box1
    x12, y12, x22, y22 = box2
    intersection = (max(y11, y12) - min(y21, y22)) * (max(x11, x12) - min(x21, x22))
    if intersection < 0:
        return 0
    union = (y21 - y11) * (x21 - x11) + (y22 - y12) * (x22 - x12) - intersection
    return intersection / (union + 1e-8)


def extract_boxes(image: np.ndarray, box_parameters: np.ndarray, h_ratio: float, w_ratio: float) -> List[np.ndarray]:
    result = []
    for x1, y1, x2, y2 in box_parameters:
        frame = image[y1:y2, x1:x2]
        h_frame, w_frame = frame.shape[:2]
        h_frame /= h_ratio
        w_frame /= w_ratio
        frame = cv2.resize(frame, (int(w_frame), int(h_frame)))
        result.append(frame)
    return result


def generate_samples(
        input_folder: str,
        output_folder: str,
        window_size: int,
        default_size: int,
        num_samples: int,
        samples_per_photo: int):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    if output_folder.exists():
        raise ValueError("Given folder already exists")
    output_folder.mkdir()
    image_files = list(input_folder.rglob("*.jpg"))
    counter = 0
    probability_map_cache = {}
    coordinates_map_cache = {}
    with tqdm.tqdm(total=num_samples) as pbar:
        while counter < num_samples:
            image_file: Path = py_rng.choice(image_files)
            image = cv2.imread(image_file.as_posix())
            boxes = load_annotation(image_file.with_suffix(".txt"))
            image, boxes, h_ratio, w_ratio = pre_process_image_and_annotation(image, boxes, default_size)
            if image_file in probability_map_cache:
                probability_map = probability_map_cache[image_file]
            else:
                probability_map = create_probability_map(image, window_size)
                probability_map_cache[image_file] = probability_map
            img_h, img_w = image.shape[:2]
            if (img_h, img_w) in coordinates_map_cache:
                coordinates = coordinates_map_cache[(img_h, img_w)]
            else:
                coordinates = get_coordinates(image)
                coordinates_map_cache[(img_h, img_w)] = coordinates
            box_parameters = sample_box_parameters_from_image(image, boxes, coordinates, probability_map,
                                                              samples_per_photo)
            extracted_boxes = extract_boxes(image, box_parameters, h_ratio, w_ratio)
            for i, box in enumerate(extracted_boxes):
                import matplotlib.pyplot as plt
                plt.imshow(box)
                plt.show()
                cv2.imwrite((output_folder / "img_{}.jpg".format(i + counter)).as_posix(), box)

            counter += len(extracted_boxes)
            pbar.update(len(extracted_boxes))


if __name__ == '__main__':
    import argparse

    argument_parser = argparse.ArgumentParser(
        description="Script for generating data for classifier for discriminating windows with no class ")
    argument_parser.add_argument("--input_folder",
                                 help="Folder path for jpgs and annotations files prepared for detection")
    argument_parser.add_argument("--output_folder", help="Ex. 'no_class' for samples with no class")
    argument_parser.add_argument("--window_size", help="Window size for calculating variance", type=int)
    argument_parser.add_argument("--default_size",
                                 help="Standard resize for img so all objects that have "
                                      "to have same size will have the same size across images",
                                 type=int)
    argument_parser.add_argument("--num_samples", help="Number of samples to generate", type=int)
    argument_parser.add_argument("--samples_per_photo", help="Number of samples to extract from single photo", type=int)

    args = argument_parser.parse_args()
    generate_samples(
        args.input_folder,
        args.output_folder,
        args.window_size,
        args.default_size,
        args.num_samples,
        args.samples_per_photo
    )
