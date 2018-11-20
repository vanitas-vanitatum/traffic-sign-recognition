import csv
from pathlib import Path
from typing import *

import cv2
import tqdm


def load_annotation(p: Path, img_h: int, img_w: int):
    with open(p.as_posix(), 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
            if img_h == img_w == 1024:
                y1, x1, y2, x2, y3, x3, y4, x4 = list(map(int, line[:8]))
            else:
                x1, y1, x2, y2, x3, y3, x4, y4 = list(map(int, line[:8]))
            yield [x1, y1, x2, y2, x3, y3, x4, y4]


def save_annotation(annotations: List, path: Path):
    annotation_strings = []
    for ann in annotations:
        annotation_strings.append(
            ",".join(str(x) for x in (ann + [""]))
        )
    with open(path.as_posix(), 'w') as f:
        f.write("\n".join(annotation_strings))


def fix_dataset(folder: str):
    imgs = list(Path(folder).rglob("*.jpg"))
    for img_file in tqdm.tqdm(imgs):
        img = cv2.imread(img_file.as_posix())
        img_h, img_w = img.shape[:2]
        if img_h != img_w:
            continue
        annotation_path = img_file.with_suffix(".txt")
        annotation = list(load_annotation(annotation_path, img_h, img_w))
        save_annotation(annotation, annotation_path)


if __name__ == '__main__':
    import argparse

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--folder', required=True)
    args = argument_parser.parse_args()

    folder = args.folder
    fix_dataset(folder)
