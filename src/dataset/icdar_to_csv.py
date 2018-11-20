import argparse
import csv
import itertools
import tqdm
from pathlib import Path
from typing import *

import cv2
import pandas as pd


def get_images(path: Path) -> List[Path]:
    files_iter = itertools.chain(
        path.rglob("*.jpg"),
        path.rglob("*.png"),
        path.rglob("*.jpeg"),
        path.rglob("*.JPG"),
        path.rglob("*.bmp"),
    )
    files_list = list(files_iter)
    return files_list


def load_annotation(p: Path):
    with open(p.as_posix(), 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]

            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(int, line[:8]))
            yield x1, y1, x2, y2, x3, y3, x4, y4


def parse_dataset(path: str):
    imgs = get_images(Path(path))
    data = []
    for img_fn in tqdm.tqdm(imgs):
        img = cv2.imread(img_fn.as_posix())
        annotations = load_annotation(img_fn.with_suffix(".txt"))
        for x1, y1, x2, y2, x3, y3, x4, y4 in annotations:
            line = [
                str(img_fn.as_posix()),
                img.shape[1],
                img.shape[0],
                "sign",
                min(x1, x2, x3, x4),
                min(y1, y2, y3, y4),
                max(x1, x2, x3, x4),
                max(y1, y2, y3, y4),
            ]
            data.append(line)
    path = Path(path)
    dataframe = pd.DataFrame(data=data,
                             columns=['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
    dataframe.to_csv(path.with_suffix(".csv"), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    args = parser.parse_args()
    parse_dataset(args.dataset)
