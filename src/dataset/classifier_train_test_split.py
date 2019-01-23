import shutil
from itertools import groupby
from pathlib import Path

import numpy as np

np.random.seed(0)


def split_dataset(input_dir: str, output_dir: str, fraction: float):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = [path.name for path in input_dir.iterdir()]
    for cls in classes:
        (output_dir / cls).mkdir()

    files_in_classes = input_dir.rglob("*.jpg")
    groupings_in_classes = groupby(files_in_classes, key=lambda x: x.parent.name)
    for a_class, group in groupings_in_classes:
        group = list(group)
        nb_to_move = int(len(group) * fraction)
        nb_to_move = max(1, nb_to_move)
        files_to_move = np.random.choice(group, size=nb_to_move, replace=False)
        for f in files_to_move:
            output_path = output_dir / a_class / f.name
            shutil.move(f.as_posix(), output_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", help="Dir with samples to take from, where each class is a separate folder")
    parser.add_argument("--output_dir", help="Bare output dir, where new samples will be put")
    parser.add_argument("--fraction", help="Fraction of samples from each class to take", default=0.1, type=float)
    args = parser.parse_args()

    split_dataset(args.input_dir, args.output_dir, args.fraction)
