from pathlib import Path

import cv2
import tqdm


def convert(path: str, separate: bool):
    path = Path(path)
    files = path.rglob("*.jpg")
    for a_file in tqdm.tqdm(list(files)):
        img = cv2.imread(a_file.as_posix())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        new_file = a_file
        if separate:
            new_file = a_file.with_suffix("_grayscale.jpg")
        cv2.imwrite(new_file.as_posix(), img)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    parser.add_argument("--separate", action="store_true", default=False,
                        help="Save as separate file with _grayscale suffix.")
    args = parser.parse_args()

    convert(args.path, args.separate)
