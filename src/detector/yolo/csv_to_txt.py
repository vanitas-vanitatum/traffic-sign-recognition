import argparse
from itertools import groupby

import pandas as pd


def build_line_from_grouper(file_name, grouper) -> str:
    data = [file_name]
    for filename, x_min, y_min, x_max, y_max in grouper:
        data.append(
            "%d,%d,%d,%d,%d" % (x_min, y_min, x_max, y_max, 0)
        )
    return " ".join(data)


def convert(input_path: str, output_path: str):
    data = pd.read_csv(input_path)
    necessary_data = list(zip(data["filename"], data["xmin"], data["ymin"], data["xmax"], data["ymax"]))
    groups = groupby(necessary_data, lambda x: x[0])
    lines = []
    for key, a_group in groups:
        lines.append(build_line_from_grouper(key, a_group))

    output_data = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(output_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path")
    parser.add_argument("--output_path")
    args = parser.parse_args()

    convert(args.input_path, args.output_path)
