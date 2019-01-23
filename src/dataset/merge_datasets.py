import itertools
import shutil
from pathlib import Path
from typing import *

import tqdm


def get_current_max_index(paths: List[Path]) -> int:
    index = -1
    for path in paths:
        current_num = path.name.split("_")[-1].split(".")[0]
        index = max(int(current_num), index)
    return index


def modify_name(path: Path, index_offset: int) -> str:
    file_name = path.name
    components = file_name.split("_")
    base_name = components[0]
    orig_index, extension = components[1].split(".")
    new_index = str(int(orig_index) + index_offset)
    return "{}_{}.{}".format(base_name, new_index, extension)


def merge(input_folder: str, output_folder: str):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder_imgs_files = list(itertools.chain(
        output_folder.rglob("*.jpg"),
    ))
    input_imgs_txts_files = list(itertools.chain(
        input_folder.rglob("*.jpg"),
        input_folder.rglob("*.txt")
    ))
    current_max_num = get_current_max_index(output_folder_imgs_files)
    index = current_max_num + 1
    for input_file in tqdm.tqdm(input_imgs_txts_files):
        new_file_name = modify_name(input_file, index)
        shutil.copy2(input_file, output_folder / new_file_name)


if __name__ == '__main__':
    import argparse

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--input_folder", required=True)
    argument_parser.add_argument("--output_folder", required=True)

    args = argument_parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    merge(input_folder, output_folder)
