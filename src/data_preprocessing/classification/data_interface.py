import json
import os
import tqdm
import cv2
import yaml
import pandas as pd
import shutil
import numpy as np

from src.data_preprocessing.base_data_interface import Sample
from src.data_preprocessing.classification.ruleset import RULESET


def is_image(file):
    return os.path.splitext(file)[1].lower() in ['.ppm', '.png', '.jpg']


class MultipleDatasetsInterface:

    def __init__(self, single_datasets):
        self.datasets = single_datasets

    def __len__(self):
        return sum(len(dset) for dset in self.datasets)

    def process_dataset(self, destination_path, initial_id=0):
        sample_id = 0
        for dataset in self.datasets:
            dataset.process_dataset(destination_path, initial_id=sample_id)
            sample_id += len(dataset)

    def __iter__(self):
        for dataset in self.datasets:
            for sample in dataset:
                yield sample


class ClassificationDatasetInterface:

    RULES = RULESET

    def __init__(self, source_data_path, considered_classes, classes_dict_filename):
        self.source_data_path = source_data_path

    def _process_sample(self, sample):
        res = []
        for rule in ClassificationDatasetInterface.RULES:
            inferred = rule(sample)
            if inferred:
                res.append(inferred)
        return res

    def process_dataset(self, destination_path, initial_id=0):
        for i, sample in tqdm.tqdm(enumerate(self, initial_id)):
            os.makedirs(f'{destination_path}/{sample.label}/', exist_ok=True)
            img = cv2.cvtColor(sample.image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(f'{destination_path}/{sample.label}/img_{i}.jpg', img)


class FolderDividedDataset(ClassificationDatasetInterface):

    def __init__(self, source_data_path, considered_classes, classes_dict_filename, dataset_name):
        super().__init__(source_data_path, considered_classes, classes_dict_filename)
        self.dataset_name = dataset_name
        with open(classes_dict_filename) as f:
            label_dct = yaml.load(f)

        classes_folders = os.listdir(source_data_path)
        def list_subfolder(folder):
            return os.listdir(f'{source_data_path}/{folder}/')

        def full_name(folder, f):
            return f'{source_data_path}/{folder}/{f}'

        self.sample_paths = {label_dct[int(fold)]: [full_name(fold, f) for f in list_subfolder(fold) if is_image(f)]
                             for fold in classes_folders if fold.isdigit() and label_dct[int(fold)] in considered_classes}

    def __len__(self):
        return sum(len(self.sample_paths[key]) for key in self.sample_paths)

    def __iter__(self):
        for label in self.sample_paths:
            for image_path in self.sample_paths[label]:
                raw_sample = Sample(cv2.imread(image_path), label, self.dataset_name)
                inferred = self._process_sample(raw_sample)
                for sample in inferred:
                    yield sample


class TsrdDataset(ClassificationDatasetInterface):

    def __init__(self, source_data_path, considered_classes, classes_dict_filename):
        super().__init__(source_data_path, considered_classes, classes_dict_filename)
        with open(classes_dict_filename) as f:
            label_dct = yaml.load(f)

        filenames_with_ids = [(f'{source_data_path}/{f}', label_dct[int(f.split('_')[0])])
                              for f in os.listdir(source_data_path)
                              if is_image(f)]
        self.sample_paths = [(f, lbl) for f, lbl in filenames_with_ids if lbl in considered_classes]

    def __len__(self):
        return len(self.sample_paths)

    def __iter__(self):
        for fname, label in self.sample_paths:
            raw_sample = Sample(cv2.imread(fname), label, 'TSRD')
            inferred = self._process_sample(raw_sample)
            for sample in inferred:
                yield sample


def balance_train_and_test(train_path, test_path):
    train_folders = set(os.listdir(train_path))
    test_folders = set(os.listdir(test_path))
    missing_folders = train_folders - test_folders
    for folder in train_folders:
        train_files = os.listdir(f'{train_path}/{folder}/')
        os.makedirs(f'{test_path}/{folder}', exist_ok=True)
        nb_test_files = len(os.listdir(f'{test_path}/{folder}'))
        nb_files_to_move = max(0, int(0.1 * len(train_files)) - nb_test_files)
        files_to_move = list(np.random.choice(train_files, nb_files_to_move, replace=False))
        for file in files_to_move:
            shutil.move(f'{train_path}/{folder}/{file}', f'{test_path}/{folder}/{file}')


if __name__ == '__main__':
    from src.data_preprocessing.classification.ruleset import CLASSES, SPEED_LIMIT_CLASSES

    DATA_PATH = '/home/mkosturek/pwr/masters/sem2/computer_vision/traffic-sign-recognition/data/'
    shutil.rmtree(DATA_PATH + 'classification/train')
    shutil.rmtree(DATA_PATH + 'classification/test')
    os.mkdir(DATA_PATH + 'classification/train')
    os.mkdir(DATA_PATH + 'classification/test')

    BELGIUM_PATH = '/data_unzipped/classification/Belgium/'
    BELGIUM_DICT = '/data_unzipped/classification/belgium_classes.yaml'

    GTSRB_PATH = '/data_unzipped/classification/GTSRB/'
    GTSRB_DICT = '/data_unzipped/classification/GTSRB_classes.yaml'

    TSRD_PATH = '/data_unzipped/classification/TSRD/'
    TSRD_DICT = '/data_unzipped/classification/TSRD_classes.yaml'

    print('TEST preparation')

    belg_test = FolderDividedDataset(DATA_PATH + BELGIUM_PATH + 'test/', CLASSES + SPEED_LIMIT_CLASSES, DATA_PATH + BELGIUM_DICT, 'Belgium')
    #gtsrb_test = FolderDividedDataset(DATA_PATH + GTSRB_PATH + 'test/', CLASSES, DATA_PATH + GTSRB_DICT, 'GTSRB')
    tsrd_test = TsrdDataset(DATA_PATH + TSRD_PATH + 'test/', CLASSES + SPEED_LIMIT_CLASSES, DATA_PATH + TSRD_DICT)

    test_dset = MultipleDatasetsInterface([belg_test, tsrd_test])
    test_dset.process_dataset(DATA_PATH + 'classification/test/')

    print('TRAIN preparation')

    belg_train = FolderDividedDataset(DATA_PATH + BELGIUM_PATH + 'train/', CLASSES + SPEED_LIMIT_CLASSES, DATA_PATH + BELGIUM_DICT, 'Belgium')
    gtsrb_train = FolderDividedDataset(DATA_PATH + GTSRB_PATH + 'train/', CLASSES + SPEED_LIMIT_CLASSES, DATA_PATH + GTSRB_DICT, 'GTSRB')
    tsrd_train = TsrdDataset(DATA_PATH + TSRD_PATH + 'train/', CLASSES + SPEED_LIMIT_CLASSES, DATA_PATH + TSRD_DICT)

    train_dset = MultipleDatasetsInterface([belg_train, gtsrb_train, tsrd_train])
    train_dset.process_dataset(DATA_PATH + 'classification/train/', initial_id=len(test_dset))

    print('TRAIN & TEST balancing')

    balance_train_and_test(DATA_PATH + 'classification/train/', DATA_PATH + 'classification/test/')

    train_count = sum([len(files) for r, d, files in os.walk(DATA_PATH + 'classification/train/')])
    test_count = sum([len(files) for r, d, files in os.walk(DATA_PATH + 'classification/test/')])

    print('TRAIN SAMPLES\t:', train_count)
    print('TEST_SAMPLES\t:', test_count)