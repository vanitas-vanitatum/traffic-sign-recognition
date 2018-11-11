import os
import cv2
from collections import namedtuple


Sample = namedtuple('Sample', ['image', 'label', 'origin'])


class DataInterface:

    def __init__(self, target_resolution):
        self.target_resolution = target_resolution

    def __len__(self):
        raise NotImplementedError

    @staticmethod
    def save_detection_sample(destination_path, sample_id, image, annotation, save_image=True, save_annot=True):
        if save_image:
            cv2.imwrite(f'{destination_path}/img_{sample_id}.jpg', image)
        if save_annot:
            with open(f'{destination_path}/img_{sample_id}.txt', 'w') as f:
                f.write(annotation)

    @staticmethod
    def save_classification_sample(destination_path, sample_id, image, class_id):
        os.makedirs(f'{destination_path}/{class_id}', exist_ok=True)
        cv2.imwrite(f'{destination_path}/{class_id}/img_{sample_id}.jpg', image)

    def __iter__(self):
        raise NotImplementedError

    def _process_sample(self, raw_sample, raw_annotation):
        raise NotImplementedError

    def process_dataset(self, destination_path, initial_id=0):
        raise NotImplementedError


class MultipleDatasetsInterface(DataInterface):

    def __init__(self, target_resolution, single_datasets):
        super().__init__(target_resolution)
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


class SingleDatasetInterface(DataInterface):

    def __init__(self, target_resolution, source_data_path, source_annotations_path):
        super().__init__(target_resolution)
        self.source_data_path = source_data_path
        self.source_annotations_path = source_annotations_path