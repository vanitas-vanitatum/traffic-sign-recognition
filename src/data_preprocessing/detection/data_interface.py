import json
import os
import tqdm
import cv2
import pandas as pd
import shutil

from src.data_preprocessing.base_data_interface import MultipleDatasetsInterface, DetectionDataInterface


class TsinghuaInterface(DetectionDataInterface):

    def __init__(self, target_resolution, source_data_path, source_annotations_path, save_image=True, save_annot=True):
        super().__init__(target_resolution, source_data_path, source_annotations_path)
        self.save_image = save_image
        self.save_annot = save_annot

        with open(f'{self.source_annotations_path}/annotations.json') as f:
            self.annotations = json.load(f)
            self.annotations = self.annotations['imgs']

        self.sample_ids = {os.path.splitext(item)[0]: item for item in os.listdir(self.source_data_path)}

    def __len__(self):
        return len(self.sample_ids)

    def __iter__(self):
        for sample_id in self.sample_ids:
            # sample = ski.io.imread(f'{self.source_data_path}/{self.sample_ids[sample_id]}')
            sample = cv2.imread(f'{self.source_data_path}/{self.sample_ids[sample_id]}')
            annot = self.annotations[sample_id]['objects']
            sample, annot = self._process_sample(sample, annot)
            yield sample, annot

    def _process_sample(self, raw_sample, raw_annotation):
        size = raw_sample.shape[:2]
        scale_ratio = self.target_resolution / max(size)

        im = cv2.resize(raw_sample, (0, 0), fx=scale_ratio, fy=scale_ratio) if self.save_image else None

        annotations = []
        for annot in raw_annotation:
            b = annot['bbox']
            points = [b['xmin'], b['ymin'], b['xmax'], b['ymin'], b['xmax'], b['ymax'], b['xmin'], b['ymax']]
            points_scaled = [int(coord * scale_ratio) for coord in points]
            line = ','.join(str(p) for p in points_scaled) + ','
            annotations.append(line)
        return im, '\n'.join(annotations)

    def process_dataset(self, destination_path, initial_id=0):
        for current_id, (sample, annotation) in tqdm.tqdm(enumerate(self, initial_id), desc='tsinghua',
                                                          total=len(self)):
            self.save_detection_sample(destination_path, current_id, sample, annotation,
                                       self.save_image, self.save_annot)


class TsignDetInterface(DetectionDataInterface):

    def __init__(self, target_resolution, source_data_path, source_annotations_path, save_image=True, save_annot=True):
        super().__init__(target_resolution, source_data_path, source_annotations_path)
        self.save_image = save_image
        self.save_annot = save_annot

        self.annotations = dict()
        for annot_file in os.listdir(self.source_annotations_path):
            with open(f'{self.source_annotations_path}/{annot_file}') as f:
                self.annotations[os.path.splitext(annot_file)[0]] = f.read()

        self.sample_ids = {os.path.splitext(item)[0]: item for item in os.listdir(self.source_data_path)}

    def __len__(self):
        return len(self.sample_ids)

    def __iter__(self):
        for sample_id in self.sample_ids:
            # sample = ski.io.imread(f'{self.source_data_path}/{self.sample_ids[sample_id]}')
            sample = cv2.imread(f'{self.source_data_path}/{self.sample_ids[sample_id]}')
            annot = self.annotations[sample_id]
            sample, annot = self._process_sample(sample, annot)
            yield sample, annot

    def _process_sample(self, raw_sample, raw_annotation):
        size = raw_sample.shape[:2]
        scale_ratio = self.target_resolution / max(size)

        im = cv2.resize(raw_sample, (0, 0), fx=scale_ratio, fy=scale_ratio) if self.save_image else None

        annotations = []
        for annot in raw_annotation.split('\n'):
            annot = annot.strip()
            if annot and annot.count(',') == 7:
                annot = ','.join(str(int(int(x) * scale_ratio)) for x in annot.split(','))
                annotations.append(annot + ',')
        return im, '\n'.join(annotations)

    def process_dataset(self, destination_path, initial_id=0):
        for current_id, (sample, annotation) in tqdm.tqdm(enumerate(self, initial_id), desc='TsignDet',
                                                          total=len(self)):
            self.save_detection_sample(destination_path, current_id, sample, annotation,
                                       self.save_image, self.save_annot)


class AmericanInterface(DetectionDataInterface):

    def __init__(self, target_resolution, source_data_path, source_annotations_path, save_image=True, save_annot=True):
        super().__init__(target_resolution, source_data_path, source_annotations_path)
        self.save_image = save_image
        self.save_annot = save_annot

        self.annotations = pd.read_csv(self.source_annotations_path, sep=';')

    def __len__(self):
        return len(self.annotations['Filename'].unique())

    def __iter__(self):
        for image_path in self.annotations['Filename'].unique():
            sample = cv2.imread(f'{self.source_data_path}/{image_path}')
            annots = self.annotations[self.annotations['Filename'] == image_path]
            annot = []
            for i in range(len(annots)):
                x1 = annots.iloc[i]['Upper left corner X']
                y1 = annots.iloc[i]['Upper left corner Y']
                x2 = annots.iloc[i]['Lower right corner X']
                y2 = annots.iloc[i]['Lower right corner Y']
                annot.append([x1, y1, x2, y1, x2, y2, x1, y2])
            yield self._process_sample(sample, annot)

    def _process_sample(self, raw_sample, raw_annotation):
        size = raw_sample.shape[:2]
        scale_ratio = self.target_resolution / max(size)

        im = cv2.resize(raw_sample, (0, 0), fx=scale_ratio, fy=scale_ratio) if self.save_image else None

        annotations = []
        for annot in raw_annotation:
            annotations.append(','.join([str(int(a * scale_ratio)) for a in annot])+',')
        return im, '\n'.join(annotations)

    def process_dataset(self, destination_path, initial_id=0):
        for current_id, (sample, annotation) in tqdm.tqdm(enumerate(self, initial_id), desc='American',
                                                          total=len(self)):

            self.save_detection_sample(destination_path, current_id, sample, annotation,
                                       self.save_image, self.save_annot)


if __name__ == '__main__':
    res = 1024
    DATA_PATH = '/home/mkosturek/pwr/masters/sem2/computer_vision/traffic-sign-recognition/data/'
    TSIGNDET_PATH = 'data_unzipped/detection/TsignDet/'
    TSINGHUA_PATH = 'data_unzipped/detection/tsinghua/'
    AMERICAN_PATH = 'data_unzipped/detection/american/'

    tsinghua_test = TsinghuaInterface(res, DATA_PATH + TSINGHUA_PATH + 'test/', DATA_PATH + TSINGHUA_PATH)
    tsigndet_test = TsignDetInterface(res, DATA_PATH + TSIGNDET_PATH + 'test/',
                                      DATA_PATH + TSIGNDET_PATH + 'test_annotation/')
    american_test = AmericanInterface(res, DATA_PATH + AMERICAN_PATH,
                                      DATA_PATH + AMERICAN_PATH + 'testAnnotations.csv')

    test_dset = MultipleDatasetsInterface([tsigndet_test,
                                           tsinghua_test,
                                           american_test])
    shutil.rmtree(DATA_PATH + 'detection/test')
    os.makedirs(DATA_PATH + 'detection/test', exist_ok=True)
    test_dset.process_dataset(DATA_PATH + 'detection/test/')

    tsinghua_train = TsinghuaInterface(res, DATA_PATH + TSINGHUA_PATH + 'train/', DATA_PATH + TSINGHUA_PATH)
    tsigndet_train = TsignDetInterface(res, DATA_PATH + TSIGNDET_PATH + 'train/',
                                       DATA_PATH + TSIGNDET_PATH + 'train_annotation/')
    american_train = AmericanInterface(res, DATA_PATH + AMERICAN_PATH,
                                       DATA_PATH + AMERICAN_PATH + 'trainAnnotations.csv')

    train_dset = MultipleDatasetsInterface([tsigndet_train, tsinghua_train, american_train])
    shutil.rmtree(DATA_PATH + 'detection/train')
    os.makedirs(DATA_PATH + 'detection/train', exist_ok=True)
    train_dset.process_dataset(DATA_PATH + 'detection/train/')
