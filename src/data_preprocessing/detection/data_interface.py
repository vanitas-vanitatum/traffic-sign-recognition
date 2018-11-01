import skimage as ski
from skimage import io


class DataInterface:

    def __init__(self, target_resolution):
        self.target_resolution = target_resolution

    def __len__(self):
        raise NotImplementedError

    @staticmethod
    def save_sample(save_path, sample_id, image, bboxes):
        ski.io.imsave(f'{save_path}/{sample_id}.jpg', image)
        lines = []
        for bbox in bboxes:
            line = ''
            for point_x, point_y in bbox:
                line += f'{point_x},{point_y},'
            lines.append(line)
        with open(f'{save_path}/{sample_id}.txt') as f:
            f.writelines(lines)

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
            sample_id = len(dataset)

    def __iter__(self):
        for dataset in self.datasets:
            for sample in dataset:
                yield sample

class SingleDatasetInterface(DataInterface):

    def __init__(self, target_resolution, source_data_path, source_annotations_path):
        super().__init__(target_resolution)
        self.source_data_path = source_data_path
        self.source_annotations_path = source_annotations_path


class TsinghuaInterface(DataInterface):
    def __len__(self):
        pass

    def _process_sample(self, raw_sample, raw_annotation):
        size = raw_sample.shape[:2]
        ratio

    def process_dataset(self, destination_path, initial_id=0):
        pass
