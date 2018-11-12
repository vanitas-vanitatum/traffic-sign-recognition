from src.data_preprocessing.base_data_interface import Sample
import cv2


class Inference:

    def _origin_name(self, sample):
        if 'inferred' in sample.origin:
            return sample.origin
        else:
            return f'inferred_{sample.origin}'

    def __call__(self, sample: Sample) -> Sample:
        raise NotImplementedError

    def __add__(self, value):
        return InferenceChain(self, value)


class InferenceChain(Inference):

    def __init__(self, inference_1, inference_2):
        self.inference_1 = inference_1
        self.inference_2 = inference_2

    def __call__(self, sample: Sample) -> Sample:
        return self.inference_2(self.inference_1(sample))


class ChangeLabel(Inference):

    def __init__(self, new_label):
        self.new_label = new_label

    def __call__(self, sample: Sample) -> Sample:
        return Sample(sample.image, self.new_label, self._origin_name(sample))


class Flip(Inference):

    def __init__(self, how='horizontal'):
        self.flip_code = 1 if how == 'horizontal' else 0

    def __call__(self, sample: Sample) -> Sample:
        flipped = cv2.flip(sample.image, flipCode=self.flip_code)
        return Sample(flipped, sample.label, self._origin_name(sample))


class Rotate(Inference):

    def __init__(self, how='90'):
        if how == '90':
            self.rot_code = cv2.ROTATE_90_CLOCKWISE
        elif how == '180':
            self.rot_code = cv2.ROTATE_180
        else:
            self.rot_code = cv2.ROTATE_90_COUNTERCLOCKWISE

    def __call__(self, sample: Sample) -> Sample:
        rotated = cv2.rotate(sample.image, self.rot_code)
        return Sample(rotated, sample.label, self._origin_name(sample))


class Transpose(Inference):

    def __init__(self, how='\\'):
        self.how = how

    def __call__(self, sample: Sample) -> Sample:
        img = sample.image
        if self.how == '/':
            img = cv2.rotate(img, cv2.ROTATE_180)
        transposed = cv2.transpose(img)
        return Sample(transposed, sample.label, self._origin_name(sample))


class Identity(Inference):
    def __call__(self, sample: Sample) -> Sample:
        return sample


class Ignore(Inference):
    def __call__(self, sample: Sample) -> Sample:
        return None