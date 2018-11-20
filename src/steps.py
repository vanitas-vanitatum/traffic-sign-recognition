from typing import Dict

import cv2
import numpy as np
from PIL import ImageDraw, ImageFont, Image
from sklearn.preprocessing import LabelEncoder

import common
from box_extractors import Extractor
from model import Model
from pipeline import Step


class Input(Step):
    def __init__(self, name: str):
        super().__init__(name)

    def perform(self, data: Dict):
        return {
            "input_frame": data[list(data.keys())[0]]
        }


class DetectingSingleFrameStep(Step):
    def __init__(self, name: str, model: Model, extractor: Extractor):
        super().__init__(name)
        self.model = model
        self.extractor = extractor
        self.required_keys = ["input_frame"]

    def perform(self, data: Dict) -> Dict:
        self.check_for_necessary_keys(data)
        input_frame = data["input_frame"]
        expanded_input_frame = np.expand_dims(input_frame, axis=0)
        detected_boxes = self.model.predict(expanded_input_frame)
        detected_boxes = np.squeeze(detected_boxes, axis=0)
        output_frames = self.extractor.extract(input_frame, detected_boxes)
        return {
            "boxes": output_frames,
            "boxes_coordinates": detected_boxes
        }


class ClassifyingBoxesStep(Step):
    def __init__(self, name: str, model: Model, input_width: int, input_height: int):
        super().__init__(name)
        self.model = model
        self.input_width = input_width
        self.input_height = input_height
        self.required_keys = ["boxes"]

    def perform(self, data: Dict) -> Dict:
        self.check_for_necessary_keys(data)
        input_data = data["boxes"]
        preprocessed_boxes = []
        classes = []
        for box in input_data:
            box = cv2.resize(box, (self.input_width, self.input_height))
            box = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
            box = np.expand_dims(box, axis=-1)
            preprocessed_boxes.append(box)
        if len(preprocessed_boxes) > 0:
            preprocessed_boxes = np.asarray(preprocessed_boxes, np.float32) / 255
            classes = self.model.predict(preprocessed_boxes)
        return {
            "predicted_classes": classes
        }


class DecodeClassesStep(Step):

    def __init__(self, name: str, label_encoder: LabelEncoder):
        super().__init__(name)
        self.le = label_encoder
        self.required_keys = ["predicted_classes"]

    def perform(self, data: Dict) -> Dict:
        self.check_for_necessary_keys(data)
        input_data = data["predicted_classes"]
        classes_names = self.le.inverse_transform(input_data)
        return {
            "classes_names": classes_names
        }


class VisualiseStep(Step):
    def __init__(self, name: str):
        super().__init__(name)
        self.required_keys = ["input_frame", "boxes", "predicted_classes"]

    def perform(self, data: Dict) -> Dict:
        self.check_for_necessary_keys(data)
        input_frame = data["input_frame"]
        boxes = data["boxes_coordinates"]
        predicted_classes = data["classes_names"]
        image = Image.fromarray(input_frame)
        if len(boxes) > 0:
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(font=common.FONT_PATH.as_posix(),
                                      size=np.floor(3e-2 * image.size[1] + 0.5).astype(np.int32))
            thickness = (image.size[0] + image.size[1]) // 500  # do day cua BB
            for (left, top, right, bottom), cls in zip(boxes, predicted_classes):

                label_size = draw.textsize(cls, font)
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                    # My kingdom for a good redistributable image drawing library.
                for j in range(thickness):
                    draw.rectangle([left + j, top + j, right - j, bottom - j], outline=(0, 255, 0))
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=(0, 255, 0))
                draw.text(text_origin, cls, fill=(0, 0, 0), font=font)

        return {
            "visualised": np.asarray(image)
        }


class ShowVisualisation(Step):
    def __init__(self, name: str):
        super().__init__(name)
        self.required_keys = ["visualised"]

    def perform(self, data: Dict) -> Dict:
        self.check_for_necessary_keys(data)
        visualised = data["visualised"]
        cv2.imshow('frame', visualised)
        return {}
