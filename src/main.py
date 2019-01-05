import imutils
import tqdm
import yaml

from box_extractors import EXTRACTORS
from classifier import Classifier
from detector import Detector
from pipeline import Pipeline
from steps import *

with open(common.CONFIG_PATH) as f:
    config = yaml.load(f)

main_pipeline = Pipeline([
    Input("input"),
    DetectingSingleFrameStep(
        "detector",
        Detector(snapshot_path=config['paths']['detector']),
        EXTRACTORS[config["extractor"]]()
    ),
    ClassifyingBoxesStep(
        "classifier",
        model=Classifier(snapshot_path=config['paths']['classifier']),
        input_width=common.CLASSIFIER_INPUT_WIDTH,
        input_height=common.CLASSIFIER_INPUT_HEIGHT
    ),
    DecodeClassesStep(
        "decoder",
        label_encoder=common.unpickle_data(config['paths']['label_encoder'])
    ),
])

visualisation_pipeline = main_pipeline + [
    VisualiseStep(
        "visualise"
    ),
    ShowVisualisation(
        "showtime"
    )
]


def process_frame_by_frame(video_file: str):
    capture = cv2.VideoCapture(video_file)
    i = 0
    with tqdm.tqdm() as pbar:
        while True:
            ret, frame = capture.read()
            if ret:
                frame = imutils.resize(frame, height=common.FRAME_HEIGHT)
                frame = frame.astype(np.uint8)
                visualisation_pipeline.perform({
                    "input": frame
                }, False)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                i += 1
            else:
                break

            pbar.update(1)

    capture.release()
    cv2.destroyAllWindows()


def process_from_camera():
    capture = cv2.VideoCapture(0)
    i = 0
    with tqdm.tqdm() as pbar:
        while True:
            ret, frame = capture.read()
            if ret:
                frame = imutils.resize(frame, height=common.FRAME_HEIGHT)
                frame = frame.astype(np.uint8)
                visualisation_pipeline.perform({
                    "input": frame
                }, False)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                i += 1
            else:
                break

            pbar.update(1)

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse

    argument_parser = argparse.ArgumentParser(description="Script for validating program")
    argument_parser.add_argument("--video", help="Path to video file")
    args = argument_parser.parse_args()

    if args.video is not None:
        process_frame_by_frame(args.video)
    else:
        process_from_camera()
