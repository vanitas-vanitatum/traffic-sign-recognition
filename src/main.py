import tqdm
import yaml

from box_extractors import EXTRACTORS
from classifier import Classifier
from detector import Detector
from pipeline import *
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

without_showing = main_pipeline + [
    VisualiseStep(
        "visualise"
    )
]


class Recorder:
    def __init__(self, output_file: Optional[str]):
        self.output_path = output_file
        self.capture = cv2.VideoWriter_fourcc(*"XVID")
        self.init = False
        self.video_writer: cv2.VideoWriter = None

    def record_frame(self, frame: np.ndarray):
        if self.output_path is None:
            return
        h, w = frame.shape[:2]
        if not self.init:
            self.video_writer = cv2.VideoWriter(self.output_path, self.capture, 24, (w, h))
            self.init = True

        self.video_writer.write(frame)

    def __del__(self):
        if self.video_writer:
            self.video_writer.release()


def process_frame_by_frame_multithreaded(video_file: str, output_path: Optional[str]):
    processor = MultiThreadedProcessor(video_file, without_showing)
    recorder = Recorder(output_path)
    i = 0
    processor = processor.start()
    with tqdm.tqdm() as pbar:
        while not processor.stopped:
            frame = processor.read()
            if frame is None:
                continue
            cv2.imshow('frame', frame)
            # recorder.record_frame(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            i += 1
            pbar.update(1)
            # processor.update()
            # processor.predict()
    processor.stop()
    cv2.destroyAllWindows()


def process_frame_by_frame(video_file: str, output_path: Optional[str]):
    capture = cv2.VideoCapture(video_file)
    recorder = Recorder(output_path)
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
                recorder.record_frame(visualisation_pipeline.get_intermediate_output("visualised"))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                i += 1
            else:
                break

            pbar.update(1)

    capture.release()
    cv2.destroyAllWindows()


def process_from_camera(output_path: Optional[str]):
    capture = cv2.VideoCapture(0)
    i = 0
    recorder = Recorder(output_path)
    with tqdm.tqdm() as pbar:
        while True:
            ret, frame = capture.read()
            if ret:
                frame = imutils.resize(frame, height=common.FRAME_HEIGHT)
                frame = frame.astype(np.uint8)
                visualisation_pipeline.perform({
                    "input": frame
                }, False)
                recorder.record_frame(visualisation_pipeline.get_intermediate_output("visualised"))
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
    argument_parser.add_argument("--output_path", required=False, help="Path to output processed file")
    args = argument_parser.parse_args()

    if args.video is not None:
        process_frame_by_frame(args.video, args.output_path)
        # process_frame_by_frame_multithreaded(args.video, args.output_path)
    else:
        process_from_camera(args.output_path)
