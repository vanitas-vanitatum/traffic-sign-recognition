import cv2
import imutils
import numpy as np
import tqdm

from detector import Detector


def validate(model_folder: str, video_file: str, counter: int):
    detector = Detector(model_folder)

    capture = cv2.VideoCapture(video_file)
    i = 0
    last_boxes = None
    with tqdm.tqdm() as pbar:
        while True:
            ret, frame = capture.read()
            if ret:
                frame = imutils.resize(frame, height=768)
                if i % counter == 0:
                    frame = frame.astype(np.float32)
                    boxes = detector.predict_single_frame(frame)

                    frame = frame.astype(np.uint8)
                    last_boxes = boxes
                if last_boxes is not None:
                    for box in last_boxes:
                        cv2.polylines(frame, [box.astype(np.int32).reshape((-1, 1, 2))], True,
                                      color=(255, 255, 0), thickness=5)

                cv2.imshow('frame', frame)
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

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--model_folder")
    parser.add_argument("-v", "--video_file")
    parser.add_argument("-c", "--counter", type=int, default=5)
    args = parser.parse_args()
    validate(args.model_folder, args.video_file, args.counter)
