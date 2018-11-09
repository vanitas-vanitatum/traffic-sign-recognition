import cv2
import numpy  as np
import tqdm
import matplotlib.pyplot as plt
from detector import Detector
import imutils


def validate(model_folder: str, video_file: str):
    detector = Detector(model_folder)

    capture = cv2.VideoCapture(video_file)
    with tqdm.tqdm() as pbar:
        while True:
            ret, frame = capture.read()
            if ret:
                frame = imutils.resize(frame, height=512)
                frame = frame[..., ::-1].astype(np.float32)
                boxes = detector.predict_single_frame(frame)
                print(boxes)
                frame = frame.astype(np.uint8)
                for box in boxes:
                    cv2.polylines(frame, [box.astype(np.int32).reshape((-1, 1, 2))], True,
                                  color=(255, 255, 0), thickness=5)

                plt.imshow(frame)
                plt.show()
            else:
                break

            pbar.update(1)

    capture.release()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--model_folder")
    parser.add_argument("-v", "--video_file")
    args = parser.parse_args()
    validate(args.model_folder, args.video_file)
