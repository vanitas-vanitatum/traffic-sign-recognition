import os

import cv2
import numpy as np
import pandas as pd
import tqdm


def mse(frame_1: np.ndarray, frame_2: np.ndarray) -> float:
    return np.mean(np.square(frame_1 - frame_2)).item()


def get_frame_luminosity_map(frame: np.ndarray) -> np.ndarray:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    return frame[..., 0]


def luminosity_difference(frame_1: np.ndarray, frame_2: np.ndarray) -> float:
    l_1 = get_frame_luminosity_map(frame_1)
    l_2 = get_frame_luminosity_map(frame_2)
    return mse(l_1, l_2)


def get_greyscale_histogram(frame: np.ndarray) -> np.ndarray:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist, bins = np.histogram(frame.ravel(), 256, [0, 256])
    hist = hist.astype(np.float32)
    hist /= np.trapz(hist)
    return hist


def kl_greyscale_divergence_histograms(frame_1: np.ndarray, frame_2: np.ndarray) -> float:
    frame_1_hist = get_greyscale_histogram(frame_1)
    frame_2_hist = get_greyscale_histogram(frame_2)
    kl_divergence = np.sum(frame_1_hist * np.log(frame_1_hist / (frame_2_hist + 1e-8) + 1e-8)).item()
    return kl_divergence


def analyse_video(video_path: str, output_path: str, frame_spacing: int):
    cap = cv2.VideoCapture(video_path)
    previous_frame: np.ndarray = None
    data = {
        "timestamp": [],
        "luminosity difference": [],
        "color difference": [],
        "kl divergence": []
    }
    fps: float = cap.get(cv2.CAP_PROP_FPS)
    total_frames: int = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    index: float = 0
    skipped_frames: int = 0
    with tqdm.tqdm(total=total_frames) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            index += 1
            pbar.update(1)
            if frame is None or np.prod(frame.shape) == 0:
                break
            if frame_spacing > 0:
                if (skipped_frames + 1) % (frame_spacing + 1) != 0 and previous_frame is not None:
                    skipped_frames += 1
                    continue
                else:
                    skipped_frames = 0
            if previous_frame is not None:
                timestamp = index / fps
                data["luminosity difference"].append(luminosity_difference(frame, previous_frame))
                data["color difference"].append(mse(frame, previous_frame))
                data["timestamp"].append(timestamp)
                data["kl divergence"].append(kl_greyscale_divergence_histograms(previous_frame, frame))
            previous_frame = frame

    frame = pd.DataFrame(data)
    output_path = os.path.join(output_path,
                               "{}_temporal_statistics_{}_skip.csv".format(os.path.basename(video_path).split('.')[0],
                                                                           frame_spacing))
    frame.to_csv(output_path, index=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Script for performing analysis of given video")
    parser.add_argument("--video_path", help="Path to video file")
    parser.add_argument("--output_path", help="Folder that will contain all files from analysis")
    parser.add_argument("--spacing", help="Spacing in frames, the more the more different consecutive frames will be",
                        type=int, default=0)

    args = parser.parse_args()

    analyse_video(args.video_path, args.output_path, args.spacing)
