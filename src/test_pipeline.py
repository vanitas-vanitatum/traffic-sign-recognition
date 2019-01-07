import datetime
import itertools
import logging
import pickle as pkl
import random
from enum import Enum
from itertools import groupby
from pathlib import Path
from typing import *

import imgaug as ia
import imgaug.augmenters as iaa
import imutils
import matplotlib.pyplot as plt
import tqdm
import yaml
from sklearn import metrics

from main import main_pipeline
from steps import *

random.seed(0)


def get_logger() -> logging.Logger:
    logger = logging.getLogger("testing pipeline")
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


logger = get_logger()

with open(common.CONFIG_PATH) as f:
    config = yaml.load(f)

AVAILABLE_DETECTION_TESTING_DATASETS = ["american", "chinese"]

# script is run from source directory
OUTPUT_TESTING_REPORT_DIRECTORY = Path("../report")
OUTPUT_TESTING_REPORT_DIRECTORY.mkdir(exist_ok=True)

# number of seconds that passed after we consider that view on the camera completely changed so we can evaluate
# it seperately as independent frame.mkdir(exist_ok=True)
TIME_INTERPOLATION_FACTOR = 7

DETECTION_AUGMENTATION = iaa.Sequential([
    iaa.Scale((1.0, 1.2)),
    iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, mode="constant", cval=0)
], deterministic=True)

CLASSIFICATION_AUGMENTATION = iaa.Sequential([
    iaa.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, mode="constant", cval=0)
], deterministic=True)

MAX_AUGMENTATION_TRIES = 50
NUM_OF_AUGMENTED_SAMPLES = 5
IOU_POSITIVE_THRESHOLD = 0.5


class DetectorMetric(Enum):
    IoU = "iou"
    Recall = "recall"
    Precision = "precision"
    OptimisticIoU = "optimistic iou"
    OptimisticRecall = "optimistic recall"
    OptimisticPrecision = "optimistic precision"
    MeanProcessingTime = "mean processing time"
    StdProcessingTime = "std processing time"


class ClassifierMetric(Enum):
    OptimisticAccuracy = "optimistic accuracy"
    OptimisticRecall = "optimistic recall"
    OptimisticPrecision = "optimistic precision"
    OptimisticAUC = "optimistic auc"
    OptimisticConfusionMatrix = "optimistic confusion matrix"
    Accuracy = "accuracy"
    Recall = "recall"
    Precision = "precision"
    AUC = "auc"
    ConfusionMatrix = "confusion matrix"
    MeanProcessingTime = "mean processing time"
    StdProcessingTime = "std processing time"


class PipelineMetric(Enum):
    Accuracy = "accuracy"
    FalsePositives = "false positives"
    TruePositives = "true positives"
    FalsePositivesInTime = "false positives in time"
    TruePositivesInTime = "true positives in time"


def augment_bounding_boxes(boxes: np.ndarray, image_shape: Tuple[int, ...]) -> np.ndarray:
    bbs = ia.BoundingBoxesOnImage([
        ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
        for x1, y1, x2, y2 in boxes], shape=image_shape)
    bbs = DETECTION_AUGMENTATION.augment_bounding_boxes([bbs])[0]
    bbs = bbs.remove_out_of_image().cut_out_of_image()
    return bbs.to_xyxy_array()


def create_mask_with_boxes(boxes: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes.astype(np.int):
        mask[y1:y2, x1:x2] = 1
    return mask


def get_single_box_iou(box_1: np.ndarray, box_2: np.ndarray) -> float:
    box_1 = ia.BoundingBox(x1=box_1[0], y1=box_1[1], x2=box_1[2], y2=box_1[3])
    box_2 = ia.BoundingBox(x1=box_2[0], y1=box_2[1], x2=box_2[2], y2=box_2[3])
    iou = box_1.iou(box_2)
    return iou


def get_iou(boxes_1: np.ndarray, boxes_2: np.ndarray, image_shape: Tuple[int, ...]) -> float:
    first_mask = create_mask_with_boxes(boxes_1, image_shape[0], image_shape[1])
    second_mask = create_mask_with_boxes(boxes_2, image_shape[0], image_shape[1])
    intersection = (first_mask & second_mask).sum()
    union = (first_mask | second_mask).sum()
    return intersection / (union + 1e-8)


def _get_closest_box(box: np.ndarray, other_boxes: np.ndarray) -> np.ndarray:
    x_c = (box[0] + box[2]) / 2
    y_c = (box[1] + box[3]) / 2
    center = np.asarray([x_c, y_c])
    best_box = None
    best_dist = np.inf
    for other_box in other_boxes:
        other_center = np.asarray([(other_box[0] + other_box[2]) / 2, (other_box[1] + other_box[3]) / 2])
        dist = np.linalg.norm(center - other_center, ord=2)
        if dist < best_dist:
            best_box = other_box
            best_dist = dist
    return best_box


def get_boxes_correct_guesses(boxes_1: np.ndarray, boxes_2: np.ndarray) -> float:
    guessed = 0
    for box_1 in boxes_1:
        closest_box = _get_closest_box(box_1, boxes_2)
        iou = get_single_box_iou(box_1, closest_box)
        if iou > IOU_POSITIVE_THRESHOLD:
            guessed += 1
    return guessed


def augment_detection_samples(samples: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
    global DETECTION_AUGMENTATION

    to_augment = NUM_OF_AUGMENTED_SAMPLES - len(samples)
    new_samples = random.choices(samples, k=to_augment)
    augmented = []
    for image, boxes in new_samples:
        is_ok = False
        try_index = 0
        new_image, new_boxes = image, boxes
        DETECTION_AUGMENTATION = DETECTION_AUGMENTATION.to_deterministic()
        while not is_ok and try_index < MAX_AUGMENTATION_TRIES:
            new_image = DETECTION_AUGMENTATION.augment_image(image)
            new_boxes = augment_bounding_boxes(boxes, image.shape)
            is_ok = len(new_boxes) == len(boxes)
            try_index += 1
        if not is_ok:
            augmented.append((image, boxes))
            logger.warning("Failed at augmenting data")
        else:
            augmented.append((new_image, new_boxes))
    return augmented


def read_detection_images_with_box_annotation(paths: Iterable[Path]) -> List[Tuple[np.ndarray, np.ndarray]]:
    result = []
    for path in paths:
        image = cv2.imread(path.as_posix())
        original_height, original_width = image.shape[:2]
        image = imutils.resize(image, height=common.FRAME_HEIGHT)
        new_height, new_width = image.shape[:2]

        w_ratio = new_width / original_width
        h_ratio = new_height / original_height

        boxes = []
        with open(path.with_suffix(".txt").as_posix()) as f:
            annotation = f.read().strip().split('\n')
            for line in annotation:
                x1, y1, x2, y2, x3, y3, x4, y4 = map(float, line.strip().split(',')[:8])
                x_min = min(x1, x2, x3, x4)
                x_max = max(x1, x2, x3, x4)
                y_min = min(y1, y2, y3, y4)
                y_max = max(y1, y2, y3, y4)
                box = [
                    x_min * w_ratio,
                    y_min * h_ratio,
                    x_max * w_ratio,
                    y_max * h_ratio
                ]
                boxes.append(box)
        result.append((image, np.asarray(boxes, dtype=np.float32)))
    return result


def evaluate_detector_for_single_dataset(directory: Path) -> Dict[str, float]:
    images_in_dir = list(directory.rglob("*.jpg"))
    by_sequence = groupby(images_in_dir, key=lambda x: x.parent.name)

    times = []
    ious, guesses = [], []
    optimistic_ious, optimistic_guesses = [], []
    total_gt_boxes = 0
    total_predicted_boxes = 0

    optimistic_total_gt_boxes = 0
    optimistic_total_predicted_boxes = 0

    for key, group in by_sequence:
        logger.info("Evaluating detector ... %s subfolder ..." % key)
        group = list(group)
        images_boxes = read_detection_images_with_box_annotation(group)
        if len(images_boxes) < NUM_OF_AUGMENTED_SAMPLES:
            images_boxes.extend(augment_detection_samples(images_boxes))

        best_iou = -1.0
        best_boxes_gt, best_boxes_predicted = None, None

        mean_guesses = []
        mean_ious = []

        for image, boxes in images_boxes:
            main_pipeline.perform({
                "input": image
            }, True)

            predicted_boxes = main_pipeline.get_intermediate_output("boxes_coordinates")
            current_iou = get_iou(boxes, predicted_boxes, image.shape)
            mean_ious.append(current_iou)
            mean_guesses.append(get_boxes_correct_guesses(predicted_boxes, boxes))

            if current_iou > best_iou:
                best_iou = current_iou
                best_boxes_gt = boxes
                best_boxes_predicted = predicted_boxes
            times.append(main_pipeline.timers["detector"])
            total_predicted_boxes += (len(predicted_boxes) / len(images_boxes))
            total_gt_boxes += (len(boxes) / len(images_boxes))

        optimistic_total_predicted_boxes += len(best_boxes_predicted)
        optimistic_total_gt_boxes += len(best_boxes_gt)

        ious.append(np.mean(mean_ious))
        guesses.append(np.mean(mean_guesses))

        optimistic_ious.append(best_iou)
        optimistic_guesses.append(get_boxes_correct_guesses(best_boxes_predicted, best_boxes_gt))
    return {
        DetectorMetric.MeanProcessingTime.value: np.mean(times).item(),
        DetectorMetric.StdProcessingTime.value: np.std(times).item(),

        DetectorMetric.IoU.value: np.mean(ious).item(),
        DetectorMetric.Precision.value: (np.sum(guesses) / total_predicted_boxes).item(),
        DetectorMetric.Recall.value: (np.sum(guesses) / total_gt_boxes).item(),

        DetectorMetric.OptimisticIoU.value: np.mean(optimistic_ious).item(),
        DetectorMetric.OptimisticPrecision.value: (
                np.sum(optimistic_guesses) / optimistic_total_predicted_boxes).item(),
        DetectorMetric.OptimisticRecall.value: (np.sum(optimistic_guesses) / optimistic_total_gt_boxes).item(),
    }


def evaluate_detector(test_data_directory: str) -> Dict[float, Dict[str, Dict[str, float]]]:
    global IOU_POSITIVE_THRESHOLD

    logger.info("Evaluating detector ...")
    results = {}
    test_directory = Path(test_data_directory) / "detection"
    iou_thresholds = np.asarray([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], dtype=np.float64)
    for th in iou_thresholds:
        IOU_POSITIVE_THRESHOLD = th
        key = th.item()
        key = round(key, 2)
        results[key] = {}
        logger.info("Evaluating detector ... %s IoU threshold ..." % key)
        for dataset_type in AVAILABLE_DETECTION_TESTING_DATASETS:
            logger.info("Evaluating detector ... %s dataset ..." % dataset_type)
            results[key][dataset_type] = evaluate_detector_for_single_dataset(test_directory / dataset_type)
    return results


def evaluate_classifier_for_each_class(test_directory: Path,
                                       label_encoder: LabelEncoder) -> Dict[str, float]:
    images_in_dir = list(test_directory.rglob("*.jpg"))
    random.shuffle(images_in_dir)
    images_in_dir = images_in_dir

    times = []
    optimistic_predictions = []
    standard_predictions = []
    ground_truth = []

    for image_path in tqdm.tqdm(images_in_dir):
        an_img = cv2.imread(image_path.as_posix())
        an_img = cv2.cvtColor(cv2.cvtColor(an_img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        a_class = str(image_path.parent.name)

        deterministic_augs = CLASSIFICATION_AUGMENTATION.to_deterministic()
        new_img = an_img
        probabilities = np.zeros((NUM_OF_AUGMENTED_SAMPLES, len(label_encoder.classes_)))
        for i in range(NUM_OF_AUGMENTED_SAMPLES):
            new_img = deterministic_augs.augment_image(new_img)
            main_pipeline.perform({
                "boxes": [new_img]
            }, True, steps_names_to_omit=["input", "detector", "decoder", "visualise", "showtime"])
            probabilities[i] = main_pipeline.get_intermediate_output("predicted_probabilities")[0]
            times.append(main_pipeline.timers["classifier"])
        optimistic_probas = np.mean(probabilities, axis=0)
        optimistic_probas /= optimistic_probas.sum(keepdims=True)
        optimistic_class = np.argmax(optimistic_probas)

        main_pipeline.perform({
            "boxes": [an_img]
        }, True, steps_names_to_omit=["input", "detector", "decoder", "visualise", "showtime"])

        predicted_class = main_pipeline.get_intermediate_output("predicted_classes")[0]
        times.append(main_pipeline.timers["classifier"])

        ground_truth.append(label_encoder.transform([a_class])[0])
        standard_predictions.append(predicted_class)
        optimistic_predictions.append(optimistic_class)

    optimistic_recall = metrics.recall_score(ground_truth, optimistic_predictions, average="weighted")
    pessimistic_recall = metrics.recall_score(ground_truth, standard_predictions, average="weighted")

    optimistic_precision = metrics.precision_score(ground_truth, optimistic_predictions, average="weighted")
    pessimistic_precision = metrics.precision_score(ground_truth, standard_predictions, average="weighted")

    optimistic_accuracy = metrics.accuracy_score(ground_truth, optimistic_predictions)
    pessimistic_accuracy = metrics.accuracy_score(ground_truth, standard_predictions)

    optimistic_cf = metrics.confusion_matrix(ground_truth, optimistic_predictions)
    pessimistic_cf = metrics.confusion_matrix(ground_truth, standard_predictions)

    return {
        ClassifierMetric.MeanProcessingTime.value: np.mean(times).item(),
        ClassifierMetric.StdProcessingTime.value: np.std(times).item(),

        ClassifierMetric.Recall.value: float(pessimistic_recall),
        ClassifierMetric.Precision.value: float(pessimistic_precision),
        ClassifierMetric.Accuracy.value: float(pessimistic_accuracy),
        ClassifierMetric.ConfusionMatrix.value: pessimistic_cf,

        ClassifierMetric.OptimisticRecall.value: float(optimistic_recall),
        ClassifierMetric.OptimisticPrecision.value: float(optimistic_precision),
        ClassifierMetric.OptimisticAccuracy.value: float(optimistic_accuracy),
        ClassifierMetric.OptimisticConfusionMatrix.value: optimistic_cf,
    }


def evaluate_classifier_for_class_vs_not_sign(test_directory: Path,
                                              label_encoder: LabelEncoder) -> Dict[str, float]:
    images_in_dir = list(test_directory.rglob("*.jpg"))
    random.shuffle(images_in_dir)
    images_in_dir = images_in_dir

    times = []
    optimistic_predictions = []
    standard_predictions = []
    ground_truth = []

    class_num_for_not_a_sign = label_encoder.transform([common.NO_SIGN_CLASS])[0]

    for image_path in tqdm.tqdm(images_in_dir):
        an_img = cv2.imread(image_path.as_posix())
        an_img = cv2.cvtColor(cv2.cvtColor(an_img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        a_class = str(image_path.parent.name)

        deterministic_augs = CLASSIFICATION_AUGMENTATION.to_deterministic()
        new_img = an_img
        probabilities = np.zeros((NUM_OF_AUGMENTED_SAMPLES, len(label_encoder.classes_)))
        for i in range(NUM_OF_AUGMENTED_SAMPLES):
            new_img = deterministic_augs.augment_image(new_img)
            main_pipeline.perform({
                "boxes": [new_img]
            }, True, steps_names_to_omit=["input", "detector", "decoder", "visualise", "showtime"])
            probabilities[i] = main_pipeline.get_intermediate_output("predicted_probabilities")[0]
            times.append(main_pipeline.timers["classifier"])
        optimistic_probas = np.mean(probabilities, axis=0)
        optimistic_class = np.argmax(optimistic_probas)

        main_pipeline.perform({
            "boxes": [an_img]
        }, True, steps_names_to_omit=["input", "detector", "decoder", "visualise", "showtime"])

        predicted_class = main_pipeline.get_intermediate_output("predicted_classes")[0]
        times.append(main_pipeline.timers["classifier"])

        ground_truth.append(int(class_num_for_not_a_sign == label_encoder.transform([a_class])[0]))
        standard_predictions.append(int(class_num_for_not_a_sign == predicted_class))
        optimistic_predictions.append(int(class_num_for_not_a_sign == optimistic_class))

    optimistic_recall = metrics.recall_score(ground_truth, optimistic_predictions, average="weighted")
    pessimistic_recall = metrics.recall_score(ground_truth, standard_predictions, average="weighted")

    optimistic_precision = metrics.precision_score(ground_truth, optimistic_predictions, average="weighted")
    pessimistic_precision = metrics.precision_score(ground_truth, standard_predictions, average="weighted")

    optimistic_accuracy = metrics.accuracy_score(ground_truth, optimistic_predictions)
    pessimistic_accuracy = metrics.accuracy_score(ground_truth, standard_predictions)

    optimistic_auc = metrics.roc_auc_score(ground_truth, optimistic_predictions, average="weighted")
    pessimistic_auc = metrics.roc_auc_score(ground_truth, standard_predictions, average="weighted")

    optimistic_cf = metrics.confusion_matrix(ground_truth, optimistic_predictions)
    pessimistic_cf = metrics.confusion_matrix(ground_truth, standard_predictions)

    return {
        ClassifierMetric.MeanProcessingTime.value: np.mean(times).item(),
        ClassifierMetric.StdProcessingTime.value: np.std(times).item(),

        ClassifierMetric.Recall.value: float(pessimistic_recall),
        ClassifierMetric.Precision.value: float(pessimistic_precision),
        ClassifierMetric.Accuracy.value: float(pessimistic_accuracy),
        ClassifierMetric.AUC.value: float(pessimistic_auc),
        ClassifierMetric.ConfusionMatrix.value: pessimistic_cf,

        ClassifierMetric.OptimisticRecall.value: float(optimistic_recall),
        ClassifierMetric.OptimisticPrecision.value: float(optimistic_precision),
        ClassifierMetric.OptimisticAccuracy.value: float(optimistic_accuracy),
        ClassifierMetric.OptimisticAUC.value: float(optimistic_auc),
        ClassifierMetric.OptimisticConfusionMatrix.value: optimistic_cf,
    }


def plot_confusion_matrix(cm, classes, output_path: Path,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    output_path = output_path.as_posix()
    plt.figure(figsize=(16, 14))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def evaluate_classifier(test_data_directory: str) -> Dict[str, Dict[str, float]]:
    logger.info("Evaluating classifier ...")
    results = {}
    test_directory = Path(test_data_directory) / "classification"
    label_encoder_path = config["paths"]["label_encoder"]
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pkl.load(f)

    normal_classification = evaluate_classifier_for_each_class(test_directory, label_encoder)
    sign_vs_not_sign = evaluate_classifier_for_class_vs_not_sign(test_directory, label_encoder)

    results["normal classification"] = normal_classification
    results["sign vs not sign"] = sign_vs_not_sign

    return results


def get_time_stamp() -> str:
    return datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")


def finish_detector(test_data_dir: str, output_detector_experiment_path: Path, note: str):
    evaluation_detector = evaluate_detector(test_data_dir)
    output_detector_experiment_path.mkdir(parents=True)
    with open((output_detector_experiment_path / "note.txt").as_posix(), 'w') as f:
        f.write(note)

    with open((output_detector_experiment_path / "results.yaml").as_posix(), 'w') as f:
        yaml.dump(evaluation_detector, f, indent=4, default_flow_style=False)


def finish_classifier(test_data_dir: str, output_classifier_experiment_path: Path, note: str):
    evaluation_classifier = evaluate_classifier(test_data_dir)
    output_classifier_experiment_path.mkdir(parents=True)
    with open((output_classifier_experiment_path / "note.txt").as_posix(), 'w') as f:
        f.write(note)

    label_encoder_path = config["paths"]["label_encoder"]
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pkl.load(f)

    normal_classification = evaluation_classifier["normal classification"]
    sign_vs_not_sign = evaluation_classifier["sign vs not sign"]
    plot_confusion_matrix(normal_classification[ClassifierMetric.ConfusionMatrix.value],
                          classes=label_encoder.classes_, title="Standard confusion matrix",
                          output_path=output_classifier_experiment_path / "cls_vs_cls_standard.pdf")
    plot_confusion_matrix(normal_classification[ClassifierMetric.OptimisticConfusionMatrix.value],
                          classes=label_encoder.classes_, title="Optimistic confusion matrix",
                          output_path=output_classifier_experiment_path / "cls_vs_cls_optimistic.pdf")
    plot_confusion_matrix(sign_vs_not_sign[ClassifierMetric.ConfusionMatrix.value],
                          classes=["sign", "not sign"], title="Standard confusion matrix",
                          output_path=output_classifier_experiment_path / "sign_vs_not_sign_standard.pdf")
    plot_confusion_matrix(sign_vs_not_sign[ClassifierMetric.OptimisticConfusionMatrix.value],
                          classes=["sign", "not sign"], title="Optimistic confusion matrix",
                          output_path=output_classifier_experiment_path / "sign_vs_not_sign_optimistic.pdf")

    normal_classification.pop(ClassifierMetric.ConfusionMatrix.value)
    normal_classification.pop(ClassifierMetric.OptimisticConfusionMatrix.value)

    sign_vs_not_sign.pop(ClassifierMetric.ConfusionMatrix.value)
    sign_vs_not_sign.pop(ClassifierMetric.OptimisticConfusionMatrix.value)

    with open((output_classifier_experiment_path / "results.yaml").as_posix(), 'w') as f:
        yaml.dump(evaluation_classifier, f, indent=4, default_flow_style=False)


def evaluate(test_data_dir: str, test_detector_only: bool, test_classifier_only: bool):
    note = input("Type short note about experiment:\n")
    timestamp = get_time_stamp()
    output_detector_experiment_path = OUTPUT_TESTING_REPORT_DIRECTORY / "detector" / timestamp
    output_classifier_experiment_path = OUTPUT_TESTING_REPORT_DIRECTORY / "classifier" / timestamp

    if test_detector_only:
        finish_detector(test_data_dir, output_detector_experiment_path, note)
    elif test_classifier_only:
        finish_classifier(test_data_dir, output_classifier_experiment_path, note)
    else:
        finish_detector(test_data_dir, output_detector_experiment_path, note)
        finish_classifier(test_data_dir, output_classifier_experiment_path, note)


if __name__ == '__main__':
    import argparse

    argument_parser = argparse.ArgumentParser(description="Script for validating program")
    argument_parser.add_argument("-t", "--test_data_dir", help="Directory of testing data")
    argument_parser.add_argument("--detector", action="store_true", default=False, help="Test detector only")
    argument_parser.add_argument("--classifier", action="store_true", default=False, help="Test classifier only")
    args = argument_parser.parse_args()

    assert not (args.detector and args.classifier), "If you want test both at once, do not put any bool flags"

    evaluate(args.test_data_dir, args.detector, args.classifier)
