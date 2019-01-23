from src.data_preprocessing.base_data_interface import Sample
from src.data_preprocessing.rule_engine.rule import Rule


class Predicate:

    def __call__(self, sample: Sample) -> bool:
        raise NotImplementedError

    def __mul__(self, value):
        return AndPredicate(self, value)

    def __and__(self, other):
        return AndPredicate(self, other)

    def then(self, inference):
        return Rule(self, inference)


class AndPredicate(Predicate):

    def __init__(self, predicate_1, predicate_2):
        self.pred_1 = predicate_1
        self.pred_2 = predicate_2

    def __call__(self, sample: Sample) -> bool:
        return self.pred_1(sample) and self.pred_2(sample)


class IfClassIs(Predicate):

    def __init__(self, class_name):
        self.class_name = class_name

    def __call__(self, sample: Sample) -> bool:
        return sample.label == self.class_name


class IfDatasetIs(Predicate):

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def __call__(self, sample: Sample) -> bool:
        return sample.origin == self.dataset_name


class IfDatasetIsnt(Predicate):

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def __call__(self, sample: Sample) -> bool:
        return sample.origin != self.dataset_name


class IfClassIn(Predicate):
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, sample: Sample) -> bool:
        return sample.label in self.classes


class IfClassStartsWith(Predicate):
    def __init__(self, phrase):
        self.phrase = phrase

    def __call__(self, sample: Sample) -> bool:
        return sample.label.startswith(self.phrase)


class IfTrue(Predicate):

    def __call__(self, sample: Sample) -> bool:
        return True