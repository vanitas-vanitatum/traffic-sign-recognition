from src.data_preprocessing.rule_engine.rule import Rule
from src.data_preprocessing.rule_engine.predicate import (IfClassIs, IfDatasetIs, IfDatasetIsnt,
                                                          IfTrue, IfClassIn, IfClassStartsWith)
from src.data_preprocessing.rule_engine.inference import Flip, Rotate, Transpose, ChangeLabel, Identity
