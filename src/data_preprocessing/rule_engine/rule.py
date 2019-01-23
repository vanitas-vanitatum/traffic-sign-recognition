class Rule:

    def __init__(self, predicate, inference):
        self.predicate = predicate
        self.inference = inference

    def __call__(self, sample):
        if self.predicate(sample):
            return self.inference(sample)
        else:
            return None
