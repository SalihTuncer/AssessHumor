from transformers import TFAlbertForSequenceClassification


class NN:
    def __init__(self):
        self._nn = self._create_nn()

    def _create_nn(self) -> TFAlbertForSequenceClassification:
        return TFAlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=1)

    def get_nn(self):
        return self._nn
