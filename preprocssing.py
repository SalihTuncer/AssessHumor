import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import AlbertTokenizer


class Preprocess:

    def __init__(self,
                 train_path='semeval-2020-task-7-dataset/subtask-1/train.csv',
                 dev_path='semeval-2020-task-7-dataset/subtask-1/dev.csv',
                 test_path='semeval-2020-task-7-dataset/subtask-1/test.csv',
                 max_seq_length=256, batch_size=16
                 ):
        self._max_seq_length = max_seq_length

        self._tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        print('Tokenizer ready...\n')
        train_df = pd.read_csv(train_path)
        dev_df = pd.read_csv(dev_path)
        test_df = pd.read_csv(test_path)
        print('Datasets were read...\n')
        # save the data appropriate as a class property
        self._train = self._process_corpus(train_df)
        self._train = tf.data.Dataset.from_tensor_slices(self._train).batch(batch_size)
        print('Training-dataset preprocessed and ready...\n')

        self._dev = self._process_corpus(dev_df)
        self._dev = tf.data.Dataset.from_tensor_slices(self._dev).batch(batch_size)
        print('Dev-dataset preprocessed and ready...\n')

        self._test = self._process_corpus(test_df)
        self._test = tf.data.Dataset.from_tensor_slices(self._test).batch(batch_size)
        print('Test-dataset preprocessed and ready...\n')

        print('Preprocessing done.\n')

    def _process_corpus(self, df: pd.DataFrame) -> ({str: np.ndarray}, np.ndarray):
        tokens = np.zeros((len(df), self._max_seq_length), dtype='int32')
        attention_mask = np.zeros((len(df), self._max_seq_length), dtype='int32')

        labels = df['meanGrade'].to_numpy(dtype='float32')

        for i in range(len(df)):
            # reformat the sentence and return additionally the replaced word as a whole sentence
            edited_sentence, sentence = self.reformat_sentence(df['original'][i], df['edit'][i])
            # save the encoded sentences in each row of the matrix
            tokens[i, :] = self.tokenize_sentence(edited_sentence, sentence)

            sentence_size = len(sentence.split(' ')) + 2  # CLS-token at the beginning and SEP-token at the end

            total_sentence_size = (sentence_size * 2) - 1  # second sentence has no CLS-token
            # fill the array with as many ones as we have words
            attention_mask[i, :total_sentence_size] = 1.0
        # return appropriate so albert can use it
        return {'input_ids': tokens,
                'attention_mask': attention_mask
                }, labels

    @staticmethod
    def reformat_sentence(sentence: str, edit: str) -> (str, str):
        # remove <..../> for the original sentence
        sen = sentence.split('<')
        sen[1] = sen[1].replace('/>', '')
        # and put the edited word in the <..../> and save it as a sentence
        sen[1:2] = sen[1].split(' ', 1)
        edited = sen[0] + edit + sen[2]
        return edited, ''.join(sen)

    # [CLS], ..., [SEP], ..., [SEP] | rest filled with [PAD]
    def tokenize_sentence(self, sentence: str, edited_sentence: str) -> [int]:
        # we want to place the special tokens ourself because we need 1 cls and 2 seps
        tokens = [self._tokenizer.cls_token_id] + self._tokenizer.encode(sentence, add_special_tokens=False) \
                 + [self._tokenizer.sep_token_id] + self._tokenizer.encode(edited_sentence, add_special_tokens=False) \
                 + [self._tokenizer.sep_token_id]
        # now we add the PAD-tokens at the end as filler tokens
        return tokens + [self._tokenizer.pad_token_id] * (self._max_seq_length - len(tokens))

    def get_train(self) -> tf.data.Dataset:
        return self._train

    def get_dev(self) -> tf.data.Dataset:
        return self._dev

    def get_test(self) -> tf.data.Dataset:
        return self._test
