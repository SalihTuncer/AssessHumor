from model import NN
import numpy as np
import pandas as pd

from transformers import AlbertTokenizer
from preprocssing import Preprocess
import tensorflow as tf

max_seq_length = 256

if __name__ == '__main__':
    # create an albert model specialised on regression
    model = NN().get_nn()
    # we load the saved weights and  give them to our model
    model.load_weights('rmsprop/rmsprop')

    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    # this dict will help at visualizing the result of our predictions
    funniness = {
        0: 'Not Funny',
        1: 'Slightly Funny',
        2: 'Moderately Funny',
        3: 'Funny',
    }
    # we use the test dataset for the predictions. this is unseen data for the model
    df = pd.read_csv('semeval-2020-task-7-dataset/subtask-1/test.csv')
    # we randomly pick five elements from the dataset
    sample = df.sample(n=5).reset_index()

    for i in range(len(sample)):
        sentence = sample['original'][i]
        edited = sample['edit'][i]
        grade = sample['meanGrade'][i]
        # replace the word with the edited word and return it as a whole sentence
        edited_sentence, _ = Preprocess.reformat_sentence(sentence, edited)
        # we format the sentences with CLS and SEP as well as encode them into numbers via the tokenizer
        tokens = [tokenizer.cls_token_id] + tokenizer.encode(edited_sentence, add_special_tokens=False) + [
            tokenizer.sep_token_id]

        sentence_size = len(tokens)
        # fill the rest  of the array with PAD - tokens reshape it properly and save it as a tensor
        tokens = tf.convert_to_tensor(
            np.array(tokens + [tokenizer.pad_token_id] * (max_seq_length - len(tokens))).reshape((1, max_seq_length))),
        # as many ones as we have words in the sentence. rest is filled up with PAD - tokens, reshaped and returned
        # as a tensor
        attention_mask = tf.convert_to_tensor(
            np.array([1] * sentence_size + [0] * (max_seq_length - sentence_size)).reshape((1, max_seq_length)))
        # now we just give the model the tokens and the attention mask as a dict and start to predict
        test = model.predict(dict(input_ids=tokens, attention_mask=attention_mask))

        # rounding precisely with a little trick :)
        prediction = int(test['logits'][0][0] + 0.5)

        print('\n' + edited_sentence)
        print('\nThe predicted funniness rating: {dict} ({grade:.1f})'.format(dict=funniness[prediction],
                                                                              grade=test['logits'][0][0]))
        print('The true funniness rating: {dict} ({grade})'.format(dict=funniness[int(grade + 0.5)], grade=grade))
