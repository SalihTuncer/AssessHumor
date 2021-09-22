from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError

from model import NN
from preprocssing import Preprocess

max_seq_length = 256
batch_size = 16

if __name__ == '__main__':

    # preprocess the datasets and return a utility class which carries the datasets
    prep = Preprocess(max_seq_length=max_seq_length, batch_size=batch_size)

    train_dataset = prep.get_train()

    dev_dataset = prep.get_dev()
    # create an albert model specialised on regression
    model = NN().get_nn()

    optimizer = RMSprop(learning_rate=1e-5, decay=1e-8)
    # we evaluate with RMSE
    model.compile(optimizer=optimizer, loss=MeanSquaredError(),
                  metrics=[RootMeanSquaredError('root_mean_squared_error')])

    print("Fit model on training data.")

    model.fit(train_dataset, validation_data=dev_dataset, epochs=3)

    print('Model successfully saved.')

    print(model.summary())

    test_dataset = prep.get_test()

    print('Evaluate model with test data.')

    print(model.evaluate(test_dataset, return_dict=True))

    print('Save the model weights.')

    model.save_weights('rmsprop/rmsprop')
