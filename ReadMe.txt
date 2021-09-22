Necessary libraries can be installed via:

$pip install -r requirements.txt

All the listed libraries in the requirements.txt are really necessary.
The are not libraries from my local computer. They are all installed in an virtual environment.

The project structure has to be like this:

├── ReadMe.txt
├── model.py
├── preprocssing.py
├── requirements.txt
├── rmsprop                         <- directory
│ ├── checkpoint
│ ├── rmsprop.data-00000-of-00001
│ └── rmsprop.index
├── semeval-2020-task-7-dataset      <- directory
│ ├── README.txt
│ └── subtask-1                      <- directory
│     ├── baseline.zip
│     ├── dev.csv
│     ├── test.csv
│     ├── train.csv
│     └── train_funlines.csv
├── test.py
└── train-evaluate-main.py

3 directories, 16 files

We use Python3.8.

For training and evaluating the model we use the command:

$python3 train-evaluate-main.py

For testing the model we use the command:

$python3 test.py

It will randomly pick five edited headlines from the test dataset and predict the funniness of it.
Additionally i printed out the float values next to it, so you can see how good or bad it is. :)

