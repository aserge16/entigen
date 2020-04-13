from keras.models import model_from_json
from parse_args import ARGS
from semval_data_process import *
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


# To reproduce, assign prediction values to binary array outputs from
# create_training_data function
prediction_values = {
    0: 'Cause-Effect',
    1: 'Component-Whole',
    2: 'Content-Container',
    3: 'Entity-Destination',
    4: 'Entity-Origin',
    5: 'Instrument-Agency',
    6: 'Member-Collection',
    7: 'Message-Topic',
    8: 'Other',
    9: 'Product-Producer',
    10: "Exit",
}


def validate():
    sent_train, sent_test, label_train, label_test, tokenizer = create_training_data(ARGS.train_path,
                                                                                    ARGS.test_path, 
                                                                                    num_words = ARGS.num_words, 
                                                                                    max_len=ARGS.max_len,
                                                                                    test_size=0.99)
    model = load_model(ARGS.model_path, ARGS.model_weights_path)
    
    print("Testing model...")
    score = model.evaluate(sent_test, label_test, batch_size = 40)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

    confusion = input("Do you wish to create a confusion matrix? yes/no: ")
    if confusion == "yes":
        predictions = model.predict_classes(sent_test)
        correct = [np.where(r==1)[0][0] for r in label_test] # convert from one-hot encoding
        matrix = confusion_matrix(y_true=correct, y_pred=predictions)
        print(matrix)


def predict_classes(sentences):
    data = create_model_data(sentences, ARGS.num_words, ARGS.max_len)
    model = load_model(ARGS.model_path, ARGS.model_weights_path)

    predictions = model.predict_classes(data)

    return predictions


def load_model(model_path, model_weights_path):
    print("Loading model from disk...")

    with open(model_path, 'r') as fh:
        loaded_model_json = fh.read()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(model_weights_path)
        print("Model successfully loaded from disk")
    
    loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return loaded_model
