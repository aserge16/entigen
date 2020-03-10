from keras.models import model_from_json
from parse_args import ARGS
from semval_data_process import *


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
}


def validate():
    sent_train, sent_test, label_train, label_test, tokenizer = create_training_data(ARGS.train_path,
                                                                                    ARGS.test_path, 
                                                                                    num_words = ARGS.num_words, 
                                                                                    max_len=ARGS.max_len,
                                                                                    test_size=0.99)
    model = load_model(ARGS.model_path)
    
    print("Testing model...")
    score = model.evaluate(sent_test, label_test, batch_size = 40)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))


def predict(sentences):
    data = create_model_data(sentences, ARGS.num_words, ARGS.max_len)
    model = load_model(ARGS.model_path)

    predictions = model.predict_classes(data)

    return predictions


def load_model(model_path):
    print("Loading model from disk...")

    with open(model_path, 'r') as fh:
        loaded_model_json = fh.read()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model.h5")
        print("Model successfully loaded from disk")
    
    loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return loaded_model


if __name__=='__main__':
    predict()
