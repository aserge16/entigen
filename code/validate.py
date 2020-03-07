from keras.models import model_from_json
from semval_data_process import create_training_data


model_path = './model.json'
train_path = "./data/TRAIN_FILE.TXT"
test_path = "./data/TEST_FILE_FULL.TXT"
num_words = 20000
max_len = 100


def validate():
    sent_train, sent_test, label_train, label_test, tokenizer = create_training_data(train_path,
                                                                                    test_path, 
                                                                                    num_words = num_words, 
                                                                                    max_len=max_len,
                                                                                    test_size=0.99)
    model = load_model(model_path)
    
    print("Testing model...")
    score = model.evaluate(sent_test, label_test, batch_size = 40)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))


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
    validate()
