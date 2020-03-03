from entigen import *
from semval_data_process import *
from keras.models import model_from_json



train_path = "./data/TRAIN_FILE.TXT"
test_path = "./data/TEST_FILE_FULL.TXT"
embedding_file = "./data/glove.6B.50d.txt"
num_words = 20000
max_len = 100


def train():
    sent_train, sent_test, label_train, label_test, tokenizer = create_training_data(train_path,
                                                                                    test_path, 
                                                                                    num_words = num_words, 
                                                                                    max_len=max_len)
    model = create_model(num_words=num_words, 
                        embedding_size=300, 
                        max_len=max_len, 
                        label_len=10,
                        word_index = tokenizer.word_index,
                        embedding_file = embedding_file)
    model.fit(sent_train, label_train, epochs=5, batch_size = 40)

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Saved model to disk")

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    score = loaded_model.evaluate(sent_test, label_test, batch_size = 120)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


if __name__=='__main__':
    train()
