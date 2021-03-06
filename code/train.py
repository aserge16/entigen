from entigen import create_model
from parse_args import ARGS
from semval_data_process import create_training_data
from keras.models import model_from_json


def train():
    print("Generating train data")
    sent_train, sent_test, label_train, label_test, tokenizer = create_training_data(ARGS.train_path,
                                                                                    ARGS.test_path, 
                                                                                    num_words = ARGS.num_words, 
                                                                                    max_len=ARGS.max_len,
                                                                                    test_size=0.15)
    model = create_model(num_words=ARGS.num_words, 
                        embedding_size=ARGS.embedding_size,
                        max_len=ARGS.max_len, 
                        label_len=10,
                        word_index = tokenizer.word_index,
                        embedding = ARGS.embedding)
    model.fit(sent_train, label_train, epochs=ARGS.num_epochs, batch_size = ARGS.batch_size)

    print("Training completed")
    print("Testing model...")
    score = model.evaluate(sent_test, label_test, batch_size = ARGS.batch_size)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

    print("Saving model to disk...")

    model_json = model.to_json()
    with open(ARGS.model_path, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(ARGS.model_weights_path)
    print("Model saved to disk")
