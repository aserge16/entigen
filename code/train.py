from entigen import *
from semval_data_process import *


train_path = "./data/TRAIN_FILE.TXT"
test_path = "./data/TEST_FILE_FULL.TXT"


def test():
    sent_train, sent_test, label_train, label_test = create_training_data(train_path, 
                                                                            test_path, 
                                                                            num_words = 20000, 
                                                                            max_len=100)
    model = create_model(num_words=20000, embedding_size=300, max_len=100, label_len=19)
    model.fit(sent_train, label_train, epochs=10, batch_size = 40)
    # score = model.evaluate(sent_test, label_test, batch_size = 120)

if __name__=='__main__':
    test()