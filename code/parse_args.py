from argparse import ArgumentParser
import sys

def parse_args():
    parser = ArgumentParser()

    # FILES
    parser.add_argument("--model_path", default="./model.json",
                        type=str, help="Model file path")
    parser.add_argument("--model_weights_path", default="./model.h5",
                        type=str, help="Model weights file path")
    parser.add_argument("--train_path", default="./data/TRAIN_FILE.TXT",
                        type=str, help="Train data file path")
    parser.add_argument("--test_path", default="./data/TEST_FILE_FULL.TXT",
                        type=str, help="Test data file path")
    parser.add_argument("--data_path", default="./data/temp_sentences.txt",
                        type=str, help="Processed data file path")
    parser.add_argument("--resources_path", default="../resources/",
                        type=str, help="Resources file path")
    parser.add_argument("--embedding_path", default="./data/glove.840B.300d.txt",
                        type=str, help="Word embedding file path")
    parser.add_argument("--predictions_save_path", default="./predictions/",
                        type=str, help="Prediction files save path")

    # VALUES
    parser.add_argument("--embedding", default=False,
                        type=bool, help="To use word embedding or not")
    parser.add_argument("--batch_size", default=40,
                        type=int, help="Training batch size")
    parser.add_argument("--max_len", default=100,
                        type=int, help="Max sentence length in data")
    parser.add_argument("--num_words", default=20000,
                        type=int, help="Max number of words to track")
    parser.add_argument("--num_epochs", default=10,
                        type=int, help="Number of epochs")
    parser.add_argument("--embedding_size", default=300,
                        type=int, help="Embedding layer size")

    args = parser.parse_args()
    return args


ARGS = parse_args()
