from argparse import ArgumentParser
import sys

def parse_args():
    parser = ArgumentParser()

    # FILES
    parser.add_argument("--model_path", default="./model.json",
                        type=str, help="Model file path")
    parser.add_argument("--train_path", default="./data/TRAIN_FILE.TXT",
                        type=str, help="Train data file path")
    parser.add_argument("--test_path", default="./data/TEST_FILE_FULL.TXT",
                        type=str, help="Test data file path")
    parser.add_argument("--data_path", default="./data/temp_sentences.txt",
                        type=str, help="Processed data file path")

    # VALUES
    parser.add_argument("--max_len", default=100,
                        type=int, help="Max sentence length in data")
    parser.add_argument("--num_words", default=20000,
                        type=int, help="Max number of words to track")
    parser.add_argument("--num_epochs", default=10,
                        type=int, help="Number of epochs")

    args = parser.parse_args()
    return args


ARGS = parse_args()
