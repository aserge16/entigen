from parse_args import ARGS
from text_process import *
from train import *
from validate import *
import os


actions = ["predict", "train model", "validate model", "exit session"]


def driver():
    while True:
        print("\nActions:")
        for i, action in enumerate(actions):
            print(str(i) + " - " + action)
        idx = int(input("Enter corresponding numer to action:"))

        if idx == 0:
            predict()
        elif idx == 1:
            train_model()
        elif idx == 2:
            validate_model()
        elif idx == 3:
            print("Thank you!")
            return
        else:
            print("Incorrect input, please try again")


def predict():
    entries = os.listdir(ARGS.resources_path)
    entries.append("exit")

    while True:
        print("\nFile selection:")
        for i, file in enumerate(entries):
            print(str(i) + " - " + file)
        idx = int(input("Enter corresponding number to file selection:"))
        try:    
            print("Selection: " + entries[idx])
            break
        except IndexError:
            print("Incorrect input, please try again")
    
    if idx == len(entries) - 1:
        print("Exiting predict")
        return
    

    ent_to_sentences = {}
    data = TextProcess(ARGS.resources_path + entries[idx], PDF=False)
    data.ie_preprocess()
    print("pdf pre-processing completed. \n")

    ent_request = ""
    while True:
        ent_request = input("Enter entity, or 'exit' to return: ")
        if ent_request == "exit":
            break
        if ent_request not in ent_to_sentences:
            ent_to_sentences[ent_request] = data.ent_preprocess(ent_request)
        pred = predict_classes(ent_to_sentences[ent_request])


def train_model():
    train()


def validate_model():
    validate()


if __name__=='__main__':
    driver()
