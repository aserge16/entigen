from parse_args import ARGS
from text_process import *
import os
import train
import validate


actions = ["predict", "train model", "validate model", "exit session"]


def driver():
    while True:
        print("\nActions:")
        for i, action in enumerate(actions):
            print(str(i) + " - " + action)
        idx = int(input("Enter corresponding numer to action:"))
        if idx == 0:
            predict()
            print("Predictions completed. Thank you!")
            return
        elif idx == 1:
            train_model()
            print("New model trained. Thank you!")
            return
        elif idx == 2:
            validate_model()
            print("Model validated. Thank you!")
            return
        elif idx == 3:
            print("Thank you!")
            return
        else:
            print("Incorrect input, please try again")


def train_model():
    train.train()


def validate_model():
    validate.validate()


def predict():
    entries = os.listdir(ARGS.resources_path)
    entries.append("exit")

    while True:
        print("\nFile selection:")
        for i, file in enumerate(entries):
            print(str(i) + " - " + file)
        idx = int(input("Enter corresponding number to file selection:"))
        if idx == len(entries) - 1:
            print("Exiting predict")
            return
        try:
            file_name = entries[idx]
            print("Selection: " + entries[idx])
            if ".pdf" in file_name:
                data = TextProcess(ARGS.resources_path + file_name, PDF=True)
            elif ".txt" in file_name:
                data = TextProcess(ARGS.resources_path + file_name, PDF=False)
            else:
                print("Unknown file format, please use only .txt or .pdf files")
        except IndexError:
            print("Incorrect input, please try again")

        ent_predictions = {}
        data.ie_preprocess()
        print("pdf pre-processing completed. \n")

        ent_request = ""
        while True:
            ent_request = input("Enter entity, or 'exit' to return: ")
            if ent_request == "exit":
                break
            if ent_request not in ent_predictions:
                pred_to_sentences = {}

                ent_sentences = data.ent_preprocess(ent_request)
                predictions = validate.predict_classes(ent_sentences)

                for i, (pred, sent) in enumerate(zip(predictions, ent_sentences)):
                    sent = TextProcess.restore_sentence(sent)
                    if pred not in pred_to_sentences:
                        pred_to_sentences[pred] = [sent]
                    elif sent not in pred_to_sentences[pred]:
                        pred_to_sentences[pred].append(sent)
                ent_predictions[ent_request] = pred_to_sentences
                view_pred = input("Do you wish to view the predictions? yes/no: ")
                if view_pred == "yes":
                    display_predictions(ent_request, pred_to_sentences)

        to_save = input("Do you wish to save all predictions to file? yes/no: ")
        if to_save == "yes":
            with open(ARGS.resources_path + "predictions_" + file_name, "w") as fh:
                for ent in ent_predictions:
                    fh.write("\n\n ENTITY - " + ent + "\n")
                    for pred in ent_predictions[ent]:
                        fh.write(validate.prediction_values[pred] + ":\n")
                        for sent in ent_predictions[ent][pred]:
                            fh.write("\t" + sent + "\n")
            print("Predictions saved to " + ARGS.resources_path + "predictions_" + file_name)


def display_predictions(ent, pred_to_sentences):
    print("Displaying predictions for entity %s" % (ent))
    while True:
        for key, value in validate.prediction_values.items():
            print(key, value)
        try:
            idx = int(input("Enter number corresponding with class you wish to view: "))
        except ValueError:
            print("Please enter only an integer")
            continue
        if not 0 <= idx <= 10:
            print("Incorrect input, please try again")
            continue
        elif idx == 10:
            break
        else:
            try:
                sentences = pred_to_sentences[idx]
            except KeyError:
                print("No sentences under this class.")
                continue
            count_sent = len(sentences)
            display = 0
            print("\n\nTotal sentences for this class is %d" % (count_sent))
            print("Displaying sentences in batches of 10 \n")
            while display < count_sent:
                print(sentences[display])
                display += 1
                if display % 10 == 0:
                    cont = input("\nDo you wish to display another batch? yes/no: ")
                    if cont == "no":
                        break
                    print("\n")
            print("\n\n")


if __name__=='__main__':
    driver()
