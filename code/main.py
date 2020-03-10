from text_process import *
import validate


def main():
    ent_to_sentences = {}

    # pdf_path = input("Enter file path: ")
    pdf_path = "../resources/chp.txt"
    data = TextProcess(pdf_path, PDF=False)
    data.ie_preprocess()
    print("pdf pre-processing completed. \n")

    # ent_request = ""

    # while ent_request != "exit":
    #     ent_request = input("Enter entity: ")
    ent_to_sentences["Harry"] = data.ent_preprocess("Harry")

    pred = validate.predict(ent_to_sentences['Harry'])


if __name__=='__main__':
    main()
