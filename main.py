from text_process import *
import spacy


def main():
    pdf_path = input("Enter file path: ")
    data = TextProcess(pdf_path)
    data.extract_text_from_pdf()
    data.ie_preprocess()

    print("pdf pre-processing completed. \n")

    ent_request = ""

    while ent_request != "exit":
        ent_request = input("Enter entity: ")
        data.ent_preprocess(ent_request)
    return


if __name__=='__main__':
    main()