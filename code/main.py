from text_process import *


def main():
    # pdf_path = input("Enter file path: ")
    pdf_path = "../resources/hp1.pdf"
    data = TextProcess(pdf_path, PDF=True)
    data.ie_preprocess()

    print("pdf pre-processing completed. \n")

    # ent_request = ""

    # while ent_request != "exit":
    #     ent_request = input("Enter entity: ")
    data.ent_preprocess("Harry")
    

if __name__=='__main__':
    main()
