from parse_args import ARGS
from text_process import *
import validate
import tkinter as tk
import os


class Entigen(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title("entigen")
        self.geometry('800x600')
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (ActionPage, ProcessTextPage, PredictionPage):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("ProcessTextPage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()


class ActionPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        actions_label = tk.Label(self, text="Available Actions")
        actions_label.pack()
        predict_button = tk.Button(self, height=1, width=12, text="Predict",
                                    command=lambda: controller.show_frame("ProcessTextPage"))
        predict_button.pack()
        train_button = tk.Button(self, height=1, width=12, text="Train model",
                                    command=lambda: controller.show_frame("TrainPage"))
        train_button.pack()
        validate_button = tk.Button(self, height=1, width=12, text="Validate model",
                                    command=lambda: controller.show_frame("ValidatePage"))
        validate_button.pack()


class ProcessTextPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        files = os.listdir(ARGS.resources_path)

        select_label = tk.Label(self, text="Select .pdf/.txt file to process").pack()

        var = tk.StringVar()
        for file in files:
            if ".pdf" in file or ".txt" in file:   
                rb = tk.Radiobutton(self, text=file, value=file, variable=var)
                rb.pack()
        
        process_button = tk.Button(self, text="Process selected file",
                                    command=lambda: self.preprocess_file(var.get()))
        process_button.pack()


    def preprocess_file(self, file_name):
        if ".pdf" in file_name:
            data = TextProcess(ARGS.resources_path + file_name, PDF=True)
        elif ".txt" in file_name:
            data = TextProcess(ARGS.resources_path + file_name, PDF=False)
        else:
            return
        data.ie_preprocess()
        prediction_frame = self.controller.frames["PredictionPage"]
        prediction_frame.data = data
        prediction_frame.processing_file.set("Processing file " + file_name)
        self.controller.show_frame("PredictionPage")


class PredictionPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.processing_file = tk.StringVar()
        self.data = None
        self.ent_predictions = {}

        file_name = tk.Label(self, textvariable=self.processing_file).pack()

        entity = tk.Label(self, text="Entity Name").pack()
        entity_input = tk.Entry(self)
        entity_input.pack()

        process_entity = tk.Button(self, text="Process entity", 
                            command=lambda: self.get_entity_sentences(entity_input))
        process_entity.pack()


    def get_entity_sentences(self, entity_input):
        ent_request = entity_input.get()
        if ent_request not in self.ent_predictions:
            pred_to_sentences = {}
            ent_sentences = self.data.ent_preprocess(ent_request)
            predictions = validate.predict_classes(ent_sentences)

            for i, (pred, sent) in enumerate(zip(predictions, ent_sentences)):
                sent = TextProcess.restore_sentence(sent)
                if pred not in pred_to_sentences:
                    pred_to_sentences[pred] = [sent]
                elif sent not in pred_to_sentences[pred]:
                    pred_to_sentences[pred].append(sent)
            print(pred_to_sentences)


if __name__ == "__main__":
    gui = Entigen()
    gui.mainloop()
