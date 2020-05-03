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

        # START FRAME
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

        self.listbox = tk.Listbox(self)
        self.listbox.pack()

        delete_entry = tk.Button(self, text="Delete Entity",
                                command=lambda: self.delete_entity())
        delete_entry.pack()
        show_entry = tk.Button(self, text="Show Entity Predictions",
                                command=lambda: self.show_predictions())
        show_entry.pack()
        save_all = tk.Button(self, text="Save predictions",
                            command=lambda: self.save_all())
        save_all.pack()
        back = tk.Button(self, text="Back",
                            command=lambda: self.back())
        back.pack()


    def delete_entity(self):
        lb=self.listbox
        entity = lb.get(tk.ACTIVE).split(" - ")[0]

        del self.ent_predictions[entity]
        lb.delete(tk.ANCHOR)


    def save_all(self):
        file_name = self.processing_file.get().split()[2]
        with open(ARGS.resources_path + "predictions_" + file_name, "w") as fh:
            for ent in self.ent_predictions:
                fh.write("\n\nENTITY - " + ent + "\n")
                for pred in self.ent_predictions[ent]:
                    fh.write(validate.prediction_values[pred] + ":\n")
                    for sent in self.ent_predictions[ent][pred]:
                        fh.write("\t" + sent + "\n")


    def back(self):
        self.ent_predictions = {}
        self.listbox.delete(0, tk.END)
        self.controller.show_frame("ProcessTextPage")


    def show_predictions(self):
        lb=self.listbox
        entity = lb.get(tk.ACTIVE).split(" - ")[0]
        predictions = self.ent_predictions[entity]

        display_string = "ENTITY: " + entity
        for pred in predictions:
            display_string += "\n\n" + validate.prediction_values[pred] + ":\n"
            for sent in predictions[pred]:
                display_string += " - " + sent + "\n"

        top = tk.Toplevel(self)
        s = tk.Scrollbar(top)
        text= tk.Text(top)
        text.pack(side=tk.LEFT, fill="both", expand=True)
        s.pack(side=tk.RIGHT, fill=tk.Y)
        s.config(command=text.yview)
        text.config(yscrollcommand=s.set)
        text.insert(tk.END, display_string)


    def get_entity_sentences(self, entity_input):
        ent_request = entity_input.get()
        if ent_request not in self.ent_predictions:
            pred_to_sentences = {}
            ent_sentences = self.data.ent_preprocess(ent_request)
            count = len(ent_sentences)
            self.listbox.insert(tk.END, ent_request + " - " + str(count))
            if count == 0:
                self.ent_predictions[ent_request] = pred_to_sentences
                return

            predictions = validate.predict_classes(ent_sentences)

            for i, (pred, sent) in enumerate(zip(predictions, ent_sentences)):
                sent = TextProcess.restore_sentence(sent)
                if pred not in pred_to_sentences:
                    pred_to_sentences[pred] = [sent]
                elif sent not in pred_to_sentences[pred]:
                    pred_to_sentences[pred].append(sent)
            self.ent_predictions[ent_request] = pred_to_sentences


if __name__ == "__main__":
    gui = Entigen()
    gui.mainloop()
