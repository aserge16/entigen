from code.parse_args import ARGS
from code.text_process import *
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
        for F in (ActionPage, PredictionPage):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("PredictionPage")

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
                                    command=lambda: controller.show_frame("PredictionPage"))
        predict_button.pack()
        train_button = tk.Button(self, height=1, width=12, text="Train model",
                                    command=lambda: controller.show_frame("TrainPage"))
        train_button.pack()
        validate_button = tk.Button(self, height=1, width=12, text="Validate model",
                                    command=lambda: controller.show_frame("ValidatePage"))
        validate_button.pack()


class PredictionPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.ent_predictions = {}
        files = os.listdir('./resources/')

        select_label = tk.Label(self, text="Select file to process")
        select_label.pack()

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
            self.data = TextProcess(ARGS.resources_path + file_name, PDF=True)
        elif ".txt" in file_name:
            self.data = TextProcess(ARGS.resources_path + file_name, PDF=False)
        else:
            self.data = None
            return
        self.data.ie_preprocess()

if __name__ == "__main__":
    gui = Entigen()
    gui.mainloop()
