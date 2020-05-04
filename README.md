# entigen
## Aleksandr Sergeev
Final project for my computer science major at Earlham College.

[ --> Project Paper](https://www.overleaf.com/8823925963hjqntbvwmptf)</br>

python3 packages required are found in requirements.txt

----------
HOW TO USE
----------
All resources with sentences to classify need to go in the resources/ directory and have the .txt/.pdf extension.
All predictions will currently save to the resources/predictions/ directory.

There is a pre-trained model with the code, trained over 20 epochs and using an embedding layer. GloVe embeddings can be downloaded free from https://nlp.stanford.edu/projects/glove/.

All arguements can be changed in the parse_args.py file.


## Using the Driver File

 There is a file, driver.py, which lets the you train, validate, and classify sentences through the terminal. To run this, you need to change your current terminal directory to the code subdirectory and use the command '$ python driver.py'.

## Using the GUI

 If you want to use the interactive GUI for predictions, you simply have to run '$ python gui_main.py' in the code subdirectory.

## PyInstaller

 PyInstaller is a libary that you can use to bundle the whole GUI into an executable. You will need to first install PyInstaller.
