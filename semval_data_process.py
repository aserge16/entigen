import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


def process_file(input_path):
    sentences = []
    labels = []

    with open(input_path, 'r') as fh:
        lines = fh.readlines()

    for i in range(0, len(lines), 4):
        line = lines[i]
        start_sent = line.find('"')
        sentence = line[start_sent+1:len(line) - 2]
        sentence = sentence.replace("<e1>", "E1_START ").replace("</e1>", " E1_END")
        sentence = sentence.replace("<e2>", "E2_START ").replace("</e2>", " E2_END")
        label = lines[i+1].rstrip()

        sentences.append(sentence)
        labels.append(label)

    return sentences, labels


def get_sentences_labels(input_path_1, input_path_2):
    sentences, labels = process_file(input_path_1)
    sentences_2, labels_2 = process_file(input_path_2)

    sentences.extend(sentences_2)
    labels.extend(labels_2)

    return sentences, labels


def create_training_data(sentences, labels, num_words, max_len):
    t = Tokenizer(num_words=num_words)
    t.fit_on_texts(sentences)
    sequences = t.texts_to_sequences(sentences)

    train_data = pad_sequences(sequences, maxlen=max_len)    

    lb = LabelBinarizer()
    labels = np.array(labels)
    train_labels = lb.fit_transform(labels)
    
    sent_train, sent_test, label_train, label_test = train_test_split(train_data , train_labels, test_size=0.20, random_state=42)

    return sent_train, sent_test, label_train, label_test
