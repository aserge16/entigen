import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


def get_sentences_labels(input_path):
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
        label = lines[i+1].rstrip().split("(")[0]

        sentences.append(sentence)
        labels.append(label)

    return sentences, labels


def create_training_data(train_data_path, test_data_path, num_words, max_len, test_size):
    sentences, labels = get_sentences_labels(train_data_path)
    sentences_2, labels_2 = get_sentences_labels(test_data_path)

    sentences.extend(sentences_2)
    labels.extend(labels_2)

    t = Tokenizer(num_words=num_words)
    t.fit_on_texts(sentences)
    sequences = t.texts_to_sequences(sentences)

    train_data = pad_sequences(sequences, maxlen=max_len)    

    lb = LabelBinarizer()
    labels = np.array(labels)
    train_labels = lb.fit_transform(labels)

    sent_train, sent_test, label_train, label_test = train_test_split(train_data , train_labels, test_size=test_size, random_state=42)

    return sent_train, sent_test, label_train, label_test, t


def create_model_data(sentences, num_words, max_len):
    t = Tokenizer(num_words=num_words)
    t.fit_on_texts(sentences)
    sequences = t.texts_to_sequences(sentences)

    data = pad_sequences(sequences, maxlen=max_len)

    return data
