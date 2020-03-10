import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional


def create_model(num_words, embedding_size, max_len, label_len, word_index, embedding_file=None):
    print("Creating model")
    model = Sequential()

    # if embedding_file is not None:
    #     print("Creating word embedding layer...\n")
    #     embeddings_index, dim = create_word_embedding(embedding_file)
    #     embedding_matrix = np.zeros((len(word_index) + 1, dim))
    #     for word, i in word_index.items():
    #         embedding_vector = embeddings_index.get(word)
    #         if embedding_vector is not None:
    #             embedding_matrix[i] = embedding_vector

    #     embedding_layer = Embedding(len(embedding_matrix), dim, weights=[embedding_matrix], input_length=max_len, trainable=False)
    #     model.add(embedding_layer)
    #     model.add(Embedding())

    model.add(Embedding(num_words, embedding_size, input_length=max_len))
    model.add(Bidirectional(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(label_len, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


def create_word_embedding(embedding_path):
    embeddings_index = dict()

    print("Loading embedding file...\n")
    with open(embedding_path, 'r') as fh:
        lines = fh.readlines()

    for line in lines:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    dim = len(lines[-1].split()) - 1
    print("Loaded embedding file of dimension size " + str(dim) + "\n")

    return embeddings_index, dim
