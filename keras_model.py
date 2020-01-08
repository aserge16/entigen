from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional


def create_model(num_words, embedding_size, max_len, label_len):
    model = Sequential()

    model.add(Embedding(num_words, embedding_size, input_length=max_len))
    model.add(Bidirectional(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(label_len, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    print(model.summary())
    return model
