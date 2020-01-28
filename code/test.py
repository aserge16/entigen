from pprint import pprint
from keras_model import *
from semval_data_process import *
import spacy

def test():
    with open('chp.txt', 'r') as fh:
        text = fh.read()

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    spans = list(doc.ents) + list(doc.noun_chunks)
    spans = spacy.util.filter_spans(spans)

    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)
    
    ent_indexes = []
    for ent in spans:
        if ent.text == "Harry":
            ent_indexes.append(ent.start)

    ent_sentences = []
    for i in ent_indexes:
        token_span = doc[i:i+1]
        sentence = token_span.sent
        ent_sentences.append(sentence)
    ent_sentences = list(dict.fromkeys(ent_sentences))

    print(ent_sentences[6].ents)
    print(ent_sentences[6].noun_chunks)
    # for i in range(9):g
    # es[i], i)


if __name__=='__main__':
    num_words = 20000
    embedding_size = 300
    max_len = 64   
    label_len = 19
    train_path = "semval/training/TRAIN_FILE.TXT"
    test_path = "semval/testing_keys/TEST_FILE_FULL.TXT"

    test()
