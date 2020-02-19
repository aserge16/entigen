from pprint import pprint
import json
import os
import numpy as np
import random
import torch
import spacy


def test():
    with open('../resources/chp.txt', 'r') as fh:
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
    print(ent_sentences[6])
    

    # for sen in ent_sentences:



if __name__=='__main__':
    test2('../MLMAN/checkpoint/MLMAN9225954.pth.tar')
    # test()
