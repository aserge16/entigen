import io
import itertools
import spacy
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage 


class TextProcess():
    def __init__(self, text_path, PDF=False):
        self.text_path = text_path
        if PDF:
            self.extract_text_from_pdf()
        else:
            self.extract_text_from_doc()


    def extract_text_from_pdf(self):
        resource_manager = PDFResourceManager()
        text_stream = io.StringIO()
        converter = TextConverter(resource_manager, text_stream)
        page_interpreter = PDFPageInterpreter(resource_manager, converter)
    
        with open(self.text_path, 'rb') as fh:
            for page in PDFPage.get_pages(fh, 
                                        caching=True,
                                        check_extractable=True):
                page_interpreter.process_page(page)
    
            text = text_stream.getvalue()
    
        text_stream.close()
        converter.close()
    
        if text:
            self.text = text


    def extract_text_from_doc(self):
        with open(self.text_path, 'r') as fh:
            self.text = fh.read()


    def ie_preprocess(self):
        nlp = spacy.load("en_core_web_sm")
        self.doc = nlp(self.text)

        spans = list(self.doc.ents) + list(self.doc.noun_chunks)
        self.spans = spacy.util.filter_spans(spans)

        with self.doc.retokenize() as retokenizer:
            for span in self.spans:
                retokenizer.merge(span)


    def ent_preprocess(self, request):
        self.ent_indexes = []
        for ent in self.spans:
            if ent.text == request:
                self.ent_indexes.append(ent.start)

        self.ent_sentences = []
        count = 20000
        for i in self.ent_indexes:
            span = self.doc[i:i+1].sent
            ents = []
            if len(span.ents) > 1:
                for e in span.ents:
                    ents.append(e.text.replace('\n', ' '))

            sentence = span.text.replace('\n', ' ')

            combinations = list(itertools.combinations(range(len(ents)), 2))
            for i, j in combinations:
                if (ents[i] == request) or (ents[j] == request):
                    if ents[i] != ents[j]:
                        temp = sentence
                        temp = str(count) + "\t\"" + temp + "\""
                        temp = temp.replace(ents[i], "<e1>" + ents[i] + "</e1>", 1)
                        temp = temp.replace(ents[j], "<e2>" + ents[j] + "</e2>", 1)
                        temp += "\nOther\n\n\n"

                        self.ent_sentences.append(temp)
                        count += 1

        self.ent_sentences = list(dict.fromkeys(self.ent_sentences))
        
        with open("./temp_sentences.txt", 'w+') as fh:
            for s in self.ent_sentences:
                fh.write(s)
