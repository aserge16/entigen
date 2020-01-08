import io
import spacy
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage 


class TextProcess():
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path


    def extract_text_from_pdf(self):
        resource_manager = PDFResourceManager()
        text_stream = io.StringIO()
        converter = TextConverter(resource_manager, text_stream)
        page_interpreter = PDFPageInterpreter(resource_manager, converter)
    
        with open(self.pdf_path, 'rb') as fh:
            for page in PDFPage.get_pages(fh, 
                                        caching=True,
                                        check_extractable=True):
                page_interpreter.process_page(page)
    
            text = text_stream.getvalue()
    
        converter.close()
        text_stream.close()
    
        if text:
            self.text = text


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
        for i in self.ent_indexes:
            token_span = doc[i:i+1]
            sentence = token_span.sent
            self.ent_sentences.append(sentence)
        self.ent_sentences = list(dict.fromkeys(self.ent_sentences))
