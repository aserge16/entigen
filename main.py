import io
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
 

def extract_text_from_pdf(pdf_path):
    resource_manager = PDFResourceManager()
    text_stream = io.StringIO()
    converter = TextConverter(resource_manager, text_stream)
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
 
    with open(pdf_path, 'rb') as fh:
        for page in PDFPage.get_pages(fh, 
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)
 
        text = text_stream.getvalue()
 
    converter.close()
    text_stream.close()
 
    if text:
        return text
