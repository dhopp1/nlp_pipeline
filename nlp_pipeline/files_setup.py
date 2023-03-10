import pandas as pd
import requests
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import pytesseract
import platform
from pdf2image import convert_from_path, pdfinfo_from_path
from pathlib import Path
from PIL import Image
from langdetect import detect
import os, glob

def setup_directories(data_path):
    "setup requied and standardized directory structure"
    
    # creating base path
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
        
    # path for raw documents, PDFs, HTML, etc.
    if not os.path.isdir(f"{data_path}raw_files/"):
        os.makedirs(f"{data_path}raw_files/")
    
    # path for .txt documents
    if not os.path.isdir(f"{data_path}txt_files/"):
        os.makedirs(f"{data_path}txt_files/")
        
    # path for potential transformed .txt files, e.g., stemmed, stopwords removed, etc.
    if not os.path.isdir(f"{data_path}transformed_txt_files/"):
        os.makedirs(f"{data_path}transformed_txt_files/")
        
    # path for transformed data CSVs
    if not os.path.isdir(f"{data_path}csv_outputs/"):
        os.makedirs(f"{data_path}csv_outputs/")
        
    # path for reports, plots, and outputs
    if not os.path.isdir(f"{data_path}visual_outputs/"):
        os.makedirs(f"{data_path}visual_outputs/")
    

def generate_metadata_file(data_path, metadata_addt_column_names):
    "generate an initial metadata file. metadata_addt_column_names is list of additional columns necessary for the analysis (publication date, etc.)"
    
    # only create the file if it doesn't exist already
    if not(os.path.isfile(f"{data_path}metadata.csv")):
        metadata = pd.DataFrame(
            columns = ["text_id", "web_filepath", "local_raw_filepath", "local_txt_filepath", "detected_language"] + metadata_addt_column_names   
        )
        metadata.to_csv(f"{data_path}metadata.csv", index=False)
    else:
        metadata = pd.read_csv(f"{data_path}metadata.csv")
        metadata.text_id = range(1, len(metadata) + 1) # ensure creation of a unique text id field
    return metadata


def download_document(metadata, data_path, text_id, web_filepath):
    "download a file from a URL and update the metadata file"
    
    if str(web_filepath) == "" or str(web_filepath) == "nan":
        return None
    else:
        web_filepath = web_filepath.split(",")[0] # may have multiple URLs stored in field, take only first (english)
        
        # first check if this file already downloaded
        if not(os.path.isfile(f"{data_path}raw_files/{text_id}.html")) and not(os.path.isfile(f"{data_path}raw_files/{text_id}.pdf")):
            response = requests.get(web_filepath)
            content_type = response.headers.get('content-type')
            
            if "application/pdf" in content_type:
                ext = ".pdf"
            elif "text/html" in content_type:
                ext = ".html"
            else:
                ext = ""
            
            if ext != "":
                file = open(f"{data_path}raw_files/{text_id}{ext}", "wb+")
                file.write(response.content)
                file.close()
                
                metadata.loc[metadata.text_id == text_id, "local_raw_filepath"] = f"{data_path}raw_files/{text_id}{ext}"
                return metadata
            else:
                return None
            
        
def parse_pdf(pdf_path):
    "parse a pdf and return text string"
    reader = PdfReader(pdf_path)
    return_text = ""
    if any(["OceScanCompression" in key for key in reader.metadata.keys()]):
        return_text = "[newpage] "
    else:
        for i in range(len(reader.pages)):
            return_text += "[newpage] " + reader.pages[i].extract_text()
        
    return return_text

def parse_ocr_pdf(data_path, pdf_path, windows_tesseract_path = None, windows_poppler_path = None):
    """convert a scanned PDF to text and return text string. From https://www.geeksforgeeks.org/python-reading-contents-of-pdf-using-ocr-optical-character-recognition/
    parameters:
        :data_path: str: filepath where all files will be created and stored
        :pdf_path: str: path of the pdf file
        :windows_tesseract_path: str: path of tesseract .exe file (https://linuxhint.com/install-tesseract-windows/)
        :windows_poppler_path: str: path of poppler .exe file (https://blog.alivate.com.au/poppler-windows/)
    """
    if platform.system() == "Windows":
        pytesseract.pytesseract.tesseract_cmd = (
            windows_tesseract_path
        )
    # storing image files temporarily
    PDF_file = Path(pdf_path)
    image_file_list = []
    
    if platform.system() == "Windows":
        info = pdfinfo_from_path(pdf_path, userpw=None, poppler_path=windows_poppler_path)
        max_pages = info["Pages"]
        pdf_pages = []
        for i in range(1, max_pages + 1):
            print(f"OCR conversion (stage 1/3) of {pdf_path}, page {i} / {max_pages}")
            tmp_pdf_pages = convert_from_path(
                PDF_file, 500, poppler_path = windows_poppler_path, first_page = i, last_page = i
            )
            pdf_pages.append(tmp_pdf_pages[0])
    else:
        info = pdfinfo_from_path(pdf_path, userpw=None)
        max_pages = info["Pages"]
        pdf_pages = []
        for i in range(1, max_pages + 1):
            print(f"OCR conversion (stage 1/3) of {pdf_path}, page {i} / {max_pages}")
            tmp_pdf_pages = convert_from_path(
                PDF_file, 500, first_page = i, last_page = i
            )
            pdf_pages.append(tmp_pdf_pages[0])
        
    # iterate through pages
    counter = 1
    for page_enumeration, page in enumerate(pdf_pages, start=1):
        print(f"OCR conversion (stage 2/3) of {pdf_path}, page {counter} / {max_pages}")
        counter += 1
        
        # Create a file name to store the image
        filename = f"{data_path}\page_{page_enumeration:03}.jpg"
 
        # Save the image of the page in system
        page.save(filename, "JPEG")
        image_file_list.append(filename)
        
    # Iterate from 1 to total number of pages
    return_text = ""
    counter = 1
    for image_file in image_file_list:
        print(f"OCR conversion (stage 3/3) of {pdf_path}, page {counter} / {len(image_file_list)}")
        counter += 1
        
        text = str(((pytesseract.image_to_string(Image.open(image_file)))))
        text = text.replace("-\n", "")
        return_text += "[newpage] " + text
    
    return return_text
        

def parse_html(html_path):
    "parse an html file and return text string"
    file = open(html_path, "r", encoding = "UTF-8")
    html = file.read()
    soup = BeautifulSoup(html, features="html.parser")
    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out
    
    # get text
    return_text = soup.get_text()
    
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in return_text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    return_text = '\n'.join(chunk for chunk in chunks if chunk)
    
    return return_text


def detect_language(stringx):
    "determine the language of a string"
    return detect(stringx)
    
        
def convert_to_text(metadata, data_path, text_id, windows_tesseract_path = None, windows_poppler_path = None):
    "convert a PDF or HTML file into raw text. In PDFs, new pages encoded with '[newpage]'"
    raw_path = metadata.loc[lambda x: x.text_id == text_id, "local_raw_filepath"].values[0]
    # path exists (not nan) and is longer than 0
    raw_exists = type(raw_path) == str
    if raw_exists:
         raw_exists = raw_exists & (len(raw_path) > 0)
    
    if raw_exists:
        # first check if this file already converted
        if not(os.path.isfile(f"{data_path}txt_files/{text_id}.txt")):
            # pdf file
            if ".pdf" in raw_path:
                try:
                    return_text = parse_pdf(raw_path)
                except:
                    return_text = ""
                try:
                    if (len(set(return_text.split("[newpage] "))) == 1) | ((return_text.lower().count("/g") / len(return_text)) > 0.01) | (return_text.lower().count("_") / len(return_text) > 0.05) | (return_text.lower().count("sqj") > 10): # if only empties, scan, needs to be OCR converted. If bunch of "/G"s, greater than 1% of all the characters, encoding error, like review of maritime transport 2006. Or if poorly digitized and a lot of 'sqj's. Force OCR.
                        return_text = parse_ocr_pdf(data_path, raw_path, windows_tesseract_path, windows_poppler_path)
                        # remove temporary image files from OCR
                        for f in glob.glob(f"{data_path}*.jpg"):
                            os.remove(f)
                except:
                    return_text = ""
            elif ".html" in raw_path:
                return_text = parse_html(raw_path)
            
            # write text file
            file = open(f"{data_path}txt_files/{text_id}.txt", "wb+")
            file.write(return_text.encode('utf-8', 'replace'))
            file.close()
            
            # update metadata file
            metadata.loc[lambda x: x.text_id == text_id, "local_txt_filepath"] = f"{data_path}txt_files/{text_id}.txt"
            try:
                detected_lang = detect(return_text)
            except:
                detected_lang = "en"
            metadata.loc[lambda x: x.text_id == text_id, "detected_language"] = detected_lang
            final_return = metadata
        else:
            final_return = None
    else:
        final_return = None
        
    return final_return


def refresh_local_metadata(metadata, data_path):
    """update the metadata to reflect the situation in the data path in terms of files/conversions done
    parameters:
        :metadata: pd.DataFrame: metadata dataframe of the object
        :data_path: str: filepath where all files are stored
    output:
        will update metadata for correct situation in terms of:
            -raw_files/
            -txt_files/
        and update downstream files (so that can't have a transformed txt or sentiment without the raw pdf, etc.),  in:
            -transformed_txt_files/
            -csv_outputs/
    """
    # raw path
    print(f"Syncing metadata to local file situation: checking raw paths...")
    def check_raw_exists(text_id):
        files = glob.glob(f"{data_path}raw_files/{text_id}.*")
        return files[0] if len(files) == 1 else ""
    raw_paths = [check_raw_exists(text_id) for text_id in metadata.text_id.values]
    metadata["local_raw_filepath"] = raw_paths
    
    # txt path
    print(f"Syncing metadata to local file situation: checking txt paths...")
    def check_txt_exists(text_id):
        txt_path = f"{data_path}txt_files/{text_id}.txt"
        return txt_path if os.path.exists(txt_path) else ""
    txt_paths = [check_txt_exists(text_id) for text_id in metadata.text_id.values]
    metadata["local_txt_filepath"] = txt_paths
    
    # remove transformed_txt/ files if there's not raw/txt path (old/stale file)
    print(f"Syncing metadata to local file situation: checking transformed_txt paths...")
    def remove_transformed_files(text_id, txt_path):
        files = glob.glob(f"{data_path}transformed_txt_files/*_{text_id}.txt")
        if len(files) > 0:
            if (str(txt_path) == "") | (str(txt_path) == "nan"):
                for file in files:
                    os.remove(file)
    [remove_transformed_files(text_id, txt_path) for text_id, txt_path in dict(zip(metadata.text_id.values, txt_paths)).items()]
    
    # csv_outputs/ files. Remove entries in CSVs where no raw/txt path exists
    print(f"Syncing metadata to local file situation: checking CSV outputs...")
    def remove_csv_outputs(text_id, txt_path, data):
        if (str(txt_path) == "") | (str(txt_path) == "nan"): # if no txt file, blank out all values for that in the csv
            data.loc[data.text_id == text_id, data.columns[data.columns != "text_id"]] = ""
        return data
    
    csv_files = glob.glob(f"{data_path}csv_outputs/*.csv")
    counter = 1
    for file in csv_files:
        print(f"Syncing metadata to local file situation: checking CSV outputs: {counter}/{len(csv_files)}")
        counter += 1
        
        data = pd.read_csv(file)
        for text_id, txt_path in dict(zip(metadata.text_id.values, txt_paths)).items():
            data = remove_csv_outputs(text_id, txt_path, data)
        data.to_csv(file, index = False)
    
    print(f"Syncing metadata to local file situation: Success!")
    
    return metadata