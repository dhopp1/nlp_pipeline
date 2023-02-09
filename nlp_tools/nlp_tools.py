from importlib import import_module
import pandas as pd

import nlp_tools.files_setup

class nlp_processor:
    """Primary class of the library
    parameters:
        :data_path: str: filepath where all files will be created and stored. Will be created if doesn't already exist. Pass absolute directory.
        :metadata_addt_column_names: list: list of additional file names relevant for the analysis, e.g., publication date, title
        :windows_tesseract_path: str: tesseract .exe file path in windows for OCR conversion
        :windows_poppler_path: str: poppler /bin file path in windows for OCR conversion
    """
    
    def __init__(
        self,
        data_path,
        metadata_addt_column_names,
        windows_tesseract_path = None, 
        windows_poppler_path = None
    ):
        # initializing parameters
        self.data_path = data_path
        self.metadata_addt_column_names = metadata_addt_column_names
        self.windows_tesseract_path = windows_tesseract_path
        self.windows_poppler_path = windows_poppler_path
        
        # setting up directory structure
        self.files_setup = import_module("nlp_tools.files_setup")
        self.files_setup.setup_directories(self.data_path)
        
        # generating metadata file
        self.files_setup.generate_metadata_file(self.data_path, self.metadata_addt_column_names)
        self.metadata = pd.read_csv(f"{data_path}metadata.csv")
        
        # making text transformations available
        self.text_transformation = import_module("nlp_tools.text_transformation")
        
    def refresh_metadata(self):
        "update the metadata of the processor in case changes are made to the file outside of the object"
        self.metadata = pd.read_csv(f"{self.data_path}metadata.csv")
        self.files_setup.generate_metadata_file(self.data_path, self.metadata_addt_column_names) # make sure text_id added
        
    def download_text_id(self, text_ids):
        "download a file from a URL and update the metadata file. Pass either single text id or list of them"
        if type(text_ids) != list:
            text_ids = [text_ids]
        counter = 1
        for text_id in text_ids:
            print(f"downloading file {counter}/{len(text_ids)}")
            counter += 1
            
            web_filepath = self.metadata.loc[lambda x: x.text_id == text_id, "web_filepath"].values[0]
            tmp_metadata = self.files_setup.download_document(self.metadata, self.data_path, text_id, web_filepath)
            
            # update the object's metadata
            if not(tmp_metadata is None):
                self.metadata = tmp_metadata
                self.metadata.to_csv(f"{self.data_path}metadata.csv", index = False)

    def convert_to_text(self, text_ids):
        "convert a pdf or html file to text. Pass either single text id or list of them"
        if type(text_ids) != list:
            text_ids = [text_ids]
        counter = 1
        for text_id in text_ids:
            print(f"converting to text: file {counter}/{len(text_ids)}")
            counter += 1
            
            tmp_metadata = self.files_setup.convert_to_text(self.metadata, self.data_path, text_id, self.windows_tesseract_path, self.windows_poppler_path)
            
            # update the object's metadata
            if not(tmp_metadata is None):
                self.metadata = tmp_metadata
                self.metadata.to_csv(f"{self.data_path}metadata.csv", index = False)

    def transform_text(
            self, 
            text_ids, 
            path_prefix,
            perform_lower = False,
            perform_replace_newline_period = False,
            perform_remove_punctuation = False,
            perform_remove_stopwords = False,
            perform_stemming = False,
            stopwords_language = "english",
            stemmer = "snowball",
            stemmer_language = "english"
    ):
        """Transforms texts in various ways and writes new text files to the transformed_txt_files/ directory
        parameters:
            :text_ids: list[float]: single text_id or list of them to perform the transformation(s) on
            :path_prefix: str: what to prefix the resulting .txt files with to label which transformations have been done
            :perform_lower: boolean: whether or not to lower case the text
            :perform_replace_newline_period: boolean: whether or not to replace new lines and periods with | so words and sentences can be identified in isolation
            :perform_remove_punctuation: boolean: whether or not to remove punctuation, except for |'s
            :perform_remove_stopwords: boolean: whether or not to remove stopwords
            :perform_stemming: boolean: whether or not to perform stemming
            :stopwords_language: str: if choosing to remove stopwords, desired language. Available ones here: https://stackoverflow.com/questions/54573853/nltk-available-languages-for-stopwords
            :stemmer: if choosing to stem, nltk stemmer. E.g., nltk.stem.snowball.SnowballStemmer("english"), or string of "snowball" or "lancaster" for one of these two in English
            :stemmer_language: str: if choosing to stem, desired language. Available languages for e.g. Snowball stemmer available with print(" ".join(SnowballStemmer.languages))
        """
        if type(text_ids) != list:
            text_ids = [text_ids]
        counter = 1
        for text_id in text_ids:
            print(f"transforming text: {counter}/{len(text_ids)}")
            text_path = self.metadata.loc[lambda x: x.text_id == text_id, "local_txt_filepath"].values[0]
            
            # only perform if text file exists
            if (".txt" in str(text_path)):
                # reading original text
                file = open(f"{text_path}", "r", encoding = "UTF-8") 
                stringx = file.read()
                file.close()
                
                if perform_lower:
                    stringx = self.text_transformation.lower(stringx)
                if perform_replace_newline_period:
                    stringx = self.text_transformation.replace_newline_period(stringx)
                if perform_remove_punctuation:
                    stringx = self.text_transformation.remove_punctuation(stringx)
                if perform_remove_stopwords:
                    stringx = self.text_transformation.remove_stopwords(stringx, stopwords_language)
                if perform_stemming:
                    stringx = self.text_transformation.stem(stringx, stemmer, stemmer_language)
                
                # write text file
                file = open(f"{self.data_path}transformed_txt_files/{path_prefix}{text_id}.txt", "wb+")
                file.write(stringx.encode())
                file.close()
                    
            