from importlib import import_module
import numpy as np
import pandas as pd
import requests

import nlp_tools.files_setup

class nlp_processor:
    """Primary class of the library
    parameters:
        :data_path: str: filepath where all files will be created and stored. Will be created if doesn't already exist. Pass absolute directory.
        :metadata_addt_column_names: list: list of additional file names relevant for the analysis, e.g., publication date, title
    """
    
    def __init__(
        self,
        data_path,
        metadata_addt_column_names
    ):
        # initializing parameters
        self.data_path = data_path
        self.metadata_addt_column_names = metadata_addt_column_names
        
        # setting up directory structure
        self.files_setup = import_module("nlp_tools.files_setup")
        self.files_setup.setup_directories(self.data_path)
        
        # generating metadata file
        self.files_setup.generate_metadata_file(self.data_path, self.metadata_addt_column_names)
        self.metadata = pd.read_csv(f"{data_path}metadata.csv")
        
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
            
            tmp_metadata = self.files_setup.convert_to_text(self.metadata, self.data_path, text_id)
            
            # update the object's metadata
            if not(tmp_metadata is None):
                self.metadata = tmp_metadata
                self.metadata.to_csv(f"{self.data_path}metadata.csv", index = False)
