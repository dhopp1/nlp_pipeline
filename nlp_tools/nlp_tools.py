from importlib import import_module
import numpy as np
import pandas as pd

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