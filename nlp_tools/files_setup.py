import pandas as pd
import os

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
            columns = ["text_id", "web_filepath", "local_raw_filepath", "local_txt_filepath"] + metadata_addt_column_names   
        )
        metadata.to_csv(f"{data_path}metadata.csv", index=False)
    else:
        metadata = pd.read_csv(f"{data_path}metadata.csv")
        metadata.text_id = range(1, len(metadata) + 1) # ensure creation of a unique text id field
    return metadata