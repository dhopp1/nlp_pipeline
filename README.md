# nlp_pipeline
Collection of NLP tools for processing and analyzing text data.

## Introduction
The fundamental input of the library is a metadata file. By default this will contain the columns `["text_id", "web_filepath", "local_raw_filepath", "local_txt_filepath", "detected_language"]`. The only one that needs to be provided by the user is `"web_filepath"`. Ergo, a list of URLs containing text documents is all that is required to use the library. If the corpus had additional columns of interest, like titles, etc., those can be passed via the `metadata_addt_column_names` argument when instantiating the initial `nlp_processor` function. More information below.

Fundamentally the library takes the list of documents and downloads, transforms, and organizes them according to a specific filestructure. These files can then be used to generate insights, such as word counts, etc.

## Example code
```py
from nlp_pipeline.nlp_pipeline import nlp_processor

# additional columns I want to track in the metadata
metadata_addt_column_names = ["title", "year"] 

# instantiating the processor object
processor = nlp_processor(
	data_path = "path_to_store_text_documents/",
	metadata_addt_column_names = metadata_addt_column_names,
	windows_tesseract_path = "path_to_tesseract.exe", # if on Windows, otherwise leave blank and have it installed in your path
	windows_poppler_path = "path_to_poppler/bin" # if on Windows, otherwise leave blank and have it installed in your path
)

# this will generate a metadata file and create the directory structure
# you can now add additional data to the metadata file, (titles, etc.). When finished, run the following so the metadata in the processor object will reflect the local file
processor.refresh_object_metadata()

# if you ever make changes to the local files, e.g., delete a PDF, run the following to make sure the metadata file reflects that
processor.sync_local_metadata()

# download some documents with metadat IDs 1, 2, and 3
text_ids = [1,2,3]
processor.download_text_id(text_ids)

# convert the PDFs or HTMLs to .txt
processor.convert_to_text(text_ids)

# transform the text (stemming, etc.)
# run help(processor.transform_text) for more information
processor.transform_text(
        text_ids = text_ids,
        path_prefix = "all_transformed", # what to prefix the files with this transformation
        perform_lower = True, # lower case the text
        perform_replace_newline_period = True, # replace periods and newline characters with |
        perform_remove_punctuation = True, # remove punctuation marks
        perform_remove_stopwords = True, # remove stopwords (the, and, etc.)
        perform_stemming = True, # stem the words (run = runs, etc.)
        stemmer = "snowball" # which stemmer to use. If in doubt, use snowball
)

# from the transformed text, generate a CSV with word counts in each document
processor.gen_word_count_csv(
        text_ids = text_ids, 
        path_prefix = "all_transformed", # prefix used previously for the transformation
        exclude_words = ["for"] # list of words to not include in the word counts
)

# get sentiment of a group of texts
processor.gen_sentiment_csv(text_ids, "all_transformed")

# get n_words, sentences, and pages of texts
processor.gen_summary_stats_csv(text_ids, "all_transformed")

# plot of word counts
p, plot_df = processor.plot_word_occurrences(
    text_ids_list = text_ids, # can be a list of lists, [[1,2,3], [4,5,6]], for counts by decade e.g.
    word = "green", 
    path_prefix = "all_transformed", 
    x_labels = [1,2,3],
    title = "Plot Title"
)

# plot average sentiment or neutral proportion in documents
p, plot_df = processor.plot_sentiment(
    text_ids_list = text_ids, 
    path_prefix = "all_transformed", 
    x_labels = [1,2,3],
    sentiment_col = "neutral_proportion",
    title = "Plot Title"
)
```