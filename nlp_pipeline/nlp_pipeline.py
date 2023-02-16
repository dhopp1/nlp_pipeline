from importlib import import_module
import pandas as pd
import os

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
        self.files_setup = import_module("nlp_pipeline.files_setup")
        self.files_setup.setup_directories(self.data_path)
        
        # generating metadata file
        self.files_setup.generate_metadata_file(self.data_path, self.metadata_addt_column_names)
        self.metadata = pd.read_csv(f"{data_path}metadata.csv")
        
        # making text transformations available
        self.text_transformation = import_module("nlp_pipeline.text_transformation")
        
        # making visualizations available
        self.visualizations = import_module("nlp_pipeline.visualizations")
        
    def refresh_object_metadata(self):
        "update the metadata of the processor in case changes are made to the file outside of the object"
        self.metadata = pd.read_csv(f"{self.data_path}metadata.csv")
        self.files_setup.generate_metadata_file(self.data_path, self.metadata_addt_column_names) # make sure text_id added
    
    def sync_local_metadata(self):
        "update the metadata local file to reflect actual state of files in the data directory"
        self.metadata = self.files_setup.refresh_local_metadata(self.metadata, self.data_path)
        self.metadata.to_csv(f"{self.data_path}metadata.csv", index = False)
        
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
            stemmer = "snowball"
    ):
        """Transforms texts in various ways and writes new text files to the transformed_txt_files/ directory
        parameters:
            :text_ids: list[float]: single text_id or list of them to perform the transformation(s) on
            :path_prefix: str: what to prefix the resulting .txt files with to label which transformations have been done. A "_" will be appended to the end of it.
            :perform_lower: boolean: whether or not to lower case the text
            :perform_replace_newline_period: boolean: whether or not to replace new lines and periods with | so words and sentences can be identified in isolation
            :perform_remove_punctuation: boolean: whether or not to remove punctuation, except for |'s
            :perform_remove_stopwords: boolean: whether or not to remove stopwords
            :perform_stemming: boolean: whether or not to perform stemming
            :stemmer: if choosing to stem, nltk stemmer. E.g., nltk.stem.snowball.SnowballStemmer("english"), or string of "snowball" or "lancaster" for one of these. Lancaster only works in english
        """
        path_prefix += "_"
        
        if type(text_ids) != list:
            text_ids = [text_ids]
        counter = 1
        for text_id in text_ids:
            print(f"transforming text: {counter}/{len(text_ids)}")
            counter += 1
            
            text_path = self.metadata.loc[lambda x: x.text_id == text_id, "local_txt_filepath"].values[0]
            language = self.metadata.loc[lambda x: x.text_id == text_id, "detected_language"].values[0]
            if (str(language) == "") | (str(language) == "nan"): # if no language, default to english
                language = "en"
            
            # only perform if text file exists and transformed file doesn't already exist
            if (".txt" in str(text_path)) & (not(os.path.exists(f"{self.data_path}transformed_txt_files/{path_prefix}{text_id}.txt"))):
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
                    stringx = self.text_transformation.remove_stopwords(stringx, language)
                if perform_stemming:
                    stringx = self.text_transformation.stem(stringx, stemmer, language)
                
                # write text file
                file = open(f"{self.data_path}transformed_txt_files/{path_prefix}{text_id}.txt", "wb+")
                file.write(stringx.encode())
                file.close()
                    
    def gen_word_count_csv(self, text_ids, path_prefix, exclude_words):
        """Gets word counts from files in transformed_txt_files/ directory and writes them to CSV
        parameters:
            :text_ids: list[float]: single text_id or list of them to perform the transformation(s) on
            :path_prefix: str: what the prefix of the files in the transformed_txt_files/ path is
        """
        path_prefix += "_"
        
        if type(text_ids) != list:
            text_ids = [text_ids]
        
        # check if CSV already exists
        csv_path = f"{self.data_path}csv_outputs/{path_prefix}word_counts.csv"
        
        if os.path.exists(csv_path):
            csv = pd.read_csv(csv_path)
        else:
            csv = pd.DataFrame({
                "text_id": self.metadata.text_id.values,
                "word_count_dict": ""
            })
        
        counter = 1
        for text_id in text_ids:
            print(f"creating word count dictionary: {counter}/{len(text_ids)}")
            counter += 1
            
            text_path = f"{self.data_path}transformed_txt_files/{path_prefix}{text_id}.txt"
            
            # only perform if text file exists and hasn't already been run
            prior_value = csv.loc[csv.text_id == text_id, "word_count_dict"].values[0]
            if os.path.exists(text_path) & ((str(prior_value) == "") | (str(prior_value) == "nan")):
                # reading original text
                file = open(f"{text_path}", "r", encoding = "UTF-8") 
                stringx = file.read()
                file.close()
                
                word_dict = self.text_transformation.gen_word_count_dict(stringx, exclude_words)
                
                # adding and writing to CSV
                csv.loc[csv.text_id == text_id, "word_count_dict"] = str(word_dict)
                csv.to_csv(csv_path, index = False)
            
    def bar_plot_word_count(self, text_ids, path_prefix, n_words=10, title=""):
        """bar plot of top words occurring in text(s)
        parameters:
            :text_ids: list[float]: single text_id or list of them to perform the transformation(s) on
            :path_prefix: str: what the prefix of the files in the transformed_txt_files/ path is
            :n_words: int: top n words to show in the plot
            :title: str: title of the plot if desired
        """
        path_prefix += "_"
        
        if type(text_ids) != list:
            text_ids = [text_ids]
        
        # only do if dataframe exists
        csv_path = f"{self.data_path}csv_outputs/{path_prefix}word_counts.csv"
        if os.path.exists(csv_path):
            csv = pd.read_csv(csv_path).loc[lambda x: x.text_id.isin(text_ids), :].reset_index(drop=True)
            df = self.visualizations.convert_word_count_dict_to_df(csv)
            p, plot_data = self.visualizations.bar_plot_word_count(df, n_words, title)
            
            return (p, plot_data)
        
    def word_cloud(self, text_ids, path_prefix, n_words=10):
        """word cloud of top words occurring in text(s)
        parameters:
            :text_ids: list[float]: single text_id or list of them to perform the transformation(s) on
            :path_prefix: str: what the prefix of the files in the transformed_txt_files/ path is
            :n_words: int: top n words to show in the plot
        """
        if type(text_ids) != list:
            text_ids = [text_ids]
        
        # only do if dataframe exists
        csv_path = f"{self.data_path}csv_outputs/{path_prefix}word_counts.csv"
        if os.path.exists(csv_path):
            csv = pd.read_csv(csv_path).loc[lambda x: x.text_id.isin(text_ids), :].reset_index(drop=True)
            df = self.visualizations.convert_word_count_dict_to_df(csv)
            p, plot_data = self.visualizations.word_cloud(df, n_words)
            
            return (p, plot_data)
        
    def gen_sentiment_csv(self, text_ids, path_prefix):
        """generating average sentiment of documents. A higher score is more positive, lower is more negative. Depends on sentences being delimited by |
        parameters:
            :text_ids: list[float]: single text_id or list of them to perform the transformation(s) on
            :path_prefix: str: what the prefix of the files in the transformed_txt_files/ path is
        output:
            :pd.DataFrame: with columns:
                :text_id: text ids
                :avg_sentiment_w_neutral: average sentiment score with 0.0 neutral sentences
                :avg_sentiment_wo_neutral: average sentiment score excluding 0.0 neutral sentences
                :neutral_proportion: % of sentences in the document with a 0.0 sentiment score
        """
        path_prefix += "_"
        
        if type(text_ids) != list:
            text_ids = [text_ids]
    
        # check if CSV already exists
        csv_path = f"{self.data_path}csv_outputs/{path_prefix}sentiments.csv"
        
        if os.path.exists(csv_path):
            csv = pd.read_csv(csv_path)
        else:
            csv = pd.DataFrame({
                "text_id": self.metadata.text_id.values,
                "avg_sentiment_w_neutral": "",
                "avg_sentiment_wo_neutral": "",
                "neutral_proportion": ""
            })
        
        counter = 1
        for text_id in text_ids:
            print(f"getting sentiments: {counter}/{len(text_ids)}")
            counter += 1
            
            txt_path = f"{self.data_path}transformed_txt_files/{path_prefix}{text_id}.txt"
            
            # only do if txt path exists and hasn't already been run
            prior_value = csv.loc[csv.text_id == text_id, "avg_sentiment_w_neutral"].values[0]
            if os.path.exists(txt_path) & ((str(prior_value) == "") | (str(prior_value) == "nan")):
                # reading original text
                file = open(f"{txt_path}", "r", encoding = "UTF-8") 
                stringx = file.read()
                file.close()
                
                # calculating sentiments
                sentiments = [self.text_transformation.get_single_sentiment(x)["compound"] for x in stringx.split("|") if (len(x) > 3) & (len(x.split(" ")) > 2) & (not(x.isnumeric()))] # only do sentiment for sentences with more than 2 words and not numeric
                
                # adding and writing to CSV
                csv.loc[csv.text_id == text_id, "avg_sentiment_w_neutral"] = sum(sentiments) / len(sentiments)
                csv.loc[csv.text_id == text_id, "avg_sentiment_wo_neutral"] = sum([x for x in sentiments if x != 0.0]) / len([x for x in sentiments if x != 0.0])
                csv.loc[csv.text_id == text_id, "neutral_proportion"] = len([x for x in sentiments if x == 0.0]) / len(sentiments)
                csv.to_csv(csv_path, index = False)
                
                
    def plot_word_occurrences(self, text_ids_list, word, path_prefix, x_labels = None, title = ""):
        """get plot of occurrences of a particular word over groups of documents. Searches for contains rather than exact matches.
        parameters:
            :text_ids_list: list[int]: either list of list of text_ids, e.g., [[1,2], [3,4]], or if individual documents rather than groups, [1,2,3,4]
            :word: str: which word to look for
            :path_prefix: str: what the prefix of the files in the csv_outputs/ path is. Need to have run the gen_word_count_csv() function.
            :x_labels: list: what to label the x-axis in the plot, what are the different documents or groups of documents. E.g., decades or years.
            :title: str: additional title to the plot
        """
        path_prefix += "_"
        
        csv_path = f"{self.data_path}csv_outputs/{path_prefix}word_counts.csv"
        
        # only run if file exists
        if os.path.exists(csv_path):
            csv = pd.read_csv(csv_path)
            p, plot_df = self.visualizations.plot_word_occurrences(csv, text_ids_list, word, x_labels, title)
            
            return (p, plot_df)
        
        
    def plot_sentiment(self, text_ids_list, path_prefix, x_labels = None, title = "", sentiment_col = "avg_sentiment_wo_neutral"):
        """get plot of sentiment of a particular word over groups of documents. Searches for contains rather than exact matches.
        parameters:
            :text_ids_list: list[int]: either list of list of text_ids, e.g., [[1,2], [3,4]], or if individual documents rather than groups, [1,2,3,4]
            :path_prefix: str: what the prefix of the files in the csv_outputs/ path is. Need to have run the get_sentiment_csv() function.
            :x_labels: list: what to label the x-axis in the plot, what are the different documents or groups of documents. E.g., decades or years.
            :title: str: additional title to the plot
            :sentiment_col: str: which column in the sentiment df to plot, one of ["avg_sentiment_wo_neutral", "avg_sentiment_w_neutral", "neutral_proportion"]
        """
        path_prefix += "_"
        
        csv_path = f"{self.data_path}csv_outputs/{path_prefix}sentiments.csv"
        
        # only run if file exists
        if os.path.exists(csv_path):
            csv = pd.read_csv(csv_path)
            p, plot_df = self.visualizations.plot_sentiment(csv, text_ids_list, x_labels, title, sentiment_col)
            return (p, plot_df)
        
    
    def gen_summary_stats_csv(self, text_ids, path_prefix):
        """generating various summary statistics: number of words, unique words, sentences, pages, etc. per document. Depends on sentences being delimited by |
        parameters:
            :text_ids: list[float]: single text_id or list of them to perform the transformation(s) on
            :path_prefix: str: what the prefix of the files in the transformed_txt_files/ path is
        output:
            :pd.DataFrame: with columns:
                :text_id: text ids
                :n_words: number of total words in the transformed document
                :n_unique_words: number of unique words in the transformed document
                :n_sentences: number of sentences in the transformed document
                :n_pages: number of pages in the original document
                :avg_word_length: average word length (number of letters) in the original document
                :avg_word_incidence: average incidence of words in the language of the document (higher is more frequent/common words, a word with Zipf value 6 appears once per thousand words and a word with Zipf value 3 appears once per million words)
        """
        path_prefix += "_"
        
        if type(text_ids) != list:
            text_ids = [text_ids]
    
        # check if CSV already exists
        csv_path = f"{self.data_path}csv_outputs/{path_prefix}summary_stats_csv.csv"
        
        if os.path.exists(csv_path):
            csv = pd.read_csv(csv_path)
        else:
            csv = pd.DataFrame({
                "text_id": self.metadata.text_id.values,
                "n_words": "",
                "n_unique_words": "",
                "n_sentences": "",
                "n_pages": "",
                "avg_word_length": "",
                "avg_word_incidence": ""
            })
        
        counter = 1
        for text_id in text_ids:
            print(f"getting word and sentence count: {counter}/{len(text_ids)}")
            counter += 1
            
            txt_path = f"{self.data_path}transformed_txt_files/{path_prefix}{text_id}.txt"
            orig_txt_path = f"{self.data_path}txt_files/{text_id}.txt" # for n_pages
            language = self.metadata.loc[lambda x: x.text_id == text_id, "detected_language"].values[0]
            
            # only do if txt path exists and hasn't already been run
            prior_value = csv.loc[csv.text_id == text_id, "n_words"].values[0]
            if os.path.exists(txt_path) & ((str(prior_value) == "") | (str(prior_value) == "nan")):
                # reading transformed text
                file = open(f"{txt_path}", "r", encoding = "UTF-8") 
                stringx = file.read()
                file.close()
                
                # reading original text for n_pages
                file = open(f"{orig_txt_path}", "r", encoding = "UTF-8") 
                orig_stringx = file.read()
                file.close()
                
                # calculating n_words
                n_words = len([x for x in stringx.split(" ") if len(x) > 1]) # minimum word length
                
                # calculating n_unique_words
                n_unique_words = len(set([x for x in stringx.split(" ") if len(x) > 1]))
                
                # calculating n_sentences
                n_sentences = len([x for x in stringx.split("|") if len(x) > 2]) # minimum sentence length
                
                # calculating n_pages
                n_pages = orig_stringx.count("[newpage]")
                
                # calculating avg_word_length
                avg_word_length = [len(x) for x in orig_stringx.split(" ") if len(x) > 1]
                avg_word_length = sum(avg_word_length) / len(avg_word_length)
                
                # calculating 
                word_list = [x for x in orig_stringx.split(" ") if len(x) > 1]
                avg_word_incidence = [self.text_transformation.get_word_frequency(x, language) for x in word_list]
                avg_word_incidence = sum(avg_word_incidence) / len(avg_word_incidence)
                
                # adding and writing to CSV
                csv.loc[csv.text_id == text_id, "n_words"] = n_words
                csv.loc[csv.text_id == text_id, "n_unique_words"] = n_unique_words
                csv.loc[csv.text_id == text_id, "n_sentences"] = n_sentences
                csv.loc[csv.text_id == text_id, "n_pages"] = n_pages
                csv.loc[csv.text_id == text_id, "avg_word_length"] = avg_word_length
                csv.loc[csv.text_id == text_id, "avg_word_incidence"] = avg_word_incidence
                csv.to_csv(csv_path, index = False)