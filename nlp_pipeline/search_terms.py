import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import re
import itertools
import os

def gen_search_terms(processor, group_name, text_ids, search_terms_df, path_prefix, character_buffer = 100):
    "generate CSVs with information on search terms"
    
    if (isinstance(text_ids, int)):
        text_ids = [text_ids]
    
    # term count and occurrences CSV
    counts_columns = ["count", "count_per_1000_words", "number_of_texts_present", "share_of_texts_present", "sentence_sentiment", "character_buffer_sentiment"]
    counts = pd.DataFrame(columns = list(search_terms_df.columns) + counts_columns)
    
    pre_agg_counts_columns = ["text_id", "count", "n_words", "sentence_sentiment", "character_buffer_sentiment"]
    pre_agg_counts = pd.DataFrame(columns = list(search_terms_df.columns) + pre_agg_counts_columns)
   
    sentence_occurrences_columns = ["text_id", "sentence_context", "character_buffer_context", "sentence_sentiment", "character_buffer_sentiment"]
    sentence_occurrences = pd.DataFrame(columns = list(search_terms_df.columns) + sentence_occurrences_columns)
    
    for i in range(len(search_terms_df)):
        word = search_terms_df.iloc[i, -1]
        
        for text_id in text_ids:
            file_path = f"{processor.data_path}transformed_txt_files/{path_prefix}_{text_id}.txt"
            file = open(file_path, "r", encoding = "latin1")
            stringx = file.read()
            file.close()
            
            n_words = len([x for x in stringx.replace("|", "").split(" ") if len(x) > 0])
            
            split_string = stringx.split("|")
                
            occurrences = [(split_string[i-1] if i > 0 else "") + "|" + split_string[i] + "|" + (split_string[i+1] if i < len(split_string)-1 else "") for i in range(len(split_string)) if split_string[i].find(" " + word + " ") > -1]
            # when it happens multiple times in one sentence
            occurrences = [[[occurrences[i]] * len([m.start() for m in re.finditer(" " + word + " ", occurrences[i].split("|")[1])])] for i in range(len(occurrences))]
            occurrences = list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(occurrences))))
            # occurrences for character buffer sentiment
            indices = [m.start() for m in re.finditer(" " + word + " ", stringx)]
            
            if len(occurrences) > 0:
                # individual occurrences
                tmp_sentence_occurrences = pd.DataFrame({
                    "sentence_context": occurrences,
                    "character_buffer_context": [stringx[np.max([0, i - character_buffer]):np.min([len(stringx), i + character_buffer])] for i in indices],
                    "sentence_sentiment": [SentimentIntensityAnalyzer().polarity_scores(x.split("|")[1])["compound"] for x in occurrences],
                    "character_buffer_sentiment": [SentimentIntensityAnalyzer().polarity_scores(stringx[np.max([0, i - character_buffer]):np.min([len(stringx), i + character_buffer])])["compound"] for i in indices]
                })
                for j in range(len(search_terms_df.columns) - 1):
                    tmp_sentence_occurrences[search_terms_df.columns[j]] = search_terms_df.iloc[i, j]
                tmp_sentence_occurrences[search_terms_df.columns[-1]] = word
                tmp_sentence_occurrences["text_id"] = text_id
                
                tmp_sentence_occurrences = tmp_sentence_occurrences.loc[:, list(search_terms_df.columns) + sentence_occurrences_columns]
                
                # aggregate counts for this word
                tmp_counts = pd.DataFrame({
                    "count": len(occurrences),
                    "n_words": n_words,
                    "sentence_sentiment": np.mean(tmp_sentence_occurrences.sentence_sentiment),
                    "character_buffer_sentiment": np.mean(tmp_sentence_occurrences.character_buffer_sentiment)
                }, index = [0])
                
                for j in range(len(search_terms_df.columns) - 1):
                    tmp_counts[search_terms_df.columns[j]] = search_terms_df.iloc[i, j]
                tmp_counts[search_terms_df.columns[-1]] = word
                tmp_counts["text_id"] = text_id
                tmp_counts = tmp_counts.loc[:, pre_agg_counts.columns]
                
                sentence_occurrences = pd.concat([sentence_occurrences, tmp_sentence_occurrences], ignore_index = True, axis = 0)
                pre_agg_counts = pd.concat([pre_agg_counts, tmp_counts], ignore_index = True, axis = 0)
    
    # writing out sentence occurrences
    sentence_occurrences.to_csv(f"{processor.data_path}csv_outputs/search_terms_{group_name}_occurrences.csv", index = False)
    
    # writing out group counts
    for i in range(len(search_terms_df.columns)):
        agg_columns = list(search_terms_df.columns[:i+1])
        
        count = pre_agg_counts.groupby(agg_columns).apply(
            lambda s: pd.Series({
                "count": s["count"].sum(),
                "count_per_1000_words": s["count"].sum() / s["n_words"].sum() * 1000,
                "number_of_texts_present": s["text_id"].nunique(),
                "share_of_texts_present": s["text_id"].nunique() / len(text_ids),
                "sentence_sentiment": s["sentence_sentiment"].mean(),
                "character_buffer_sentiment": s["character_buffer_sentiment"].mean()
            })
        ).reset_index()
        
        count.to_csv(f"{processor.data_path}csv_outputs/search_terms_{group_name}_counts_by_{search_terms_df.columns[i]}.csv", index = False)
        
        
def gen_aggregated_search_terms(processor, group_names, text_ids, search_terms_df, path_prefix, character_buffer = 100):
    "aggregate search terms by group"
    for i in range(len(group_names)):
        # generate data
        gen_search_terms(processor, f"temp_helper_{group_names[i]}", text_ids[i], search_terms_df, path_prefix, character_buffer = 100)
    
    # aggregate it
    for j in range(len(search_terms.columns)):
        for i in range(len(group_names)):
            tmp_df = pd.read_csv(f"{processor.data_path}csv_outputs/search_terms_temp_helper_{group_names[i]}_counts_by_{search_terms.columns[j]}.csv")
            tmp_df["group"] = group_names[i]
            if i == 0:
                df = tmp_df.copy()
            else:
                df = pd.concat([df, tmp_df], ignore_index = True, axis = 0)
    
        df = df.loc[:, ["group"] + list(df.columns[:-1])]
        df.to_csv(f"{processor.data_path}csv_outputs/search_terms_grouped_by_{search_terms.columns[j]}.csv", index = False)
    
    # delete temporary files
    for i in range(len(group_names)):
        os.remove(f"{processor.data_path}csv_outputs/search_terms_temp_helper_{group_names[i]}_occurrences.csv")
        for j in range(len(search_terms.columns)):
            os.remove(f"{processor.data_path}csv_outputs/search_terms_temp_helper_{group_names[i]}_counts_by_{search_terms.columns[j]}.csv")