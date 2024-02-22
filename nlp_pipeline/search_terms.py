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
        print(f"processing search terms for group {group_name}: {i+1}/{len(search_terms_df)}")
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
        print(f"aggregated_search: processing search terms for group {i+1}/{len(group_names)}")
        # generate data
        gen_search_terms(processor, f"temp_helper_{group_names[i]}", text_ids[i], search_terms_df, path_prefix, character_buffer = 100)
    
    # aggregate it
    for j in range(len(search_terms_df.columns)):
        for i in range(len(group_names)):
            tmp_df = pd.read_csv(f"{processor.data_path}csv_outputs/search_terms_temp_helper_{group_names[i]}_counts_by_{search_terms_df.columns[j]}.csv")
            tmp_df["group"] = group_names[i]
            if i == 0:
                df = tmp_df.copy()
            else:
                df = pd.concat([df, tmp_df], ignore_index = True, axis = 0)
    
        df = df.loc[:, ["group"] + list(df.columns[:-1])]
        df.to_csv(f"{processor.data_path}csv_outputs/search_terms_grouped_by_{search_terms_df.columns[j]}.csv", index = False)
    
    # delete temporary files
    for i in range(len(group_names)):
        os.remove(f"{processor.data_path}csv_outputs/search_terms_temp_helper_{group_names[i]}_occurrences.csv")
        for j in range(len(search_terms_df.columns)):
            os.remove(f"{processor.data_path}csv_outputs/search_terms_temp_helper_{group_names[i]}_counts_by_{search_terms_df.columns[j]}.csv")
            
            
def gen_co_occurring_terms(processor, group_name, co_occurrence_terms_df, n_words = 50):
    "get a list of the top words occurring alongside terms. Uses character_buffer of the gen_search_terms_function, which must be run first"
    co_corpus = pd.read_csv(f"{processor.data_path}csv_outputs/search_terms_{group_name}_occurrences.csv")
    
    co_occurrence_terms_df = co_occurrence_terms_df.replace("", np.nan)
    for i in range(len(co_occurrence_terms_df)):
        print(f"co-occurence search for group {group_name}: {i+1}/{len(co_occurrence_terms_df)}")
        most_specific_col = co_occurrence_terms_df.iloc[i, :].last_valid_index()
        most_specific_term = co_occurrence_terms_df.loc[i, most_specific_col]
        
        # most specific permutations in this group
        permutations = list(co_corpus.loc[lambda x: x[most_specific_col] == most_specific_term, co_occurrence_terms_df.columns[-1]].unique())

        tmp_corpus = co_corpus.loc[lambda x: x[most_specific_col] == most_specific_term, "character_buffer_context"].reset_index(drop =True)
        
        corpus_string = ""
        for j in range(len(tmp_corpus)):
            corpus_string += tmp_corpus[j]
        
        corpus_string = corpus_string.replace(" | ", " ").replace("|", "").replace("  ", " ").replace("   ", " ").split(" ")
        corpus_string = [x for x in corpus_string if x not in stopwords.words("english") + ["also"]]
        counts = dict()
        for word in corpus_string:
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1
        
        counts = {word:count for word, count in counts.items() if (len(word) > 1) and not(word.isnumeric()) and not(word in permutations)}
        tmp_counts = pd.DataFrame({
            "co_occurrent_word": counts.keys(),
            "count": counts.values()
        }).sort_values(["count", "co_occurrent_word"], ascending = [False, True], axis = 0).reset_index(drop=True).loc[:n_words-1, :]
        for col in co_occurrence_terms_df.columns:
            tmp_counts[col] = co_occurrence_terms_df.loc[i, col]
        tmp_counts = tmp_counts.loc[:, list(co_occurrence_terms_df.columns) + ["co_occurrent_word", "count"]]
            
        if i == 0:
            co_occurrence = tmp_counts.copy()
        else:
            co_occurrence = pd.concat([co_occurrence, tmp_counts], ignore_index = True)
    
    co_occurrence.to_csv(f"{processor.data_path}csv_outputs/search_terms_{group_name}_co_occurrences.csv", index = False)
    
def gen_second_level_search_terms(processor, group_name, second_level_search_terms_df):
    "get counts of words that appear within the subcorpus of matched search terms"
    corpus = pd.read_csv(f"{processor.data_path}csv_outputs/search_terms_{group_name}_occurrences.csv")
    
    second_level_search_terms_df = second_level_search_terms_df .replace("", np.nan)
    for i in range(len(second_level_search_terms_df)):
        print(f"second-level search for group {group_name}: {i+1}/{len(second_level_search_terms_df)}")
        search_term = second_level_search_terms_df.iloc[i, -1]
        most_specific_col = second_level_search_terms_df.iloc[i, :-1].last_valid_index()
        most_specific_term = second_level_search_terms_df.loc[i, most_specific_col]
        
        tmp_corpus = corpus.loc[lambda x: x[most_specific_col] == most_specific_term, "character_buffer_context"].reset_index(drop =True)
        
        corpus_string = ""
        for j in range(len(tmp_corpus)):
            corpus_string += tmp_corpus[j]
        
        occurrences = len([x.start() for x in re.finditer(search_term, corpus_string)])
        
        tmp_occurrences = pd.DataFrame({
            "count": occurrences
        }, index = [0])
        for col in second_level_search_terms_df.columns:
            tmp_occurrences[col] = second_level_search_terms_df.loc[i, col]
        tmp_occurrences = tmp_occurrences.loc[:, list(second_level_search_terms_df.columns) + ["count"]]
        
        if i == 0:
            second_level_occurrences = tmp_occurrences.copy()
        else:
            second_level_occurrences = pd.concat([second_level_occurrences, tmp_occurrences], ignore_index = True)
            
    second_level_occurrences.to_csv(f"{processor.data_path}csv_outputs/search_terms_{group_name}_second_level_counts.csv", index = False)
    
def gen_top_words(processor, group_names, text_ids, path_prefix, per_1000 = True, top_n = 100, exclude_words = []):
    "get a list of the top words. If more than one list of groups is passed, subsequent ones will look for counts on the word list created from the first group"
    # word count dict function
    def gen_word_count_df(stringx, exclude_words, n, per_1000 = False):
        counts = dict()
        words = stringx.split(" ")
        
        for word in words:
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1
        
        if per_1000:
            counts = {word:(count / len(words) * 1000) for word, count in counts.items() if (len(word) > 1) and not(word in exclude_words) and not(word.isnumeric())}
        else:
            counts = {word:count for word, count in counts.items() if (len(word) > 1) and not(word in exclude_words) and not(word.isnumeric())}
        count_df = pd.DataFrame({"word": counts.keys(), "count": counts.values()}).sort_values("count", ascending = False).reset_index(drop = True).iloc[:n,:]
        return count_df
    
    # create a corpus from text ids
    def create_corpus(text_ids_i):
        corpus = ""
        for text_id in text_ids_i:
            file_path = f"{processor.data_path}transformed_txt_files/{path_prefix}_{text_id}.txt"
            file = open(file_path, "r", encoding = "latin1")
            stringx = file.read()
            file.close()
            
            corpus += stringx
        return corpus
    
    # creating the initial word list from the first group
    text_ids_i = text_ids[0]
    corpus = create_corpus(text_ids_i)
    count_df = gen_word_count_df(stringx = corpus, exclude_words = exclude_words, n = top_n, per_1000 = per_1000)
    count_df = count_df.rename(columns = {"count": group_names[0]})
    
    # adding counts of other columns
    if len(text_ids) > 1:
        for i in range(1, len(group_names)):
            text_ids_i  = text_ids[i]
            # create the corpus
            corpus = create_corpus(text_ids_i)
            tmp_count_df = gen_word_count_df(stringx = corpus, exclude_words = exclude_words, n = 999999999999999, per_1000 = per_1000)
            
            count_df = count_df.merge(tmp_count_df, how = "left", on = "word")
            count_df = count_df.rename(columns = {"count": group_names[i]})
    count_df = count_df.fillna(0)
    count_df.to_csv(f"{processor.data_path}csv_outputs/top_{top_n}_words.csv", index = False)