import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
from wordcloud import WordCloud

def convert_word_count_dict_to_df(df):
    "helper function to convert word counts dictionaries to one dataframe"
    clean_data = pd.DataFrame(columns=["word", "count"])
    
    for data in df["word_count_dict"]:
        tmp_dict = eval(data)
        tmp_df = pd.DataFrame(tmp_dict, index=[0]).transpose().reset_index().rename(columns={"index":"word", 0:"count"})
        clean_data = pd.concat([clean_data, tmp_df])
    
    # aggregate
    clean_data = clean_data.groupby("word").sum().reset_index().sort_values(["count"], ascending=False).reset_index(drop=True)
    
    return clean_data
    
def bar_plot_word_count(df, n_words, title=""):
    "get a bar plot of top words"
    plot_df = df.iloc[:n_words,:].reset_index(drop=True)
    
    p = plt.figure()
    plt.bar(plot_df.word, plot_df["count"])
    plt.xticks(rotation = 90)
    plt.ylabel("Occurrences")
    plt.title(title)
    
    return (p, plot_df)

def word_cloud(df, n_words):
    plot_df = df.iloc[:n_words,:].reset_index(drop=True)
    text_list = [[plot_df.loc[i, "word"]] * plot_df.loc[i, "count"] for i in range(len(plot_df))]
    flat_list = [item for sublist in text_list for item in sublist]
    text = " ".join(flat_list)
    
    word_cloud = WordCloud(
        collocations = False, 
        background_color = "white", 
        color_func=lambda *args, **kwargs: (0,0,0)
    ).generate(text)
    
    p = plt.figure()
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    
    return (p, plot_df)

def get_word_occurrences(df_list, word):
    "get counts of a specific word in different documents or groups of documents. Goes on contains rather than exact match."
    counts = []
    for df in df_list:
        counts.append(sum(list(df.loc[df.word.str.contains(word), "count"].values)))
    return counts

def plot_word_occurrences(df, text_ids_list, word, x_labels = None, title = ""):
    "plot occurrences of a word in different documents or group of documents. df = word count dict csv, text_ids_list = list of text ids, x_labels = xlabels for the document gorups"
    # if no x labels just 1:n
    if x_labels is None:
        x_labels = list(range(1, len(text_ids_list)+1))
        
    df_list = []
    for id_group in text_ids_list:
        if type(id_group) != list:
            id_group = [id_group]
        group_df = df.loc[df.text_id.isin(id_group), :].reset_index(drop=True)
        df_list.append(convert_word_count_dict_to_df(group_df))

    counts = get_word_occurrences(df_list, word)
        
    p = plt.figure()
    plt.plot(x_labels, counts)
    plt.title(f"{title} Occurrences of '{word}'")
    
    return (p, pd.DataFrame({"x_label":x_labels, "count":counts}))

def plot_summary_stats(df, text_ids_list, x_labels = None, title = "", summary_stats_col = "n_words"):
    "plot summary stats in different documents or group of documents. df = summary_stats csv, text_ids_list = list of text ids, x_labels = xlabels for the document gorups"
    # if no x labels just 1:n
    if x_labels is None:
        x_labels = list(range(1, len(text_ids_list)+1))
        
    # subtitle text for which column is being shown
    if summary_stats_col == "n_words":
        subtitle = "number of words"
    elif summary_stats_col == "n_unique_words":
        subtitle = "number of unique words"
    elif summary_stats_col == "n_sentences":
        subtitle = "number of sentences"
    elif summary_stats_col == "n_pages":
        subtitle = "number of pages"
    elif summary_stats_col == "avg_word_length":
        subtitle = "average word length"
    elif summary_stats_col == "avg_word_incidence":
        subtitle = "average word incidence"
    elif summary_stats_col == "numeric_proportion":
        subtitle = "numeric proportion"
    else:
        subtitle = ""
        
    df_list = []
    for id_group in text_ids_list:
        if type(id_group) != list:
            id_group = [id_group]
        group_df = df.loc[df.text_id.isin(id_group), :].reset_index(drop=True)
        df_list.append(group_df)

    values = [sum(x.loc[:, summary_stats_col]) / len(x.loc[:, summary_stats_col]) for x in df_list]
        
    p = plt.figure()
    plt.plot(x_labels, values)
    plt.title(f"{title} {subtitle}")
    
    return (p, pd.DataFrame({"x_label":x_labels, "value":values}))

def plot_sentiment(df, text_ids_list, x_labels = None, title = "", sentiment_col = "avg_sentiment_wo_neutral"):
    "plot average sentiment in different documents or group of documents. df = sentiment csv, text_ids_list = list of text ids, x_labels = xlabels for the document gorups"
    # if no x labels just 1:n
    if x_labels is None:
        x_labels = list(range(1, len(text_ids_list)+1))
        
    # subtitle text for which column is being shown
    if sentiment_col == "avg_sentiment_wo_neutral":
        subtitle = "Average sentiment, excluding neutral phrases"
    elif sentiment_col == "avg_sentiment_w_neutral":
        subtitle = "Average sentiment, including neutral phrases"
    elif sentiment_col == "neutral_proportion":
        subtitle = "Percentage of neutral phrases"
        
    df_list = []
    for id_group in text_ids_list:
        if type(id_group) != list:
            id_group = [id_group]
        group_df = df.loc[df.text_id.isin(id_group), :].reset_index(drop=True)
        df_list.append(group_df)

    values = [sum(x.loc[:, sentiment_col]) / len(x.loc[:, sentiment_col]) for x in df_list]
        
    p = plt.figure()
    plt.plot(x_labels, values)
    plt.title(f"{title} {subtitle}")
    
    return (p, pd.DataFrame({"x_label":x_labels, "value":values}))

def gen_similarity(processor, text_ids, path_prefix = ""):
    "generate text similary matrix from TF-IDF cosine similarity"
    docs = []
    
    for text_id in text_ids:
        if path_prefix == "":
            text_path = processor.metadata.loc[lambda x: x.text_id == text_id, "local_txt_filepath"].values[0]
        else:
            text_path = f"{processor.data_path}transformed_txt_files/{path_prefix}_{text_id}.txt"
        file = open(f"{text_path}", "r", encoding = "UTF-8") 
        stringx = file.read()
        file.close()
        
        docs.append(stringx)
    
    tfidf = TfidfVectorizer().fit_transform(docs)
    pairwise_similarity = tfidf * tfidf.T
    
    # text ids for rows and columns
    df = pd.DataFrame(pairwise_similarity.toarray())
    df.columns = text_ids
    df["text_id"] = text_ids
    df = df.set_index("text_id", drop=True)
    
    return df

def gen_similarity_plot(processor, text_ids, path_prefix = "", label_column = "text_id", figsize = (22, 16)):
    "plot text similarity matrix. Label column is metadata column to rename text ids"
    df = gen_similarity(processor, text_ids, path_prefix)
    
    x_axis_labels = [processor.metadata.loc[lambda x: x.text_id == text_id, label_column].values[0] for text_id in text_ids] # labels for x-axis
    y_axis_labels = x_axis_labels
    
    p = plt.figure(figsize=figsize)
    sns.heatmap(df, cmap="Reds", xticklabels=x_axis_labels, yticklabels=y_axis_labels, annot=True)
    plt.ylabel("")
    
    return (p, df, x_axis_labels)

def gen_cluster_df(processor, text_id_dict, path_prefix = ""):
    "given dict of groups + text ids, return two principal components of text similarity"
    # flattened list of all text_ids
    flattened_list = [item for sublist in list(text_id_dict.values()) for item in sublist]
    similarities = processor.plot_text_similarity(flattened_list, path_prefix)[1]
    
    # initialize empty dataframe
    df = pd.DataFrame(columns = ["text_id"] + list(text_id_dict.keys()))
    df["text_id"] = flattened_list
        
    # PCA to reduce to 2 axes
    pca = PCA(n_components = 2, random_state = 42)
    
    # pass our X to the pca and store the reduced vectors into pca_vecs
    pca_vecs = pca.fit_transform(similarities.values)
    
    plot_df = pd.DataFrame({
        "text_id": flattened_list,
        "group": [list(filter(lambda x: text_id in text_id_dict[x], text_id_dict))[0] for text_id in flattened_list],
        "pc1": pca_vecs[:,0],
        "pc2": pca_vecs[:,1]
    })
    
    return plot_df

def plot_cluster(plot_df, color_column): 
    "scatter plot of a cluster df. color_column = column of cluster df to ues for coloring groups"
    p = plt.figure()
    sns.scatterplot("pc1", "pc2", data=plot_df, hue=color_column)
    plt.ylabel("")
    plt.xlabel("")
    
    return p