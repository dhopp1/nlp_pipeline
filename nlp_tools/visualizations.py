import pandas as pd
import matplotlib.pyplot as plt
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