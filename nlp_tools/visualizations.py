import pandas as pd
import matplotlib.pyplot as plt

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
    plt.ylabel("Occurences")
    plt.title(title)
    
    return (p, plot_df)