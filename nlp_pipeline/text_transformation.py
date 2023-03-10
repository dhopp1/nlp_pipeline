import pandas as pd
import string
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from wordfreq import zipf_frequency
import os

# key between langdetect language ISO code and NLTK's names for snowball and stopwords
nltk_langdetect_dict = {
    'ar':'arabic',
    'az':'azerbaijani',
    'eu':'basque',
    'bn':'bengali',
    'ca':'catalan',
    'zh':'chinese',
    'da':'danish',
    'nl':'dutch',
    'en':'english',
    'fi':'finnish',
    'fr':'french',
    'de':'german',
    'el':'greek',
    'he':'hebrew',
    'hu':'hungarian',
    'id':'indonesian',
    'it':'italian',
    'kk':'kazakh',
    'ne':'nepali',
    'no':'norwegian',
    'pt':'portuguese',
    'ro':'romanian',
    'ru':'russian',
    'sl':'slovene',
    'es':'spanish',
    'sv':'swedish',
    'tg':'tajik',
    'tr':'turkish'
}
snowball_languages = [
    'arabic',
    'danish',
    'dutch',
    'english',
    'finnish',
    'french',
    'german',
    'hungarian',
    'italian',
    'norwegian',
    'portuguese',
    'romanian',
    'russian',
    'spanish',
    'swedish'
]
def gen_nltk_lang_dict(dictionary, lang):
    try:
        return dictionary[lang]
    except:
        return "english"

def lower(stringx):
    "lower case the text"
    return stringx.lower()
    
def replace_newline_period(stringx):
    "replace new line characters, new page characters, and periods with |. Also remove multiple white spaces"
    # replace newlines with |s
    stringx = stringx.replace("\n", " | ")
    
    # replace [newpage]
    stringx = stringx.replace("[newpage]", "")

    # remove multiple whitespace
    stringx = " ".join(stringx.split())

    # replace all periods with |, including a space so the words can be found independently of the period
    stringx = stringx.replace(".", " | ")
    
    # append a vertical line to beginning and end so all sentences are enclosed by two |s
    stringx = "| " + stringx + " |"
    
    return stringx
    
def remove_punctuation(stringx):
    "remove punctuation, except |s"
    stringx = stringx.translate(
        str.maketrans(string.punctuation.replace("|", "") + "??????????????", ' '*len(string.punctuation.replace("|", "") + "??????????????"))
    )
    return stringx

def remove_stopwords(stringx, language):
    "remove stopwords"
    eng_stopwords = stopwords.words(gen_nltk_lang_dict(nltk_langdetect_dict, language))

    # remove stopwords, tweet tokenizer because doens't split apostrophes
    tk = TweetTokenizer()
    tokenized_string = tk.tokenize(stringx)
    stringx = [item for item in tokenized_string if item not in eng_stopwords]

    stringx = " ".join(stringx)

    return stringx
    
def stem(stringx, stemmer=None, language=None):
    "stem a string. snowball is less agressive, lancaster only works with english"
    if stemmer == "snowball":
        if not(language in snowball_languages):
            language = "english"
        stemmer = SnowballStemmer(gen_nltk_lang_dict(nltk_langdetect_dict, language))
    elif stemmer == "lancaster":
        stemmer = LancasterStemmer()
    if stemmer != None:
        tk = TweetTokenizer()
        tokenized_string = tk.tokenize(stringx)
        stringx = [stemmer.stem(item) for item in tokenized_string]
        stringx = " ".join(stringx)
    return stringx

def gen_word_count_dict(stringx, exclude_words):
    "create a dictionary of word counts in a string. exclude_words is a list of words to filter out"
    counts = dict()
    words = stringx.split(" ")

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    
    counts = {word:count for word, count in counts.items() if (len(word) > 1) and not(word in exclude_words) and not(word.isnumeric())}

    return counts

def get_single_sentiment(stringx, sentiment_analyzer = SentimentIntensityAnalyzer):
    "get sentiment of a single string"
    sia = sentiment_analyzer()
    return sia.polarity_scores(stringx)  

def get_word_frequency(word, language):
    "https://pypi.org/project/wordfreq/"
    try:
        freq = zipf_frequency(word, language)
    except:
        freq = zipf_frequency(word, "en")
    return freq

def gen_sentiment_report(stringx, sentiment_analyzer = SentimentIntensityAnalyzer):
    "generate sentiment report of a string"
    stringx = lower(stringx)
    stringx = replace_newline_period(stringx)
    stringx = remove_punctuation(stringx)
    
    string_list = stringx.split("|")
    string_list = [x for x in string_list if len(set(x)) > 1]
    sentiment_list = [get_single_sentiment(x, sentiment_analyzer)["compound"] for x in string_list]
    
    sentiment_report = pd.DataFrame({
        "sentence_number": list(range(1,len(string_list)+1)),
        "sentence": string_list,
        "sentiment": sentiment_list
    })
    
    return sentiment_report