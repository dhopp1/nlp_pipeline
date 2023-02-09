import pandas as pd
import string
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
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
        str.maketrans(string.punctuation.replace("|", "") + "”“’;•", ' '*len(string.punctuation.replace("|", "") + "”“’;•"))
    )
    return stringx

def remove_stopwords(stringx, language):
    "remove stopwords"
    eng_stopwords = stopwords.words(nltk_langdetect_dict[language])

    # remove stopwords, tweet tokenizer because doens't split apostrophes
    tk = TweetTokenizer()
    tokenized_string = tk.tokenize(stringx)
    stringx = [item for item in tokenized_string if item not in eng_stopwords]

    stringx = " ".join(stringx)

    return stringx
    
def stem(stringx, stemmer=None, language=None):
    "stem a string. snowball is less agressive, lancaster only works with english"
    if stemmer == "snowball":
        stemmer = SnowballStemmer(nltk_langdetect_dict[language])
    elif stemmer == "lancaster":
        stemmer = LancasterStemmer()
    if stemmer != None:
        tk = TweetTokenizer()
        tokenized_string = tk.tokenize(stringx)
        stringx = [stemmer.stem(item) for item in tokenized_string]
        stringx = " ".join(stringx)
    return stringx

