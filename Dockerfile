FROM python

WORKDIR /app

# install poppler, tesseract, antiword
RUN apt-get update
RUN apt-get install poppler-utils -y
RUN apt-get install tesseract-ocr -y
RUN apt-get install antiword

# install required python libraries
RUN pip install -r https://raw.githubusercontent.com/dhopp1/nlp_pipeline/main/requirements.txt
RUN pip install --ignore-installed six
RUN pip install nlp-pipeline

# download nltk stopwords and sentiment lexicon
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('vader_lexicon')"

# download desired Spacy models
#RUN python -m spacy download ca_core_news_lg
#RUN python -m spacy download zh_core_web_lg
#RUN python -m spacy download hr_core_news_lg
#RUN python -m spacy download nl_core_news_lg
#RUN python -m spacy download en_core_web_lg
#RUN python -m spacy download fi_core_news_lg
#RUN python -m spacy download fr_core_news_lg
#RUN python -m spacy download de_core_news_lg
#RUN python -m spacy download el_core_news_lg
#RUN python -m spacy download it_core_news_lg
#RUN python -m spacy download ja_core_news_lg
#RUN python -m spacy download ko_core_news_lg
#RUN python -m spacy download lt_core_news_lg
#RUN python -m spacy download mk_core_news_lg
#RUN python -m spacy download nb_core_news_lg
#RUN python -m spacy download pl_core_news_lg
#RUN python -m spacy download pt_core_news_lg
#RUN python -m spacy download ro_core_news_lg
#RUN python -m spacy download ru_core_news_lg
#RUN python -m spacy download es_core_news_lg
#RUN python -m spacy download sv_core_news_lg
#RUN python -m spacy download uk_core_news_lg