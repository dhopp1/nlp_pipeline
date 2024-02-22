
# Change Log
### 0.0.31
### Fixed
* fixed typo in gen_top_words function

### 0.0.30
### Added
* ability to get top n word counts per 1000 words via the `gen_top_words` function

### 0.0.29
### Added
* progress counters to all search term functions

## 0.0.28
### Added
* ability to select out certain PDF pages when processing via the `filter_pdf_pages` function
* ability to clear directories via the `clear_directories` function
* ability to find top co-occurring words via the `gen_co_occurring_terms` function
* ability to get counts of second-level search terms via the `gen_second_level_search_terms` function

## 0.0.27
### Added
* ability to convert texts to UTF-8 via the `convert_utf8` function
* ability to replace words in the raw text files or transformed text files via the `replace_words` function
* ability to search for terms and get their context and sentiment via the `gen_search_terms` and `gen_aggregated_search_terms` functions

## 0.0.26
### Added
* ability to manually force OCR PDF conversion via the `force_ocr` parameter of the `convert_to_text` function

## 0.0.25
### Fixed
* better detect more OCR PDFs

## 0.0.24
### Added
* calculation of number of numeric and alpha characters in text
* enable initial input of local text files instead of web URLs

## 0.0.23
### Fixed
* better detect more OCR PDFs

## 0.0.22
### Added
* ability to only show certain topics and text IDs in visualize\_topics\_presence plot, as well as save the dataframe

## 0.0.21
### Fixed
* better detect more OCR PDFs

## 0.0.20
### Added
* added ability to plot entities with the bar plot, word cloud plot, and occurrences over time plot

## 0.0.19
### Added
* added BERTopic functionality

## 0.0.18
### Fixed
* NER for longer texts

## 0.0.17
### Added
* named entity recognition (NER), counts of entities in each text

## 0.0.16
### Added
* better detect some OCR PDFs

## 0.0.15
### Added
* ability to create text similarity cluster plots

## 0.0.14
### Added
* ability to calculate and plot text similarity with `processor.plot_text_similarity` function

## 0.0.13
### Fixed
* make word length and incidence more robust to errors

## 0.0.12
### Fixed
* make pdf conversion more robust to errors

## 0.0.11

### Fixed
* make robust to word incidence language not found

## 0.0.10

### Fixed
* make robust to sentiment divide by 0 errors

## 0.0.9

### Added
* better detect some OCR PDFs

## 0.0.8

### Added
* ability to generate sentence-by-sentence sentiment report for a text_id or string with the `processor.gen_sentiment_report()` function

## 0.0.7

### Fixed
* make robust to language not available for Snowball stemmer

## 0.0.6

### Fixed
* better detect some OCR PDFs

## 0.0.5

### Fixed
* make robust to language detect issues

## 0.0.4

### Fixed
* fix writing UTF-8 characters to text

## 0.0.3

### Added
* add plotting for summary statistics

## 0.0.2

### Fixed
* default to English stopwords if detected language isn't in NLTK stopwords list
* catch more OCR PDFs

## 0.0.1

### Added
* initial release
