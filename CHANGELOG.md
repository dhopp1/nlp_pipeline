
# Change Log

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
