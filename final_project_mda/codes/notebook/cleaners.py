import re
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
import spacy
import spacy.cli
spacy.cli.download("en_core_web_sm")
import string



class DataCleaners():

    def clean_text(text):
        text = text.lower()
        text = re.sub('\n', ' ', text)
        text = re.sub('\t', ' ', text)
        text = re.sub("b'", ' ', text)
        text = re.sub('b"', ' ', text)
        text = re.sub(r'[0-9]', ' ', text) #ideally this removes only line number such as "xxx." 9/11 can be an important nb
        text = re.sub(r'\[,!.*?\]', ' ', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text) # removes the possibility to tokenize by sentence
        text = re.sub(r'xef*?', ' ', text)
        text = re.sub(r'xbb*?', ' ', text)
        text = re.sub(r'xbf*?', ' ', text)
        text = re.sub(r"\b[a-zA-Z]\b", " ", text)
        text = re.sub(' +', ' ', text)
        return text
    
    def remove_stopwords(text,stopwords):
        filtered = []
        stopwords_corpus = nltk.corpus.stopwords.words('english')
        stopwords_additional = stopwords
        stop_words = stopwords_corpus + stopwords_additional 
        stemmer = WordNetLemmatizer()
        word_tokens = word_tokenize(text)
        for w in word_tokens:
            if w not in stop_words:
                w = stemmer.lemmatize(w)
                filtered.append(w)
        filtered_doc = ' '.join(str(i) for i in filtered)
        return filtered_doc
    
    def lemmatizer(text):
        nlp = spacy.load('en_core_web_sm')
        sent = []
        doc = nlp(text)
        for word in doc:
            sent.append(word.lemma_)
        return " ".join(sent)

