from joblib import load
import logging
import pandas as pd
import numpy as np
import re
import seldon_core
import os

# For downloading the model and OHE encoder from GCS
from io import BytesIO
from google.cloud import storage

import nltk
from nltk.corpus import stopwords
nltk.download("stopwords", download_dir="./nltk")
nltk.data.path.append("./nltk")

logger = logging.getLogger(__name__)


class TweetSentiment(object):

    def __init__(self, model_path, tfidf_path):
        logger.info(f"Connecting to GCS")
        self.client = storage.Client.create_anonymous_client()
        self.bucket = self.client.bucket('tom-seldon-examples')

        logger.info(f"Model name: {model_path}")
        self.model_path = model_path

        logger.info(f"TF-IDF Name: {tfidf_path}")
        self.tfidf_path = tfidf_path

        logger.info("Loading model file and TF-IDF vectorizer.")
        self.load_deployment_artefacts()
        self.ready = False

    def load_deployment_artefacts(self):
        logger.info("Loading model")
        model_file = BytesIO()
        model_blob = self.bucket.get_blob(f'{self.model_path}')
        model_blob.download_to_file(model_file)
        self.model = load(model_file)

        logger.info("Loading TF-IDF vectorizer")
        tfidf_file = BytesIO()
        tfidf_blob = self.bucket.get_blob(f'{self.tfidf_path}')
        tfidf_blob.download_to_file(tfidf_file)
        self.tfidf = load(tfidf_file)
        
        self.ready = True

    # Remove stop words
    def remove_stopwords(self, text):
        text = ' '.join([word for word in text.split() if word not in (stopwords.words('english'))])
        return text

    # Remove url  
    def remove_url(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)

    # Remove punct
    def remove_punct(self, text):
        table = str.maketrans('', '', string.punctuation)
        return text.translate(table)

    # Remove html 
    def remove_html(self, text):
        html=re.compile(r'<.*?>')
        return html.sub(r'',text)

    # Remove @username
    def remove_username(self, text):
        return re.sub('@[^\s]+','',text)

    # Remove emojis
    def remove_emoji(self, text):
        emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    # Decontraction text
    def decontraction(self, text):
        text = re.sub(r"won\'t", " will not", text)
        text = re.sub(r"won\'t've", " will not have", text)
        text = re.sub(r"can\'t", " can not", text)
        text = re.sub(r"don\'t", " do not", text)
        text = re.sub(r"can\'t've", " can not have", text)
        text = re.sub(r"ma\'am", " madam", text)
        text = re.sub(r"let\'s", " let us", text)
        text = re.sub(r"ain\'t", " am not", text)
        text = re.sub(r"shan\'t", " shall not", text)
        text = re.sub(r"sha\n't", " shall not", text)
        text = re.sub(r"o\'clock", " of the clock", text)
        text = re.sub(r"y\'all", " you all", text)
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"n\'t've", " not have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'d've", " would have", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ll've", " will have", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)
        text = re.sub(r"\'re", " are", text)
        return text  

    # Seperate alphanumeric
    def seperate_alphanumeric(self, text):
        words = text
        words = re.findall(r"[^\W\d_]+|\d+", words)
        return " ".join(words)

    def cont_rep_char(self, text):
        tchr = text.group(0) 
        
        if len(tchr) > 1:
            return tchr[0:2] 

    def unique_char(self, rep, text):
        substitute = re.sub(r'(\w)\1+', rep, text)
        return substitute

    def char(self, text):
        substitute = re.sub(r'[^a-zA-Z]',' ',text)
        return substitute

    def predict(self, tweets, names=[], meta={}):
        try:
            if not self.ready:
                self.load_deployment_artefacts()
            else:
                final_text = []

                for text in tweets:
                    # Apply functions to tweets
                    text = self.remove_username(text)
                    text = self.remove_url(text)
                    text = self.remove_emoji(text)
                    text = self.decontraction(text)
                    text = self.seperate_alphanumeric(text)
                    text = self.unique_char(self.cont_rep_char,text)
                    text = self.char(text)
                    text = text.lower()
                    text = self.remove_stopwords(text)
                    final_text.append(text)

                logger.info(f"Final text to be embedded: {final_text}")
                embeddings = self.tfidf.transform(final_text)
                sentiment = self.model.predict(embeddings)
                return sentiment

        except Exception as ex:
            logging.exception(f"Failed during predict: {ex}")