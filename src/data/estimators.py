from sklearn.base import BaseEstimator

import numpy as np
import spacy

from ..logging import logger

class SpacyTokenize(BaseEstimator):
    def __init__(self, n_threads=4, batch_size=50,
                 stopwords=None, punctuation=None, lemmatize=False,
                 language_model='en_core_web_sm'):
        self.n_threads = n_threads
        self.batch_size = batch_size
        self.stopwords = stopwords
        self.punctuation = punctuation
        self.language_model = language_model
        self.lemmatize = lemmatize
    
    def fit(self, X, y=None):
        """Prepare the spacy parser"""
        if self.stopwords is None:
            self.stopwords = []
        if self.punctuation is None:
            self.punctuation = []
        logger.debug(f"Loading language model:{self.language_model}")
        self.parser_ = spacy.load(self.language_model)
        return self
    
    def transform(self, X, y=None):
        """Transform X into list of sentences"""
        if not hasattr(self, 'parser_'):
            raise Exception(f"Must fit() before transform()")
        logger.debug(f"Tokenizing sentences...")
        self.sentences_ = []
        for i, doc in enumerate(self.parser_.pipe(X, batch_size=self.batch_size, n_threads=self.n_threads)):
            if i and i % 1000 == 0:
                logger.debug(f"tokenized {i} sentences.")
            for sentence in doc.sents:
                if self.lemmatize:
                    tokens = [tok for tok in sentence]
                    tokens = [tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_ for tok in tokens]
                else:
                    tokens = [tok.text.lower().strip() for tok in sentence]
                tokens = [tok for tok in tokens if (tok not in self.stopwords and tok not in self.punctuation)]

                self.sentences_.append(tokens)
        self.sentences_ = np.array(self.sentences_)
        return self.sentences_
    
