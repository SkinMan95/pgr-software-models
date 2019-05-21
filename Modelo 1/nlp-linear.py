import os
import sys
from sys import stdin, stdout
import argparse
import time
import datetime
import logging
import traceback
import json
import csv
import re
from ast import literal_eval
import hashlib

from nlpStreamIterator import *

# external deps
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from scipy import sparse as sp_sparce


def command_line_parser():
    """
    returns a dict with the options passed to the command line
    according with the options available
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--inputfile", type=str, required=True, default="-",
                        help="input file, stdin as default")

    parser.add_argument("-dt", "--data_tag", type=str, required=True, default="text",
                        help="tweet tags from which to get the input data")

    parser.add_argument("-pt", "--prediction_tag", type=str, required=True, default="hashtags",
                        help="tweet tags to predict and to train")
    
    parser.add_argument("-o", "--outputfile", type=str, default="-",
                        help="output file name, stdout as default")

    parser.add_argument("-v", "--verbose", action='store_true',
                        help="verbose output")

    args = parser.parse_args()

    return args

class Representation(object):
    def __init__(self, X_train, y_train):
        pass

class BagOfWords(Representation):
    def __init__(self, X_train, y_train, dict_size = 5000):
        self.objid  = id(self)
        self.logger = logging.getLogger('BagOfWords' + str(self.objid))

        self.X_train = X_train
        self.y_train = y_train
        self.dict_size = dict_size

        tags_counts, words_counts = self.words_tags_count()
        self.calc_variables(tags_counts, words_counts)

    def calc_variables(self, tags_counts, words_counts):
        most_common_tags = sorted(tags_counts.items(), key=lambda x: x[1], reverse=True)[:dict_size]
        self.logger.debug("most common tags: %s", most_common_tags[:5])
        most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:dict_size]
        self.logger.debug("most common words: %s", most_common_words[:5])
        self.WORDS_TO_INDEX = {key[0]:idx for idx,key in enumerate(most_common_words)}
        self.INDEX_TO_WORDS = [word[0] for word in most_common_words]
        self.ALL_WORDS = self.WORDS_TO_INDEX.keys()

    def words_tags_count(self):
        tags_counts = {}
        words_counts = {}

        for line in self.X_train:
            for word in line.split():
                t = word.strip()
                words_counts[t] = words_counts.get(t, 0) +1
        
        for tags in self.y_train:
            for tag in tags:
                t = tag.strip()
                tags_counts[t] = tags_counts.get(t, 0) +1

        return (words_counts, tags_counts)

    def calc_bow(self, text):
        """
        text: a string
        
        return a vector which is a bag-of-words representation of 'text'
        """
        result_vector = np.zeros(self.dict_size)
        keys = self.WORDS_TO_INDEX.keys()
    
        for word in text.split():
            if word in keys:
                result_vector[self.WORDS_TO_INDEX[word]] += 1
    
        return result_vector

    def calc_bow_dataset(self, dataset):
        return sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text)) \
                                 for text in dataset])

class TFIDF(Representation):
    def __init__(self):
        pass

class NLPLinearMod(object):
    def __init__(self):
        self.objid  = id(self)
        self.logger = logging.getLogger('NLPLinearMod' + str(self.objid))
        
        self.get_dependencies()
        
        self.logger.debug("initialized constructor")

    def get_dependencies(self):
        self.logger.info('getting dependencies ...')
        nltk.download('stopwords')

        NLPLinearMod.REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        NLPLinearMod.BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_:]')
        NLPLinearMod.STOPWORDS = set(stopwords.words('english'))
        self.logger.info('finished getting dependencies')

    def text_prepare(self, text):
        """
        text: a string
        
        return: modified initial string
        """
        text = text.lower()
        text = NLPLinearMod.REPLACE_BY_SPACE_RE.sub(" ", text)
        text = NLPLinearMod.BAD_SYMBOLS_RE.sub("", text)
        text = " ".join([word for word in text.split() if word not in NLPLinearMod.STOPWORDS])
        return text
        
    def get_hashtags(self, txt):
        return [t.strip("#") for t in re.findall(r'#\S+', txt)]
        
    def get_predictions(self, d, inputtag, tag=None):
        if tag is None or tag not in d.keys():
            # gets hashtags manually
            return self.get_hashtags(d[inputtag])
        else:
            return d[tag]

    def calc_hash(self, txt):
        return hashlib.md5(txt.encode())
        
    def read_data(self, stream, inputtag, predicttag, filter=False):
        features = []
        hyp = []
        hashes = set()
        for dd in stream:
            hash = self.calc_hash(dd[inputtag])
            if hash in hashes: continue # skip repetitions, should prevent overfitting
            hashes.add(hash)
            
            preds = self.get_predictions(dd, inputtag, predicttag)
            if not filter or len(preds) > 0: # implication
                features.append(self.text_prepare(dd[inputtag]))
                hyp.append([self.text_prepare(t) for t in preds])

        d = {inputtag: features, predicttag: hyp}
        data = pd.DataFrame.from_dict(d)
        return data
        
    def trainModel(self, stream, inputtag, predicttag):
        train = self.read_data(stream, inputtag, predicttag, filter=True)
        X_train, y_train = train[inputtag].to_numpy(), train[predicttag].to_numpy()
        
        self.logger.debug("%s\n\n%s", X_train, y_train)
        self.logger.debug(X_train.shape, y_train.shape)

        
    
def main():
    options = command_line_parser()

    logging.basicConfig(level    = logging.DEBUG if options.verbose else logging.INFO,
                        format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt  = '%m-%d %H:%M',
                        filename = 'nlp-linear.log',
                        filemode = 'w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    ################ APPLICATION STARTS ################
    inputfile = open(options.inputfile, "r") if options.inputfile != '-' else stdin
    outputfile = open(options.outputfile + '.json', "w") if options.outputfile != '-' else stdout

    logging.debug("app options: %s", str(options))

    streamit = DictStreamIteratorJson(inputfile)
    
    model = NLPLinearMod()
    model.trainModel(streamit, options.data_tag, options.prediction_tag)

    inputfile.close()
    outputfile.close()
    
if __name__ == "__main__":
    main()
