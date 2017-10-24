# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import re, os
import time
import numpy as np 
from nltk.stem import WordNetLemmatizer
from nltk.stem import LancasterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import sys
reload(sys)
sys.setdefaultencoding("ISO-8859-1")

WnL = WordNetLemmatizer() # Lemmatizer instance
LS = LancasterStemmer() # Stemmer instance

def load_stop_words(path):
    """ return stop words in a python list """
    with open(path, 'rb') as f:
        stopwords = f.read()
    return stopwords.split('\r\n')

def load_docs_ap(path):
    """ read ap.txt file"""
    begin = False
    docs = []
    num_docs = 0
    with open(path, 'r') as f:
        for line in f:
            if line == "<TEXT>\n":
                begin = True
                docs.append("")
                continue
            if line == " </TEXT>\n":
                begin = False
                num_docs += 1
                continue
            if begin:
                docs[num_docs] += line
    
    return docs

def load_20newsgroups(path, num_docs=100):
    """ 
    arguments
    ----------
    path is the root directory which contains all newsgroup, 
    num_docs is number of documents in each label we want to select

    return
    -------
    docs:list - a list contains all documents
    labels:list - labels of all documents ordered as docs
    """
    labels_folder = os.listdir(path)
    docs = []
    labels = []
    docs_id = []
    i = 0
    for label in labels_folder:
        path2files = path + '/' + label
        files = os.listdir(path2files)
        numdoc = 0
        for fn in files:
            if numdoc == num_docs: break
            numdoc += 1
            with open(path2files + '/' + fn, 'rb') as f:
                docs.append(f.read())
                labels.append(i)
                docs_id.append(fn)
        i += 1
    
    return docs, labels, docs_id

def normalize(doc, stopwords = None, lemma = False, stem= False):
    """
    Parameters
    -----------
    - doc:str - is a document that needs to be normalized
    - stopwords:list - if it is supplied, the document will be eliminated all the stopwords
    - lemmatize:bool - if it is True, all the word will be converted into lemma form

    Returns
    -------
    - doc:str - normalized document
    """
    # change currency sign followed by number to ' currency '
    # doc = re.compile(r'(\€|\¥|\£|\$)\d+([\.\,]\d+)*').sub(' currency ', doc )

    # change hh:mm:ss to " timestr "
    # doc = re.compile(r'(\d{2}):(\d{2}):(\d{2})').sub(' timestr ', doc)

    # # change email to ' emailaddr '
    # doc = re.compile(r'[^\s]+@[^\s]+').sub(' emailaddr ', doc)

    # # change link to ' urllink '
    # doc = re.compile(r'(((http|https):*\/\/[^\s]*)|((www)\.[^\s]*)|([^\s]*(\.com|\.co\.uk|\.net)[^\s]*))').sub(' urllink ', doc)

    # change phone number into ' phone_numb '
    # doc = re.compile(r'\(?([0-9]{3})\)?([ .-]?)([0-9]{3})\2([0-9]{4})').sub(' phonenumb ', doc)

    # change sequence of number to ' numb_seq '
    # doc = re.compile(r'\d+[\.\,]*\d*').sub(' numbseq ', doc)

    # lowercase and split doc by characters are not in  0-9A-Za-z
    # docArr = re.compile(r'[^a-zA-Z0-9]').split(doc.lower())
    
    docArr = []
    for sent in sent_tokenize(doc.lower()):
        for word in word_tokenize(sent):
            docArr.append(word)

    if lemma:
        docArr = [lemmatize(word) for word in docArr]

    if stem:
        docArr =[LS.stem(word) for word in docArr]
        
    # remove stopwords
    if stopwords:
        stopwords = set(stopwords)
        docArr = [word for word in docArr if ((word not in stopwords) and (word != ''))]

    # return
    doc = ' '.join(docArr)
    return doc

def lemmatize(word):
    """
    transform word into lemma form
    ```
        { Part-of-speech constants
        ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
        }
    ```
    """
    rootWord = WnL.lemmatize(word, pos='n')
    if rootWord == word:
        rootWord = WnL.lemmatize(word, pos='a')
        if rootWord == word:
            rootWord = WnL.lemmatize(word, pos='v')
            if rootWord == word:
                rootWord = WnL.lemmatize(word, pos='r')
                if rootWord == word:
                    rootWord = WnL.lemmatize(word, pos='s')
    return rootWord

def stats_and_make_BoW(normalized_docs, max_df = 1.0, min_df = 0.0):
    """
    Parameters:
    ----------
    - `normalized_docs:list` - a list that contains all documents which are normalized
    - `max_df:float` - prune words that appear too common over all documents (not help us to distinguish docs)
    - `min_df:float` - prune words that appear too seldom over all documents (seem not appear in the future)
    
    Returns :
    -------
    - `bag:list` - bag of words
      * a list with all words are pairwise different (Bag Of Word),
    all the words in stats_all that has DF(word) value (the number of documents 
    has that word) not too big (not too popular in dataset since it is less 
    important for differentiate documents)

    - `stats_in_docs:list`
      * `stats_in_doc[i][word] = freq`, is the frequency of the word in document i'th
      * *NOTE*: `i` depends on the order of the document in list all documents

    - `stats_all:dict`
      * `stats_all[word]['df'] = number`, (document frequency) is the number of documents has that word
      * `stats_all[word]['freq'] = freq`, is the frequency of the word in all document
    """
    
    stats_in_doc = []
    for i in range(len(normalized_docs)):
        doc = normalized_docs[i]
        stats_in_doc.append({})
        for word in doc.split(' '):
            # Increase the frequency of the word in a documents
            if stats_in_doc[i].has_key(word):
                stats_in_doc[i][word] += 1
            else:
                stats_in_doc[i][word] = 1
    
    # Make stats_all dictionary
    stats_all = {}
    for i in range(len(normalized_docs)):
        for (word, freq) in stats_in_doc[i].items():
            if stats_all.has_key(word):
                stats_all[word]['freq'] += freq
                stats_all[word]['df'] += 1
            else:
                stats_all[word] = {}
                stats_all[word]['freq'] = freq
                stats_all[word]['df'] = 1 
    
    # create bag of words base on stats_all
    bag = []
    for word in stats_all.keys():
        if stats_all[word]['df'] <= max_df if (type(max_df) == int) else (len(normalized_docs) * max_df) \
        and stats_all[word]['df'] >= min_df if (type(min_df) == int) else (len(normalized_docs) * min_df):
            bag.append(word)
    
    return bag, stats_in_doc, stats_all

def vectorize(doc, pos, bag, stats_in_doc, stats_all):
    """
    Parameters:
    --------------
    - `doc:str` - is a normalized document will be vectorized
    - `pos:interger` - is the doc's position in dataset 
    - `stats_in_docs:list` 
    `stats_in_doc[i][word] = freq`, is the frequency of the word in document i'th
    *NOTE*: `i` depends on the order of the document in list all documents

    - `stats_all:dict`
    `stats_all[word]['df'] = number`, is the number of documents has that word
    `stats_all[word]['freq'] = freq`, is the frequency of the word in all document

    Returns :
    ------------
    - `xi:ndarray` - a vector of tf-idf value of each word represent for the input document
    """
    D = len(stats_in_doc)
    xi = []
    for i in range(len(bag)):
        xi.append(0)
    words = set(doc.split(' '))
    max_freq = max(stats_in_doc[pos].values())
    for word in words:
        if word in bag:
            tf =stats_in_doc[pos][word]/max_freq
            idf = np.log((D ) / ( stats_all[word]['df']))
            xi[bag.index(word)] = tf * idf
            # print("tfxid({}) = {} x {} = {}".format(word, tf, idf, tf*idf))

    xi = np.array(xi).reshape(1, len(xi))
    return xi


def vectorize_by_lib(normalized_docs, max_df = 1.0, min_df=1):
    tfidfVectorizer = TfidfVectorizer(encoding='utf-8', max_df=max_df, min_df=min_df)
    vecs = tfidfVectorizer.fit_transform(normalized_docs)
    BoW = tfidfVectorizer.get_feature_names()

    return vecs.todense(), BoW

if __name__ == "__main__":
    print("START PROGRAM")

    time_stack = [time.time()]
    savedir = os.path.dirname(os.path.realpath(__file__)) + '/generated_files'
    # Preprocessing
    print("Loading data from file...")
    docs, origin_labels, docs_id = load_20newsgroups('/media/hieu/DATA/ShareOS/PythonVirtualEnv/WorkSpace/test/unsupervised_learning/k-mean/datasets/20_newsgroups')    
    stopwords = load_stop_words('/media/hieu/DATA/ShareOS/PythonVirtualEnv/WorkSpace/test/unsupervised_learning/k-mean/stopwords.txt')
    print("{} documents".format(len(docs)))
    print("{} words in stopwords".format(len(stopwords)))

    time_stack.append(time.time())
    print("Done in {} seconds \n".format(time_stack[-1] - time_stack[-2]))
    
    print("Normalizing all documents...")
    skip = True
    if skip:
        print("Loading from file...")
        docs = pickle.load(open(savedir + '/normalized_docs.data', 'rb'))
        print("{} documents".format(len(docs)))
    else:
        docs = [normalize(doc, stopwords, lemma=False, stem= True) for doc in docs]
        print("{} documents".format(len(docs)))
        print("Normalized docs: ")
        print("writing to file...")
        pickle.dump(docs, open(savedir + '/normalized_docs.data', 'wb'))
    time_stack.append(time.time())
    print("Done in {} seconds \n".format(time_stack[-1] - time_stack[-2]))

    print("Making dictionary and statistic... ")
    skip = True
    if skip:
        print("Loading from file...")
        bag = pickle.load(open(savedir + '/BoW.data'))
        stats_in_doc = pickle.load(open(savedir + '/stats_in_doc.data'))
        stats_all = pickle.load(open(savedir + '/stats_all.data'))
        print("{} words in bag of words".format(len(bag)))
    else:
        bag, stats_in_doc, stats_all = stats_and_make_BoW(docs, max_df = 0.8, min_df = 3)
        print("{} words in bag of words".format(len(bag)))
        print("writing to file...")
        pickle.dump(bag, open(os.path.join(savedir, 'BoW.data'), 'wb'))
        pickle.dump(stats_in_doc, open(os.path.join(savedir, 'stats_in_doc.data'), 'wb'))
        pickle.dump(stats_all, open(os.path.join(savedir, 'stats_all.data'), 'wb'))
    
    time_stack.append(time.time())
    print("Done in {} seconds \n".format(time_stack[-1] - time_stack[-2]))

    print("press Enter to continue...")
    a = raw_input()
    
    print("Vectorizing all documents...")
    skip = True
    if skip:
        print("Loading from file...")
        X = np.zeros((len(docs), len(bag)))
        with open(savedir + '/training_set.data', 'r+b') as f:
            for i in range(len(docs)):
                X[i] = pickle.load(f)

        print("X.shape = ", X.shape)
    else:
        X = np.zeros((len(docs), len(bag)))
        print("X.shape = ", X.shape)
        open(savedir + '/training_set.data', 'w').close()
        with open(savedir + '/training_set.data', 'w+b') as f:
            for i in range(len(docs)):
                X[i] = vectorize(docs[i], i, bag, stats_in_doc, stats_all)
                pickle.dump(X[i], f)

    time_stack.append(time.time())
    print("Done in {} seconds \n".format(time_stack[-1] - time_stack[-2]))

    print("Writing metadata to files...")
    pickle.dump((len(docs), len(bag), origin_labels), open(savedir + '/meta.data', 'wb'))

    # X, bag = prep.vectorize_by_lib(docs, max_df = 0.7, min_df= 3)
    # print("{} words in bag of words".format(len(bag)))
    # X = X.todense()
    # print("writing to file...")
    # pickle.dump(X, open(savedir + '/X_by_lib.data', 'wb'))
    # pickle.dump(bag, open(os.path.join(savedir, 'BoW_by_lib.data'), 'wb'))
    # print("number of words: ", len(bag))
    

