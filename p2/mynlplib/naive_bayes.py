from mynlplib.constants import OFFSET
from mynlplib import clf_base, evaluation, preproc

import numpy as np
from collections import defaultdict, Counter

def get_nb_weights(trainfile, smoothing):
    """
    estimate_nb function assumes that the labels are one for each document, where as in POS tagging: we have labels for 
    each particular token. So, in order to calculate the emission score weights: P(w|y) for a particular word and a 
    token, we slightly modify the input such that we consider each token and its tag to be a document and a label. 
    The following helper code converts the dataset to token level bag-of-words feature vector and labels. 
    The weights obtained from here will be used later as emission scores for the viterbi tagger.
    
    inputs: train_file: input file to obtain the nb_weights from
    smoothing: value of smoothing for the naive_bayes weights
    
    :returns: nb_weights: naive bayes weights
    """
    token_level_docs=[]
    token_level_tags=[]
    for words,tags in preproc.conll_seq_generator(trainfile):
        token_level_docs += [{word:1} for word in words]
        token_level_tags +=tags
    nb_weights = estimate_nb(token_level_docs, token_level_tags, smoothing)
    
    return nb_weights


# Can copy from P1
def get_corpus_counts(x,y,label):
    """
    Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """
    counts = Counter()
    for i in range(len(x)):
      if(y[i]==label):
        counts.update(x[i])
    #print(counts)
    res = defaultdict(int, counts)
    return res

    

# Can copy from P1
def estimate_pxy(x,y,label,smoothing,vocab):
    '''
    Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param smoothing: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word

    '''
    corpus_ct = get_corpus_counts(x, y, label)
    #print(corpus_ct)
    deno = 0;
    for word in vocab:
      deno = deno + corpus_ct[word]
    prob = 1
    logprob_dict={}
    for word in vocab:
      prob = np.log((corpus_ct[word] + smoothing)/(deno + smoothing*len(vocab)))
      logprob_dict[word] = prob
    res = defaultdict(float, logprob_dict)
    return res



# Can copy from P1
def estimate_nb(x,y,smoothing):
    """
    estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights
    :rtype: defaultdict 

    """
    count = Counter()
    for small_bow in x:
        count.update(small_bow)
    prob_labels={}
    vocab = []
    for ele in count:
      vocab.append(ele)
    #print(vocab)
    label_count = Counter(y)
    #print(label_count)
    labels = set(y)
    for lbl in labels:
      prob_labels[lbl] = np.log(label_count[lbl]/len(y))
    #print(prob_labels)
    weights=defaultdict(float)
    for lbl in labels:
      temp = estimate_pxy(x, y, lbl, smoothing, vocab)
      temp[OFFSET] = prob_labels[lbl]
      weights[lbl] = temp
    result = defaultdict(float)
    for lbl in weights.keys():
      for key2 in weights[lbl].keys():
        result[(lbl, key2)] = weights[lbl][key2]
    # result = defaultdict(float, weights)    
    return result

    

# Can copy from P1
def find_best_smoother(x_tr,y_tr,x_dv,y_dv,smoothers):
    '''
    find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param smoothers: list of smoothing values
    :returns: best smoothing value
    :rtype: float

    '''
    my_acc_dict = {}
    max_score = 0.0;
    for i in range(len(smoothers)):
      weights = estimate_nb(x_tr, y_tr, smoothers[i])
      y_hat = clf_base.predict_all(x_dv,weights,y_dv)
      acc = evaluation.acc(y_hat,y_dv)
      if( acc > max_score):
        max_score = acc
      my_acc_dict[smoothers[i]] = acc
    
    return max_score, my_acc_dict
