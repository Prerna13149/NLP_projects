from collections import Counter

import pandas as pd
import numpy as np

# deliverable 1.1
def bag_of_words(text):
    '''
    Count the number of word occurences for each document in the corpus

    :param text: a document, as a single string
    :returns: a Counter for a single document
    :rtype: Counter
    '''
    word_list = text.split()#[i for item in text for i in item.split()]
    output = Counter(word_list)
    return output
    #raise NotImplementedError

# deliverable 1.2
def aggregate_counts(bags_of_words):
    '''
    Aggregate word counts for individual documents into a single bag of words representation

    :param bags_of_words: a list of bags of words as Counters from the bag_of_words method
    :returns: an aggregated bag of words for the whole corpus
    :rtype: Counter
    '''

    counts = Counter()
    for small_bow in bags_of_words:
    	counts.update(small_bow)
    # YOUR CODE GOES HERE

    
    return counts

# deliverable 1.3
def compute_oov(bow1, bow2):
    '''
    Return a set of words that appears in bow1, but not bow2

    :param bow1: a bag of words
    :param bow2: a bag of words
    :returns: the set of words in bow1, but not in bow2
    :rtype: set
    '''
    bow1_dict = dict(Counter(bow1))
    bow2_dict = dict(Counter(bow2))
    final_dict = {x:bow1_dict[x] for x in bow1_dict if x not in bow2_dict}
    return set(final_dict.keys())
    #raise NotImplementedError

# deliverable 1.4
def prune_vocabulary(training_counts, target_data, min_counts):
    '''
    prune target_data to only words that appear at least min_counts times in training_counts

    :param training_counts: aggregated Counter for training data
    :param target_data: list of Counters containing dev bow's
    :returns: new list of Counters, with pruned vocabulary
    :returns: list of words in pruned vocabulary
    :rtype: list of Counters, set
    '''
    train_data_dict = dict(training_counts) 
    pruned_target = {x:train_data_dict[x] for x in train_data_dict if train_data_dict[x]>=min_counts}
    #print(final_prune)
    vocab = pruned_target.keys()
    #result_target = target_data
    result_target = []
    
    for mycounter in target_data:
      new_counter = Counter()
      for ele in mycounter:
        if ele in vocab:
          new_counter[ele] = mycounter[ele]
      result_target.append(new_counter)
    return result_target, vocab

# deliverable 5.1
def make_numpy(bags_of_words, vocab):
    '''
    Convert the bags of words into a 2D numpy array

    :param bags_of_words: list of Counters
    :param vocab: pruned vocabulary
    :returns: the bags of words as a matrix
    :rtype: numpy array
    '''
    vocab = sorted(vocab)
    vocab_size = len(vocab)
    bow_size = len(bags_of_words)
    X = np.zeros((bow_size,vocab_size))
    for bow_index in range(len(bags_of_words)):
      for voc_index in range(len(vocab)):
        word = vocab[voc_index]
        if word in bags_of_words[bow_index]:
          X[bow_index, voc_index] = bags_of_words[bow_index][word]
    return X


### helper code

def read_data(filename,label='Era',preprocessor=bag_of_words):
    df = pd.read_csv(filename)
    return df[label].values,[preprocessor(string) for string in df['Lyrics'].values]

def oov_rate(bow1,bow2):
    return len(compute_oov(bow1,bow2)) / len(bow1.keys())
