from mynlplib.preproc import conll_seq_generator
from mynlplib.constants import START_TAG, END_TAG, OFFSET, UNK
from mynlplib import naive_bayes, most_common 
import numpy as np
from collections import defaultdict
import torch
import torch.nn
from torch.autograd import Variable


# Deliverable 4.2
def compute_transition_weights(trans_counts, smoothing):
    """
    Compute the HMM transition weights, given the counts.
    Don't forget to assign smoothed probabilities to transitions which
    do not appear in the counts.
    
    This will also affect your computation of the denominator.

    :param trans_counts: counts, generated from most_common.get_tag_trans_counts
    :param smoothing: additive smoothing
    :returns: dict of features [(curr_tag,prev_tag)] and weights

    """
    
    weights = defaultdict(float)
    
    all_tags = list(trans_counts.keys())+ [END_TAG]
    print(all_tags)
    
    for tag1 in all_tags:
      for tag2 in all_tags:
        if tag2 == START_TAG or tag1 == END_TAG:
          weights[(tag2, tag1)] = -np.inf
        else:
          num = np.log(smoothing + trans_counts[tag1][tag2])
          deno = np.log((len(all_tags) - 1) * smoothing + sum(trans_counts[tag1].values()))
          weights[(tag2, tag1)] = num - deno#np.log(smoothing + trans_counts[tag1][tag2]) - np.log((len(all_tags) - 1) * smoothing + sum(trans_counts[tag1].values()))
    return weights


# Deliverable 3.2
def compute_weights_variables(nb_weights, hmm_trans_weights, vocab, word_to_ix, tag_to_ix):
    """
    Computes autograd Variables of two weights: emission_probabilities and the tag_transition_probabilties
    parameters:
    nb_weights: -- a dictionary of emission weights
    hmm_trans_weights: -- dictionary of tag transition weights
    vocab: -- list of all the words
    word_to_ix: -- a dictionary that maps each word in the vocab to a unique index
    tag_to_ix: -- a dictionary that maps each tag (including the START_TAG and the END_TAG) to a unique index.
    
    :returns:
    emission_probs_vr: torch Variable of a matrix of size Vocab x Tagset_size
    tag_transition_probs_vr: torch Variable of a matrix of size Tagset_size x Tagset_size
    :rtype: autograd Variables of the the weights
    """
    # Assume that tag_to_ix includes both START_TAG and END_TAG
    
    # Assume that tag_to_ix includes both START_TAG and END_TAG
    ix_to_tag = {v:k for k,v in tag_to_ix.items()}
    tag_transition_probs = np.full((len(tag_to_ix), len(tag_to_ix)), -np.inf)
    emission_probs = np.full((len(vocab),len(tag_to_ix)), 0.0)
    for idx1 in range(len(ix_to_tag)):
      for idx2 in range(len(ix_to_tag)):
        val1 = ix_to_tag[idx1]
        val2 = ix_to_tag[idx2]
        if (val1, val2) in hmm_trans_weights.keys():
          tag_transition_probs[idx1][idx2] = hmm_trans_weights[(val1, val2)]
        else:
          tag_transition_probs[idx1][idx2] = -np.inf
    
    for idx1 in range(len(vocab)):
      for idx2 in range(len(ix_to_tag)):
        val1 = vocab[idx1]
        val2 = ix_to_tag[idx2]
        #print(val1)
        if val2 == '--END--' or val2 == '--START--':
          emission_probs[idx1][idx2] = -np.inf
        elif (val2, val1) in nb_weights.keys():
          emission_probs[idx1][idx2] = nb_weights[(val2, val1)]
        else:
          emission_probs[idx1][idx2] = 0
  
    
    
    
    emission_probs_vr = Variable(torch.from_numpy(emission_probs.astype(np.float32)))
    tag_transition_probs_vr = Variable(torch.from_numpy(tag_transition_probs.astype(np.float32)))
    
    return emission_probs_vr, tag_transition_probs_vr
