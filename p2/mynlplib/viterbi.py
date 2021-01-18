import operator
from collections import defaultdict, Counter
from mynlplib.constants import START_TAG,END_TAG, UNK
import numpy as np
import torch
import torch.nn
from torch import autograd
from torch.autograd import Variable

def get_torch_variable(arr):
    # returns a pytorch variable of the array
    torch_var = torch.autograd.Variable(torch.from_numpy(np.array(arr).astype(np.float32)))
    return torch_var.view(1,-1)

def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


# Deliverable 3.3
def viterbi_step(all_tags, tag_to_ix, cur_tag_scores, transition_scores, prev_scores):
    """
    Calculates the best path score and corresponding back pointer for each tag for a word in the sentence in pytorch, which you will call from the main viterbi routine.
    
    parameters:
    - all_tags: list of all tags: includes both the START_TAG and END_TAG
    - tag_to_ix: a dictionary that maps each tag (including the START_TAG and the END_TAG) to a unique index.
    - cur_tag_scores: pytorch Variable that contains the local emission score for each tag for the current token in the sentence
                       it's size is : [ len(all_tags) ] 
    - transition_scores: pytorch Variable that contains the tag_transition_scores
                        it's size is : [ len(all_tags) x len(all_tags) ] 
    - prev_scores: pytorch Variable that contains the scores for each tag for the previous token in the sentence: 
                    it's size is : [ 1 x len(all_tags) ] 
    
    :returns:
    - viterbivars: a list of pytorch Variables such that each element contains the score for each tag in all_tags for the current token in the sentence
    - bptrs: a list of idx that contains the best_previous_tag for each tag in all_tags for the current token in the sentence
    """
    bptrs = []
    viterbivars=[]
    tagList = list(all_tags)
    for curr_tag in list(all_tags):
      tempArr = []
      for prev_tag in list(all_tags):
        if prev_tag == END_TAG or curr_tag == START_TAG:
          tempScore = torch.tensor(-np.inf)
        else:
          curr_idx = tag_to_ix[curr_tag]
          prev_idx = tag_to_ix[prev_tag]
          tempScore = prev_scores[0][prev_idx] + transition_scores[curr_idx][prev_idx] + cur_tag_scores[curr_idx]
        #print(tempScore)
        tempArr.append(tempScore)
      bptrs.append(tempArr.index(max(tempArr))) ## Storing the max index
      tmp_torch = torch.FloatTensor(tempArr)
      viterbivars.append(max(tempArr))
    return viterbivars, bptrs


# Deliverable 3.4
def build_trellis(all_tags, tag_to_ix, cur_tag_scores, transition_scores):
    """
    This function should compute the best_path and the path_score. 
    Use viterbi_step to implement build_trellis in viterbi.py in Pytorch.
    
    parameters:
    - all_tags: a list of all tags: includes START_TAG and END_TAG
    - tag_to_ix: a dictionary that maps each tag to a unique id.
    - cur_tag_scores: a list of pytorch Variables where each contains the local emission score for each tag for that particular token in the sentence, len(cur_tag_scores) will be equal to len(words)
                        it's size is : [ len(words in sequence) x len(all_tags) ] 
    - transition_scores: pytorch Variable (a matrix) that contains the tag_transition_scores
                        it's size is : [ len(all_tags) x len(all_tags) ] 
    
    :returns:
    - path_score: the score for the best_path
    - best_path: the actual best_path, which is the list of tags for each token: exclude the START_TAG and END_TAG here.
    
    Hint: Pay attention to the dimension of cur_tag_scores. It's slightly different from the one in viterbi_step().
    """
    ix_to_tag = { v:k for k,v in tag_to_ix.items() }
    
    # setting all the initial score to START_TAG
    # remember that END_TAG is in all_tags
    initial_vec = np.full((1, len(all_tags)),-np.inf)
    initial_vec[0][tag_to_ix[START_TAG]] = 0
    prev_scores = torch.autograd.Variable(torch.from_numpy(initial_vec.astype(np.float32))).view(1,-1)
    whole_bptrs = []

    ## Main loop
    for m in range(len(cur_tag_scores)):
      
      ## calling step
      viterbivars, bkptrs = viterbi_step(all_tags, tag_to_ix, cur_tag_scores[m], transition_scores, prev_scores)
      
      ## updating previous scores
      temp = [viterbivars[v].item() for v in range(len(viterbivars))]
      prev_scores = torch.autograd.Variable(torch.from_numpy(np.asarray([temp])))
      
      # updating backptrs
      # m = np.argmax(viterbivars)
      whole_bptrs.append(bkptrs)

    ## Termination step

    initial_vec2 = np.full((1,len(all_tags)),-np.inf)
    initial_vec2[0][tag_to_ix[END_TAG]] = 0
    prev_scores2 = torch.from_numpy(initial_vec2)
    temp = prev_scores2[0]

    viterbi_lastword, bkptrs_lastwrd = viterbi_step(all_tags, tag_to_ix, temp, transition_scores, prev_scores)
    t = np.argmax(viterbi_lastword)
    best_score =  viterbi_lastword[t]
    viterbi_last = ix_to_tag[bkptrs_lastwrd[t]]
    whole_bptrs.append(bkptrs_lastwrd)


    i = len(whole_bptrs)-1
    ## Creating path and calculating score
    path_score = best_score
    best_path = []
    #best_path.append(viterbi_last)
    #print(whole_bptrs)
    ## reverse[] python
    while(i != 0):
      best_path.append(ix_to_tag[whole_bptrs[i][t]])
      t = whole_bptrs[i][t]
      i = i - 1
    
    # Calculate the best_score and also the best_path using backpointers and don't forget to reverse the path
    best_path = best_path[::-1]
    
    return path_score, best_path
