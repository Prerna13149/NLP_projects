from mynlplib.constants import OFFSET
import numpy as np
import torch

# deliverable 6.1
def get_top_features_for_label(weights,label,k=5):
    '''
    Return the five features with the highest weight for a given label.

    :param weights: the weight dictionary
    :param label: the label you are interested in 
    :returns: list of tuples of features and weights
    :rtype: list
    '''
    counts = {}
    for key in weights.keys():
      if key[0]==label:
        counts[key] = weights[key]

    temp = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return temp[:k]

# deliverable 6.2
def get_top_features_for_label_torch(model,vocab,label_set,label,k=5):
    '''
    Return the five words with the highest weight for a given label.

    :param model: PyTorch model
    :param vocab: vocabulary used when features were converted
    :param label_set: set of ordered labels
    :param label: the label you are interested in 
    :returns: list of words
    :rtype: list
    '''
    vocab = sorted(vocab)
    wt=[]
    for param in model.parameters():
      wt.append(param.data)
    t2 = wt[1].view(4, 1)
    myfinaltensor = torch.cat((wt[0], t2), dim=1)
    val = 0
    for e in range(len(label_set)):
      if label_set[e]==label:
        val = e
        break
    mytemptensor = myfinaltensor[val] 
    value, index = torch.topk(mytemptensor, k)
    ind = index.numpy()
    output_words =[]
    for i in ind:
      output_words.append(vocab[i])
    return output_words

# deliverable 7.1
def get_token_type_ratio(counts):
    '''
    compute the ratio of tokens to types

    :param counts: bag of words feature for a song, as a numpy array
    :returns: ratio of tokens to types
    :rtype: float

    '''
    my_unique = []
    for x in np.nditer(counts):
      if(x>0):
        my_unique.append(x)
    return counts.sum()/len(my_unique)

# deliverable 7.2
def concat_ttr_binned_features(data):
    '''
    Discretize your token-type ratio feature into bins.
    Then concatenate your result to the variable data

    :param data: Bag of words features (e.g. X_tr)
    :returns: Concatenated feature array [Nx(V+7)]
    :rtype: numpy array

    '''
    #print(data.shape)
    X = np.zeros((data.shape[0], 7))
    # X[0:data.shape[0], 0:data.shape[1]] = data
    # token_ratio = []
    for bow_index in range(data.shape[0]):
      value = get_token_type_ratio(data[bow_index])
      # token_ratio.append(value)
      if(value>=0 and value<1):
        X[bow_index, 0] = 1
      elif(value>=1 and value<2):
        X[bow_index, 1] = 1
      elif(value>=2 and value<3):
        X[bow_index, 2] = 1
      elif(value>=3 and value<4):
        X[bow_index, 3] = 1
      elif(value>=4 and value<5):
        X[bow_index, 4] = 1
      elif(value>=5 and value<6):
        X[bow_index, 5] = 1
      else:
        X[bow_index, 6] = 1
    #print(X.shape)
    res = np.concatenate((data,X),axis=1)
    return res
