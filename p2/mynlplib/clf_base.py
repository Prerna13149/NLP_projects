from mynlplib.constants import OFFSET
import numpy as np

import operator

# hint! use this.
def argmax(scores):
    items = list(scores.items())
    items.sort()
    return items[np.argmax([i[1] for i in items])][0]

# argmax = lambda x : max(x.items(),key=operator.itemgetter(1))[0]


# Deliverable 2.1 - can copy from P1
def make_feature_vector(base_features,label):
    """take a counter of base features and a label; return a dict of features, corresponding to f(x,y)

    :param base_features: counter of base features
    :param label: label string
    :returns: dict of features, f(x,y)
    :rtype: dict

    """
    ans_dict = {}
    for key in base_features:
      ans_dict[(label, key)] = base_features[key]
    ans_dict[(label, OFFSET)] = 1
    return ans_dict
    

# Deliverable 2.1 - can copy from P1
def predict(base_features,weights,labels):
    """prediction function

    :param base_features: a dictionary of base features and counts
    :param weights: a defaultdict of features and weights. features are tuples (label,base_feature).
    :param labels: a list of candidate labels
    :returns: top scoring label, scores of all labels
    :rtype: string, dict

    """
    res_dict ={}
    for lbl in labels:
      feature_vector = make_feature_vector(dict(base_features), lbl)
      #print(feature_vector)
      total = 0
      for feature, count in feature_vector.items():#base_features
        total += weights[feature] * count
      res_dict[lbl] = total
    max_val = argmax(res_dict)
    return max_val, res_dict
