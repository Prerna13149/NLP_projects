from collections import defaultdict
from mynlplib.clf_base import predict,make_feature_vector

# deliverable 4.1
def perceptron_update(x,y,weights,labels):
    '''
    compute the perceptron update for a single instance

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param weights: a weight vector, represented as a dict
    :param labels: set of possible labels
    :returns: updates to weights, which should be added to weights
    :rtype: defaultdict

    '''
    ypred, scores = predict(x, weights, labels)
    orig_feature = make_feature_vector(x, y)
    if(ypred ==y):
      return defaultdict(float)
    else:
      pred_feature = make_feature_vector(x, ypred)
      res = {}
      for key in orig_feature.keys():
        if key in pred_feature.keys():
          res[key] = orig_feature[key]-pred_feature[key]
        else:
          res[key] = orig_feature[key]
      for key in pred_feature.keys():
        if key not in orig_feature.keys():
          res[key] = -pred_feature[key]
      output = defaultdict(float, res)
      return output

# deliverable 4.2
def estimate_perceptron(x,y,N_its):
    '''
    estimate perceptron weights for N_its iterations over the dataset (x,y)

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param N_its: number of iterations over the entire dataset
    :returns: weight dictionary
    :returns: list of weights dictionaries at each iteration
    :rtype: defaultdict, list

    '''

    labels = set(y)
    weights = defaultdict(float)
    weight_history = []
    for it in range(N_its):
        for x_i,y_i in zip(x,y):
            # YOUR CODE GOES HERE
            update_wt = perceptron_update(x_i, y_i, weights, labels)
            for k,v in update_wt.items():
              if (k in weights):
                weights[k] = weights[k] + update_wt[k]
              else:
                weights[k] = update_wt[k]
        weight_history.append(weights.copy())
    return weights, weight_history
