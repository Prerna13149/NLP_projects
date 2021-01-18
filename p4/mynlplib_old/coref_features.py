import itertools
from . import coref_rules
from collections import defaultdict

## deliverable 3.1
def minimal_features(markables,a,i):
    '''
    Compute a minimal set of features for antecedent a and mention i

    :param markables: list of markables for the document
    :param a: index of antecedent
    :param i: index of mention
    :returns: dict of features
    :rtype: defaultdict
    '''    
    f = defaultdict(float)
    if a == i:
      f['new-entity'] = 1
    else:
      if coref_rules.exact_match(markables[a], markables[i]):
        f['exact-match'] = 1
      if coref_rules.match_last_token(markables[a], markables[i]):
        f['last-token-match'] = 1
      if coref_rules.match_on_content(markables[a], markables[i]):
        f['content-match'] = 1
      if overlap(markables[a], markables[i]):
        f['crossover'] = 1
    return f

def overlap(m_a, m_i):
  m_a_range = list(range(m_a.start_token, m_a.end_token))
  m_i_range = list(range(m_i.start_token, m_i.end_token))
  over_lap = [value for value in m_a_range if value in m_i_range]
  if len(over_lap) == 0:
    return False
  return True


## deliverable 3.5
def distance_features(x,a,i,
                      max_mention_distance=5,
                      max_token_distance=10):
    '''
    compute a set of distance features for antecedent a and mention i

    :param x: markable list for document
    :param a: antecedent index
    :param i: mention index
    :param max_mention_distance: upper limit on mention distance
    :param max_token_distance: upper limit on token distance
    :returns: dict of features
    :rtype: defaultdict
    '''
    
    f = defaultdict(float)
    if a != i:
      mention_distance = min(i - a, max_mention_distance)
      token_distance = min(x[i].start_token - x[a].end_token, max_token_distance)
      f['mention-distance-' + str(mention_distance)] = 1
      f['token-distance-' + str(token_distance)] = 1
    return f

## deliverable 3.6
def make_feature_union(feat_func_list):
    '''
    return a feature function that is the union of the feature functions in the list

    :param feat_func_list: list of feature functions
    :returns: feature function
    :rtype: function
    '''
    def f_out(x, a, i):
        combinedFeatures = defaultdict(float)
        for feat_func in feat_func_list:
            features = feat_func(x, a, i)
            for feature in features:
                combinedFeatures[feature] = features[feature]
        return combinedFeatures
    return f_out


## deliverable 6
def make_bakeoff_features():
    raise NotImplementedError

