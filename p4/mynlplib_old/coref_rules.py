### Rule-based coreference resolution  ###########
# Lightly inspired by Stanford's "Multi-pass sieve"
# http://www.surdeanu.info/mihai/papers/emnlp10.pdf
# http://nlp.stanford.edu/pubs/conllst2011-coref.pdf

import nltk
from nltk.tag import pos_tag

# this may help
pronouns = ['i', 'me', 'mine', 'you', 'your', 'yours', 'she', 'her', 'hers'] +\
           ['he', 'him', 'his', 'it', 'its', 'they', 'them', 'their', 'theirs'] +\
           ['this', 'those', 'these', 'that', 'we', 'our', 'us', 'ours']
downcase_list = lambda toks : [tok.lower() for tok in toks]

############## Pairwise matchers #######################

def exact_match(m_a, m_i):
    '''
    return True if the strings are identical

    :param m_a: antecedent markable
    :param m_i: referent markable
    :returns: True if the strings are identical
    :rtype: boolean
    '''
    return downcase_list(m_a.string) == downcase_list(m_i.string)

# deliverable 2.2
def singleton_matcher(m_a, m_i):
    '''
    return value such that a document consists of only singleton entities

    :param m_a: antecedent markable
    :param m_i: referent markable
    :returns: 
    :rtype: boolean
    '''
    if m_a.start_token == m_i.start_token and m_a.end_token == m_i.end_token:
      return True
    return False


# deliverable 2.2
def full_cluster_matcher(m_a, m_i):
    '''
    return value such that a document consists of a single entity

    :param m_a: antecedent markable
    :param m_i: referent markable
    :returns: 
    :rtype: boolean
    '''
    return True


# deliverable 2.3
def exact_match_no_pronouns(m_a, m_i):
    '''
    return True if strings are identical and are not pronouns

    :param m_a: antecedent markable
    :param m_i: referent markable
    :returns: True if the strings are identical and are not pronouns
    :rtype: boolean
    '''
    match = exact_match(m_a, m_i)
    if match:
        if len(m_a.string) > 1:
            return True
        temp = (m_a.string)[0]
        if temp.lower() in pronouns:
            return False
        temp = (m_i.string)[0]
        if temp.lower() in pronouns:
            return False
        return True
    return False

# deliverable 2.4
def match_last_token(m_a, m_i):
    '''
    return True if final token of each markable is identical

    :param m_a: antecedent markable
    :param m_i: referent markable
    :rtype: boolean
    '''
    temp1 = (m_a.string)[-1]
    temp2 = (m_i.string)[-1]
    if temp1.lower() == temp2.lower():
      return True
    return False

# deliverable 2.5
def match_last_token_no_overlap(m_a, m_i):
    '''
    return True if last tokens are identical and there's no overlap

    :param m_a: antecedent markable
    :param m_i: referent markable
    :returns: True if final tokens match and strings do not overlap
    :rtype: boolean
    '''
    m_a_temp = list(range(m_a.start_token, m_a.end_token))
    m_i_temp = list(range(m_i.start_token, m_i.end_token))
    overlap = [value for value in m_a_temp if value in m_i_temp]
    if match_last_token(m_a, m_i):
      if len(overlap)==0:
        return True
    return False

# deliverable 2.6
def match_on_content(m_a, m_i):
    '''
    return True if all content words are identical and there's no overlap

    :param m_a: antecedent markable
    :param m_i: referent markable
    :returns: True if all match on all "content words" (defined by POS tag) and markables do not overlap
    :rtype: boolean
    '''
    k = 0
    content_words = ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$', 'JJ', 'JJR', 'JJS', 'CD']#['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
    m_a_words = [m_a.string[i] for i in range(len(m_a.string)) if m_a.tags[i] in content_words]
    k = k + len(m_a_words)
    m_i_words = [m_i.string[i] for i in range(len(m_i.string)) if m_i.tags[i] in content_words]
    k = k + len(m_i_words)
    m_a_content = downcase_list(m_a_words)
    m_i_content = downcase_list(m_i_words)
    k = k + 2
    if m_a_content!=m_i_content:
      return False
    m_a_temp = range(m_a.start_token, m_a.end_token)
    m_i_temp = range(m_i.start_token, m_i.end_token)
    m_a_list = list(m_a_temp)
    m_i_list = list(m_i_temp)
    overlap = [value for value in m_a_list if value in m_i_list]
    if len(overlap) == 0:
      return True
    return False
    
    
########## helper code

def most_recent_match(markables, matcher):
    '''
    given a list of markables and a pairwise matcher, return an antecedent list
    assumes markables are sorted

    :param markables: list of markables
    :param matcher: function that takes two markables, returns boolean if they are compatible
    :returns: list of antecedent indices
    :rtype: list
    '''
    antecedents = list(range(len(markables)))
    for i,m_i in enumerate(markables):
        for a,m_a in enumerate(markables[:i]):
            if matcher(m_a,m_i):
                antecedents[i] = a
    return antecedents

def make_resolver(pairwise_matcher):
    '''
    convert a pairwise markable matching function into a coreference resolution system, which generates antecedent lists

    :param pairwise_matcher: function from markable pairs to boolean
    :returns: function from markable list and word list to antecedent list
    :rtype: function

    The returned lambda expression takes a list of words and a list of markables.
    The words are ignored here. However, this function signature is needed because
    in other cases, we want to do some NLP on the words.
    '''
    return lambda markables : most_recent_match(markables, pairwise_matcher)
