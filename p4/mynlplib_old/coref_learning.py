import torch
from torch import nn
from torch import autograd as ag
import torch.nn.functional as F

from collections import defaultdict

from . import utils, coref

class FFCoref(nn.Module):
    '''
    A component that scores coreference relations based on a one-hot feature vector
    Architecture: input features -> Linear layer -> tanh -> Linear layer -> score
    '''
    
    ## deliverable 3.2
    def __init__(self, feat_names, hidden_dim):
        '''
        :param feat_names: list of keys to possible pairwise matching features
        :param hidden_dim: dimension of intermediate layer
        '''
        super(FFCoref, self).__init__()
        
        # STUDENT
        self.hidden_dim = hidden_dim
        self.linear_layer1 = nn.Linear(len(feat_names), hidden_dim)
        self.linear_layer2 = nn.Linear(hidden_dim, 1)
        self.feat_list = feat_names

        # END STUDENT
        
        
    ## deliverable 3.2
    def forward(self, features):
        '''
        :param features: defaultdict of pairwise matching features and their values for some pair
        :returns: model score
        :rtype: 1x1 torch Variable
        '''
        in_vec = torch.tensor([0.0 for i in range(len(self.feat_list))])
        for i, e in enumerate(self.feat_list):
          if e in features.keys():
            in_vec[i] = 1
        #print(input_vec)
        out1 = self.linear_layer1(in_vec)
        out2 = torch.tanh(out1)
        out3 = self.linear_layer2(out2)
        return out3

        
    ## deliverable 3.3
    def score_instance(self, markables, feats, i):
        '''
        A function scoring all coref candidates for a given markable
        Don't forget the new-entity option!
        :param markables: list of all markables in the document
        :param i: index of current markable
        :param feats: feature extraction function
        :returns: list of scores for all candidates
        :rtype: torch.FloatTensor of dimensions 1x(i+1)
        '''
        total = torch.tensor([0.0 for j in range(i+1)])
        for pos in range(i + 1):
          features = feats(markables, pos, i)
          score = self.forward(features)
          total[pos] = score
        return total.view(1, -1)          


    ## deliverable 3.4
    def instance_top_scores(self, markables, feats, i, true_antecedent):
        '''
        Find the top-scoring true and false candidates for i in the markable.
        If no false candidates exist, return (None, None).
        :param markables: list of all markables in the document
        :param i: index of current markable
        :param true_antecedent: gold label for markable
        :param feats: feature extraction function
        :returns trues_max: best-scoring true antecedent
        :returns false_max: best-scoring false antecedent
        '''
        if i == 0 or true_antecedent==0:
            return None, None
        else:
          only_false = True
          only_true = True
          scores = self.score_instance(markables, feats, i)
          #print(scores)
          true_ant_score=[]
          false_ant_score=[]
          for idx, score in enumerate(scores[0]):
            if idx == (scores.size()[1]-1):
              if only_false:
                true_ant_score.append(score)
              else:
                false_ant_score.append(score)
              break
            if markables[idx].entity == markables[i].entity:
              only_false = False
              true_ant_score.append(score)
            else:
              only_true = False
              false_ant_score.append(score)
          trues_max_val = torch.max(torch.stack(true_ant_score))
          false_max_val = torch.max(torch.stack(false_ant_score))
        ## 2nd im
        # if i == 0 or true_antecedent==0:
        #   return None, None
        # else:
        #   last_mark = markables[true_antecedent]
        #   scores = self.score_instance(markables, feats, i)
        #   all_trues_indices = torch.LongTensor([index for index in range(true_antecedent+1) if markables[index].entity == last_mark.entity])
        #   all_false_indices = torch.LongTensor([index for index in range(i+1) if markables[index].entity != last_mark.entity or index>true_antecedent])
        #   if true_antecedent==i:
        #     return 0, torch.max(scores[0,i])
        #   if all_trues_indices.shape[0] == i:
        #     return None, None
        #   trues_max_val = torch.max(scores[:,all_trues_indices])#()
        #   false_max_val = torch.max(scores[:,all_false_indices])#torch.max()
        #   return trues_max_val, false_max_val
        return trues_max_val, false_max_val
        


def train(model, optimizer, markable_set, feats, margin=1.0, epochs=2):
    _zero = ag.Variable(torch.Tensor([0])) # this var is reusable
    model.train()
    for i in range(epochs):
        tot_loss = 0.0
        instances = 0
        for doc in markable_set:
            true_ants = coref.get_true_antecedents(doc)
            for i in range(len(doc)):
                optimizer.zero_grad()
                max_t, max_f = model.instance_top_scores(doc, feats, i, true_ants[i])
                # print(max_t)
                # print(max_f)
                if max_t is None: continue
                unhinged_loss = -max_t + max_f + margin
                loss = F.relu(unhinged_loss)
                tot_loss += utils.to_scalar(loss)
                loss.backward()
                optimizer.step()
                instances += 1
        print(f'Loss = {tot_loss / instances}')
        
def evaluate(model, markable_set, feats):
    model.eval()
    coref.eval_on_dataset(make_resolver(feats, model), markable_set)
    
# helper
def make_resolver(features, model):
    return lambda markables : [utils.argmax(model.score_instance(markables, features, i)) for i in range(len(markables))]