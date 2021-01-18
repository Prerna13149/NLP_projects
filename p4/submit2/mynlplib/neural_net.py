import torch
import torch.nn as nn
import torch.autograd as ag
import torch.nn.functional as F

import mynlplib.utils as utils
import mynlplib.coref as coref

class BiLSTMWordEmbedding(nn.Module):
    '''
    In this component, you will use a Bi-Directional LSTM to get initial embeddings.
    The embedding for word i is the i'th hidden state of the LSTM
    after passing the sentence through the LSTM.
    '''

    ## deliverable 4.1
    def __init__(self, word_to_ix, word_embedding_dim, hidden_dim, num_layers, dropout):
        '''
        :param word_to_ix: dict mapping words to unique indices
        :param word_embedding_dim: the dimensionality of the input word embeddings
        :param hidden_dim: the dimensionality of the output embeddings that go to the classifier
        :param num_layers: the number of LSTM layers to use
        :param dropout: amount of dropout between LSTM layers (won't be activated if num_layers=1)
        '''
        super(BiLSTMWordEmbedding, self).__init__()
        self.word_to_ix = word_to_ix
        self.num_layers = num_layers
        self.word_embedding_dim = word_embedding_dim
        self.hidden_dim = hidden_dim
        self.use_cuda = False

        self.output_dim = hidden_dim

        # STUDENT
        # Construct the needed components in this order:
        # 1. An embedding lookup table
        # 2. The LSTM
        # Note we want the output dim to be hidden_dim, but since our LSTM
        # is bidirectional, we need to make the output of each direction hidden_dim/2
        # name your embedding member "word_embeddings"
        self.word_embeddings = nn.Embedding(len(self.word_to_ix), self.word_embedding_dim)
        self.lstm = nn.LSTM(self.word_embedding_dim, self.hidden_dim//2, self.num_layers, dropout=dropout,bidirectional=True,batch_first=True)
        

        
        # END STUDENT

        self.hidden = self.init_hidden()

    ## deliverable 4.1
    def forward(self, document):
        '''
        This function has several parts.
        1. Look up the embeddings for the words in the document.
           These will be the inputs to the LSTM sequence model.
           NOTE: At this step, rather than a list of embeddings, it should be a single tensor.
        2. Now that you have your tensor of embeddings, You can pass it through your LSTM.
        3. Convert the outputs into the correct return type, which is a list of lists of
           embeddings, each of shape (1, hidden_dim)
        NOTE: Make sure you are reassigning self.hidden to the new hidden state!
        :param document: a list of strs, the words of the document
        :returns: a list of embeddings for the document
        '''
        assert self.word_to_ix is not None, "ERROR: Make sure to set word_to_ix on \
                the embedding lookup components"
        temp = []
        k = 0
        for word in document:
          k = k + 1
          temp.append(self.word_to_ix[word])
        embed = self.word_embeddings(ag.Variable(torch.LongTensor(temp)))
        x = embed.expand(1, embed.size()[0],embed.size()[1])
        k = k + 1
        embed_out,hn = self.lstm(x,self.hidden)
        #self.hidden = hn
        n = embed_out.size()
        result = []
        for i in range(len(document)):
            k = k * 2
            temp = embed_out[0,i].expand(1,n[2])
            k = k/2
            result.append(temp)
        return result


    def init_hidden(self):
        '''
        PyTorch wants you to supply the last hidden state at each timestep
        to the LSTM.  You shouldn't need to call this function explicitly
        '''
        if self.use_cuda:
            return (ag.Variable(cuda.FloatTensor(self.num_layers * 2, 1, self.hidden_dim//2).zero_()),
                    ag.Variable(cuda.FloatTensor(self.num_layers * 2, 1, self.hidden_dim//2).zero_()))
        else:
            return (ag.Variable(torch.zeros(self.num_layers * 2, 1, self.hidden_dim//2)),
                    ag.Variable(torch.zeros(self.num_layers * 2, 1, self.hidden_dim//2)))

    def clear_hidden_state(self):
        self.hidden = self.init_hidden()

    def to_cuda(self):
        self.use_cuda = True
        self.cuda()

class AttentionBasedMarkableEmbedding(nn.Module):
    '''
    This class accepts embeddings from the entire document and a target markable.
    Its job is to produce a single embedding for that markable based on a trained attention component.
    '''
    
    ## deliverable 4.2
    def __init__(self, embedding_dim):
        '''
        :param embedding_dim: the embedding of inputs to be received,
            also to be used for the attention vector
        '''
        super(AttentionBasedMarkableEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.linear_layer = nn.Linear(self.embedding_dim, 1)
        

        
        # END STUDENT
        self.use_cuda = False
        
    ## deliverable 4.2
    def forward(self, embeddings, markable):
        '''
        :param embeddings: all embeddings for words in the document
        :param markable: the markable for which we want a weighted embedding
        :returns: attended embedding for markable (1d vector)
        '''
        k = 0
        span_embeddings = embeddings[markable.start_token: markable.end_token]
        k = k + 1
        out1 = torch.cat(span_embeddings)
        out2 = self.linear_layer(out1)
        k = k - 1
        out2 = F.softmax(out2, dim=0)
        return torch.sum(out2.mul(out1), dim=0)

        
    def to_cuda(self):
        self.use_cuda = True
        self.cuda()

class SequentialScorer(nn.Module):
    '''
    This class scores coreference between markables based on a concatenated embedding input
    Architecture: input embedding -> Linear layer -> ReLU -> Linear layer -> score
    '''
    
    ## deliverable 4.3
    def __init__(self, mark_embedding_dim, feat_set, feat_emb_dim, hidden_dim):
        '''
        :param mark_embedding_dim: dimension of markable embeddings
        :param feat_set: list of features expected to occur
        :param feat_emb_dim: dimension of boolean feature embeddings
        :param hidden_dim: dimension for intermediate representations
        '''
        super(SequentialScorer, self).__init__()
        
        self.feat_set = feat_set
        self.feat_to_idx = {feat:idx for idx, feat in enumerate(feat_set)}
        self.mark_embed_dim = mark_embedding_dim
        
        # STUDENT
        self.feat_emb_dim = feat_emb_dim
        # keep this order to pass tests
        self.feat_off_embs = nn.Embedding(num_embeddings=len(feat_set), embedding_dim=feat_emb_dim)
        self.feat_on_embs = nn.Embedding(num_embeddings=len(feat_set), embedding_dim=feat_emb_dim)
        self.linear1 = nn.Linear(2 * mark_embedding_dim + feat_emb_dim * len(feat_set), hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        
        # END STUDENT
        self.use_cuda = False
    
    ## deliverable 4.3
    def forward(self, emb_i, emb_a, pos_feats):
        '''
        :param emb_i: embedding for current markable
        :param emb_a: embedding for antecedent markable
        :param pos_feats: features with positive value
        :returns: score
        :rtype: 1x1 Variable
        '''
        emb_i, emb_a = emb_i.reshape((self.mark_embed_dim)), emb_a.reshape((self.mark_embed_dim))
        embed = [emb_i, emb_a]
        temp = pos_feats
        k = 0
        for feat, idx in self.feat_to_idx.items():
          if feat in temp:
            k = k + 1
            embed.append(self.feat_on_embs(torch.LongTensor([idx]))[0])
          else:
            k = k + 1
            embed.append(self.feat_off_embs(torch.LongTensor([idx]))[0])
        embed = torch.cat(embed,dim=0)
        k = k - 1
        out1 = self.linear1(embed)
        m = nn.ReLU()
        out2 = m(out1)
        out3 = self.linear2(out2)
        return out3
        
    ## deliverable 4.4
    def score_instance(self, doc_embs, markables, i, feats):
        '''
        A function scoring all coref candidates for a given markable
        Don't forget the new-entity option!
        :param doc_embs: embeddings for markables in the document
        :param markables: list of all markables in the document
        :param i: index of current markable
        :param feats: feature extraction function
        :returns: list of scores for all candidates
        :rtype: torch.FloatTensor of dimensions 1x(i+1)
        '''
        def get_pos_feats(markables, a, i):
            return [k for k,v in feats(markables, a, i).items() if v > 0]
        
        scores = ag.Variable(torch.FloatTensor(1, i + 1))
        k = 0

        for ant_i, p in enumerate(markables[:i + 1]):
            k = k + 1
            score_var = self.forward(doc_embs[i], doc_embs[ant_i], get_pos_feats(markables, ant_i, i))
            k = k - 1
            scores[0, ant_i] = score_var[0]
        return scores

        
    ## deliverable 4.4
    def instance_top_scores(self, doc_embs, markables, i, true_antecedent, feats):
        '''
        Find the top-scoring true and false candidates for i in the markable.
        If no false candidates exist, return (None, None).
        You can probably just copy this over from 'coref_learning.py'
        :param doc_embs: list of embeddings for all words in the document
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
          scores = self.score_instance(doc_embs, markables, i, feats)
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
          return trues_max_val, false_max_val

    def to_cuda(self):
        self.use_cuda = True
        self.cuda()

def train(doc_lstm_model, attn_model, scoring_model, optimizer, words_set, markable_set, feats, word_limit, epochs=2, margin=1.0, use_cuda=False):
    if not use_cuda:
        _zero = ag.Variable(torch.Tensor([0]))
    else:
        _zero = ag.Variable(torch.cuda.FloatTensor([0]))
        doc_lstm_model.to_cuda()
        attn_model.to_cuda()
        scoring_model.to_cuda()
    for ep in range(epochs):
        tot_loss = 0.0
        instances = 0
        doc_losses = []
        for words, marks in zip(words_set, markable_set):
            words = words[:word_limit]
            marks = [m for m in marks if m.end_token < word_limit]
            optimizer.zero_grad()
            doc_lstm_model.clear_hidden_state()

            if not use_cuda:
                loss = ag.Variable(torch.FloatTensor([0.0]))
            else:
                loss = ag.Variable(torch.cuda.FloatTensor([0.0]))

            base_embs = doc_lstm_model(words)
            att_embs = [attn_model(base_embs, m) for m in marks]
            true_ants = coref.get_true_antecedents(marks)
            for i in range(len(marks)):
                max_t, max_f = scoring_model.instance_top_scores(att_embs, marks, i, true_ants[i], feats)
                if max_t is None: continue

                if not use_cuda:
                    marg = ag.Variable(torch.Tensor([margin])) - max_t + max_f
                else:
                    marg = ag.Variable(torch.cuda.FloatTensor([margin])) - max_t + max_f

                loss += torch.max(torch.cat((_zero, marg)))
            instances += len(marks)
            sc_loss = utils.to_scalar(loss)
            tot_loss += sc_loss
            doc_losses.append(f'{sc_loss / len(marks):.5f}')
            loss.backward()
            optimizer.step()
        print(f'Epoch {ep+1} complete.\nDocument losses = {", ".join(doc_losses)}')
        print(f'Overall loss = {tot_loss / instances:.5f}')
        
def evaluate(doc_lstm_model, attn_model, scoring_model, words_set, markable_set, feats):
    doc_lstm_model.eval()
    attn_model.eval()
    scoring_model.eval()
    emb_dict = {} # for getting around matcher's signature
    for words, marks in zip(words_set, markable_set):
        doc_lstm_model.clear_hidden_state()
        base_embs = doc_lstm_model(words)
        att_embs = [attn_model(base_embs, m) for m in marks]
        emb_dict[marks[0].entity] = att_embs # don't try this at home.
    resolver = make_resolver(feats, emb_dict, scoring_model)
    coref.eval_on_dataset(resolver, markable_set)
    return resolver

# helper
def make_resolver(feats, emb_dict, scoring_model):
    return lambda markables : [utils.argmax(scoring_model.score_instance(emb_dict[markables[0].entity], markables, i, feats))\
                               for i in range(len(markables))]
