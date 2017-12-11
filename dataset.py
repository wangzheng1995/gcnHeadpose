import utils.buildGraph as bg
import random


class Dataset(object):
    def __init__(self):
        self.feat = None
        self.lbl = None
        self.adj = None

        self.batch_idx = 0
        self.epoch_idx = 0

        self.feat, self.lbl, self.adj = bg.prepare()
        self.feat = [bg.preprocess_feat(f) for f in self.feat]
        model = 'GCN'
        if model == 'GCN':
            self.adj = [[bg.preprocess_adj(a)] for a in self.adj]

    def _shuffle(self, a, b, c):
        d = list(zip(a,b,c))
        random.shuffle(d)
        return zip(*d)
            
    def sample_batch(self, batch_size):
        feat = self.feat[self.batch_idx:self.batch_idx+batch_size]
        lbl = self.lbl[self.batch_idx:self.batch_idx+batch_size]
        adj = self.adj[self.batch_idx:self.batch_idx+batch_size]
        end_of_epoch = False
        if len(feat) < batch_size:  # end of epoch
            len_more = batch_size - len(feat)
            feat += self.feat[:len_more]
            lbl += self.lbl[:len_more]
            adj += self.adj[:len_more]
            self.epoch_idx += 1
            self.batch_idx = 0
            self.feat, self.lbl, self.adj = self._shuffle(self.feat, self.lbl, self.adj)
            end_of_epoch = True
        else:
            self.batch_idx += batch_size

        assert len(feat) == batch_size

        return end_of_epoch, feat, lbl, adj
