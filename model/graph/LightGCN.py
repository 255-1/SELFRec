import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise, next_batch_pairwise_k, sample_cl_negtive_idx, next_batch_pairwise_inbatch
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss,l2_reg_loss, bpr_k, kssm, kssm_p, kssm_dict, SSSM
import torch.nn.functional as F
import time
# paper: LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR'20

cuda_id=1
class LightGCN(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(LightGCN, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['LightGCN'])
        self.n_layers = int(args['-n_layer'])
        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers)

    def train(self):
        model = self.model.cuda(cuda_id)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            start = time.time()
            for n, batch in enumerate(next_batch_pairwise_inbatch(self.data, self.batch_size)):
            # for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx, n_negs = batch
                # user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                neg_sample_idx = torch.Tensor(neg_idx).type(torch.long).view(-1, n_negs) #n_negs
                bpr_loss = SSSM(rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_sample_idx], 0.2, False)
                l2_loss = l2_reg_loss(self.reg, user_emb,rec_item_emb[pos_idx],rec_item_emb[neg_sample_idx].view(-1, self.emb_size))/self.batch_size
                batch_loss = bpr_loss+l2_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = model()
            if(self.fast_evaluation(epoch)):
                break;
            end = time.time()
            print(start-end)
        self.writer.flush()
        self.writer.close()
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda(cuda_id)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings


