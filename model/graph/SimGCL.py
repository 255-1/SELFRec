import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise, sample_cl_negtive_idx, sampler_single, sampler_dual
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE, bpr_k, kssm, kssm_p, kssm_dict, SSM, SInfoNCE
import numpy as np
# Paper: Are graph augmentations necessary? simple graph contrastive learning for recommendation. SIGIR'22
torch.cuda.set_device(2)
class SimGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(SimGCL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['SimGCL'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])
        self.model = SimGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers)

    def get_parameter_number(self,net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def train(self):
        model = self.model.cuda()
        self.writer.flush()
        print(self.get_parameter_number(model))
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            n_negs = 1024
            rec_temp = 0.2
            rec_norm = False
            strategy = 'inbatch'
            bpr = False
            print('strategy: ', strategy, 'n_negs: ', n_negs, 'rec_temp: ', rec_temp, 'rec_norm: ', rec_norm, 'bpr: ', bpr)
            for n, batch in enumerate(sampler_dual(self.data, self.batch_size, n_negs, strategy=strategy)):
                if strategy == 'mns':
                    if n_negs <= 1:
                        print('one negative sample not suitable for mns sampling strategy')
                        break
                    user_idx, pos_idx, neg_idx, freq_pos, freq_neg, neg_idx2 = batch
                    rec_user_emb, rec_item_emb = model()
                    user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                    neg_item_idx = torch.Tensor(neg_idx).type(torch.long).view(-1, n_negs)
                    freq_neg = torch.Tensor(freq_neg).view(-1, n_negs).cuda()
                    if bpr:
                        neg_item_idx = neg_item_idx[:,0].view(-1,1)
                        rec_loss = SSM(user_emb, rec_item_emb[pos_idx], rec_item_emb[neg_item_idx], rec_temp, rec_norm)
                    else:
                        rec_loss = SSM(user_emb, rec_item_emb[pos_idx], rec_item_emb[neg_item_idx], rec_temp, rec_norm, None, freq_neg)
                elif strategy == 'sbcnm':
                    user_idx, pos_idx, neg_idx, freq_pos, freq_neg, neg_idx2 = batch
                    rec_user_emb, rec_item_emb = model()
                    user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                    neg_item_idx = torch.Tensor(neg_idx).type(torch.long).view(-1, n_negs)
                    freq_pos = torch.Tensor(freq_pos).cuda()
                    freq_neg = torch.Tensor(freq_neg).view(-1, n_negs).cuda()
                    rec_loss = SSM(user_emb, rec_item_emb[pos_idx], rec_item_emb[neg_item_idx], rec_temp, rec_norm, freq_pos, freq_neg)
                elif strategy == 'inbatch':
                    user_idx, pos_idx, neg_idx, neg_idx2 = batch
                    rec_user_emb, rec_item_emb = model()
                    user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                    neg_item_idx = torch.Tensor(neg_idx).type(torch.long).view(-1, n_negs)
                    if bpr:
                        neg_item_idx = neg_item_idx[:,0].view(-1,1)
                        rec_loss = SSM(user_emb, rec_item_emb[pos_idx], rec_item_emb[neg_item_idx], rec_temp, rec_norm)
                    else:
                        rec_loss = SSM(user_emb, rec_item_emb[pos_idx], rec_item_emb[neg_item_idx], rec_temp, rec_norm)
                elif strategy == 'random':
                    user_idx, pos_idx, neg_idx, neg_idx2 = batch
                    rec_user_emb, rec_item_emb = model()
                    user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                    neg_item_idx = torch.Tensor(neg_idx).type(torch.long).view(-1, n_negs)
                    if bpr:
                        neg_item_idx = neg_item_idx[:,0].view(-1,1)
                        rec_loss = SSM(user_emb, rec_item_emb[pos_idx], rec_item_emb[neg_item_idx], rec_temp, rec_norm)
                    else:
                        rec_loss = SSM(user_emb, rec_item_emb[pos_idx], rec_item_emb[neg_item_idx], rec_temp, rec_norm)

                # cl_loss = self.cl_rate * model.cal_cl_loss([user_idx,pos_idx],dropped_adj1,dropped_adj2)
                user_view_1, item_view_1 = self.model(perturbed=True)
                user_view_2, item_view_2 = self.model(perturbed=True)
                user_cl_loss = SInfoNCE(user_view_1[user_idx], user_view_2[user_idx], user_view_2[neg_idx2], 0.2, True)
                item_cl_loss = SInfoNCE(item_view_1[pos_idx], item_view_2[pos_idx], item_view_2[neg_idx], 0.2, True)
                cl_loss = self.cl_rate * (user_cl_loss + item_cl_loss)
                batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb,neg_item_emb) + cl_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                self.writer.add_scalars('SGL', {'rec_loss:':rec_loss.item(), 'cl_loss:':cl_loss.item()}, epoch*self.batch_size+n)
                if n % 100==0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            if(self.fast_evaluation(epoch)):
                break
        self.writer.flush()
        self.writer.close()
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def cal_cl_loss(self, idx):
        # u_idx = idx[0]
        # i_idx = idx[1]
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.model(perturbed=True)
        user_view_2, item_view_2 = self.model(perturbed=True)
        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.2)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.2)
        return user_cl_loss + item_cl_loss

    def cal_cl_loss_test(self, idx):
        u_idx, u_neg_idx = sample_cl_negtive_idx(idx[0])
        i_idx, i_neg_idx = sample_cl_negtive_idx(idx[1])
        user_view_1, item_view_1 = self.model(perturbed=True)
        user_view_2, item_view_2 = self.model(perturbed=True)
        user_cl_loss = kssm(user_view_1[u_idx], user_view_2[u_idx], user_view_2, u_neg_idx, temperature=0.2, normalized=True)
        item_cl_loss = kssm(item_view_1[i_idx], item_view_2[i_idx], item_view_2, i_neg_idx, temperature=0.2, normalized=True)
        return user_cl_loss + item_cl_loss


    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class SimGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers):
        super(SimGCL_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        # self.a = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.b = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.a.data.fill_(0.5)
        # self.b.data.fill_(0.5)
        self.dropout = nn.Dropout(0.1)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        ego_embeddings = self.dropout(ego_embeddings)
        all_embeddings = []
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings)
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings
