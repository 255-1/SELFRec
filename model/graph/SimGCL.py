import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise, next_batch_pairwise_k, sample_cl_negtive_idx
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE, bpr_k, kssm, kssm_p, kssm_dict
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
        print(self.get_parameter_number(model))
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            n_negs = 512
            for n, batch in enumerate(next_batch_pairwise_k(self.data, self.batch_size, n_negs=n_negs)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                # user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                # rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                # cl_loss = self.cl_rate * self.cal_cl_loss([user_idx,pos_idx])
                #kssm
                # neg_sample_idx = torch.Tensor(neg_idx).type(torch.long).view(-1, 1)
                # rec_loss = kssm(user_emb, pos_item_emb, rec_item_emb, neg_sample_idx)
                # cl_loss = self.cl_rate * self.cal_cl_loss_test([user_idx,pos_idx])
                # batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb, rec_item_emb[neg_sample_idx].view(-1, self.emb_size)) + cl_loss


                # #全部重写成我的模式
                # #rec
                # #anchor: user, pos: interaction item , neg: negative sample
                # #pos
                # rec_item_idx = torch.Tensor(pos_idx).type(torch.long).view(-1, 1)
                # #neg
                # neg_item_idx = torch.Tensor(neg_idx).type(torch.long).view(-1,n_negs)[:,0].view(-1,1) #为了对比测试, 先取一个neg
                # #anchor:user kssm_loss
                # rec_loss = kssm_p(rec_user_emb[user_idx], rec_item_emb, rec_item_idx, rec_item_emb, neg_item_idx)

                # user_view_1, item_view_1 = self.model(perturbed=True)
                # user_view_2, item_view_2 = self.model(perturbed=True)
                # #cl: user
                # u_pos_idx, u_neg_idx = sample_cl_negtive_idx(user_idx, len(user_idx))
                # user_cl_loss = kssm_p(user_view_1[u_pos_idx.view(-1)], user_view_2, u_pos_idx, user_view_2, u_neg_idx, temperature=0.2, normalized=True)
                # #cl: item
                # i_pos_idx, i_neg_idx = sample_cl_negtive_idx(pos_idx, len(user_idx))
                # item_cl_loss = kssm_p(item_view_1[i_pos_idx.view(-1)], item_view_2, i_pos_idx, item_view_2, i_neg_idx, temperature=0.2, normalized=True)
                # cl_loss = self.cl_rate * (user_cl_loss + item_cl_loss)
                # batch_loss =  rec_loss + l2_reg_loss(self.reg, rec_user_emb[user_idx], rec_item_emb[rec_item_idx].view(-1, self.emb_size), rec_item_emb[neg_item_idx].view(-1, self.emb_size)) + cl_loss

                user_view_1, item_view_1 = model(perturbed=True)
                user_view_2, item_view_2 = model(perturbed=True)
                #anchor: user, pos: interaction item + cl_inst_disc, neg: negative item + cl_inst_disc
                rec_item_idx = torch.Tensor(pos_idx).type(torch.long).view(-1, 1)
                neg_item_idx = torch.Tensor(neg_idx).type(torch.long).view(-1, n_negs)
                loss_1 = kssm_dict(rec_user_emb[user_idx], {rec_item_emb:rec_item_idx}
                                   ,{rec_item_emb: neg_item_idx}, [1],[False],[1])
                #anchor item, pos: interaction user + cl_inst_disc, neg: cl_inst_disc
                # user_idx = torch.Tensor(user_idx).type(torch.long).view(-1, 1)
                # i_pos_idx, i_neg_idx = sample_cl_negtive_idx(pos_idx, n_negs)
                u_pos_idx, u_neg_idx = sample_cl_negtive_idx(user_idx, n_negs)
                loss_2 = kssm_dict(item_view_1[pos_idx], {item_view_2: rec_item_idx}, {item_view_2: neg_item_idx}, [0.2],[True], [0.5])
                loss_3 = kssm_dict(user_view_1[user_idx], {user_view_2: u_pos_idx}, {user_view_2: u_neg_idx}, [0.2], [True], [0.5])
                l2_loss = l2_reg_loss(self.reg, rec_user_emb[user_idx], rec_item_emb[rec_item_idx].view(-1, self.emb_size), rec_item_emb[neg_item_idx].view(-1, self.emb_size))
                batch_loss = loss_1 + loss_2 + loss_3 +l2_loss
                optimizer.zero_grad()
                batch_loss.backward()
                for name, parms in model.named_parameters():
                    self.writer.add_histogram(name+'_grad', parms.grad, epoch*self.batch_size+n)
                    self.writer.add_histogram(name+'_data', parms, epoch*self.batch_size+n)
                optimizer.step()
                self.writer.add_scalars('SimGCL', {'loss_1_rec:':loss_1.item(), 'item_cl:':loss_2.item(), 'user_cl':loss_3.item()}, epoch*self.batch_size+n)
                if n % 100==0:
                    print('training:', epoch + 1, 'batch', n, 'loss_1_rec:', loss_1.item(), 'loss_2_item_cl:', loss_2.item(), 'loss_3_user_cl', loss_3.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            if(self.fast_evaluation(epoch)):
                break
        self.writer.flush()
        self.writer.close()
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def cal_cl_loss(self, idx):
        u_idx = idx[0]
        i_idx = idx[1]
        # u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        # i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
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

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
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
