import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise, sampler_dual
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE, kssm_dict, SSM, SInfoNCE

# Paper: XSimGCL - Towards Extremely Simple Graph Contrastive Learning for Recommendation

torch.cuda.set_device(1)
class XSimGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(XSimGCL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['XSimGCL'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.temp = float(args['-tau'])
        self.n_layers = int(args['-n_layer'])
        self.layer_cl = int(args['-l*'])
        self.model = XSimGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers,self.layer_cl)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            n_negs = 1024
            rec_temp = 0.2
            rec_norm = True
            strategy = 'mns'
            bpr = False
            dual_sample = True
            print('strategy: ', strategy, 'n_negs: ', n_negs, 'rec_temp: ', rec_temp, 'rec_norm: ', rec_norm, 'bpr: ', bpr, 'dual_sample: ', dual_sample)
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
                rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb  = model(True)
                user_cl_loss = SInfoNCE(rec_user_emb[user_idx], cl_user_emb[user_idx], cl_user_emb[neg_idx2], self.temp, True)
                item_cl_loss = SInfoNCE(rec_item_emb[pos_idx], cl_item_emb[pos_idx], cl_item_emb[neg_idx], self.temp, True)
                cl_loss = self.cl_rate *(user_cl_loss + item_cl_loss)
                batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            if(self.fast_evaluation(epoch)):
                break
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def cal_cl_loss(self, idx, user_view1,user_view2,item_view1,item_view2):
        pass
        # u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        # i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        # user_cl_loss = InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temp)
        # item_cl_loss = InfoNCE(item_view1[i_idx], item_view2[i_idx], self.temp)
        # return user_cl_loss + item_cl_loss


    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class XSimGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers, layer_cl):
        super(XSimGCL_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.layer_cl = layer_cl
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

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
        all_embeddings_cl = ego_embeddings
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
            if k==self.layer_cl-1:
                all_embeddings_cl = ego_embeddings
        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.data.user_num, self.data.item_num])
        user_all_embeddings_cl, item_all_embeddings_cl = torch.split(all_embeddings_cl, [self.data.user_num, self.data.item_num])
        if perturbed:
            return user_all_embeddings, item_all_embeddings,user_all_embeddings_cl, item_all_embeddings_cl
        return user_all_embeddings, item_all_embeddings
