import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise, next_batch_pairwise_k, sampler_dual, next_batch_pairwise2, sampler_single
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE, kssm_dict, SSM, SInfoNCE
import faiss
# paper: Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning. WWW'22
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
cuda_id = 0
torch.cuda.set_device(cuda_id)
class NCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(NCL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['NCL'])
        self.n_layers = int(args['-n_layer'])
        self.ssl_temp = float(args['-tau'])
        self.ssl_reg = float(args['-ssl_reg'])
        self.hyper_layers = int(args['-hyper_layers'])
        self.alpha = float(args['-alpha'])
        self.proto_reg = float(args['-proto_reg'])
        self.k = int(args['-num_clusters'])
        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers)
        self.user_centroids = None
        self.user_2cluster = None
        self.item_centroids = None
        self.item_2cluster = None

    def e_step(self):
        user_embeddings = self.model.embedding_dict['user_emb'].detach().cpu().numpy()
        item_embeddings = self.model.embedding_dict['item_emb'].detach().cpu().numpy()
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings)
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings)

    def run_kmeans(self, x):
        """Run K-means algorithm to get k clusters of the input tensor x        """
        kmeans = faiss.Kmeans(d=self.emb_size, k=self.k, gpu=cuda_id)
        kmeans.train(x)
        cluster_cents = kmeans.centroids
        _, I = kmeans.index.search(x, 1)
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(cluster_cents).cuda()
        node2cluster = torch.LongTensor(I).squeeze().cuda()
        return centroids, node2cluster

    def ProtoNCE_loss(self, initial_emb, user_idx, item_idx):
        user_emb, item_emb = torch.split(initial_emb, [self.data.user_num, self.data.item_num])
        user2cluster = self.user_2cluster[user_idx]
        user2centroids = self.user_centroids[user2cluster]
        proto_nce_loss_user = InfoNCE(user_emb[user_idx],user2centroids,self.ssl_temp) * self.batch_size
        item2cluster = self.item_2cluster[item_idx]
        item2centroids = self.item_centroids[item2cluster]
        proto_nce_loss_item = InfoNCE(item_emb[item_idx],item2centroids,self.ssl_temp) * self.batch_size
        proto_nce_loss = self.proto_reg * (proto_nce_loss_user + proto_nce_loss_item)
        return proto_nce_loss

    def ssl_layer_loss(self, context_emb, initial_emb, user, item):
        context_user_emb_all, context_item_emb_all = torch.split(context_emb, [self.data.user_num, self.data.item_num])
        initial_user_emb_all, initial_item_emb_all = torch.split(initial_emb, [self.data.user_num, self.data.item_num])
        context_user_emb = context_user_emb_all[user]
        initial_user_emb = initial_user_emb_all[user]
        norm_user_emb1 = F.normalize(context_user_emb)
        norm_user_emb2 = F.normalize(initial_user_emb)
        norm_all_user_emb = F.normalize(initial_user_emb_all)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        # ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        ttl_score_user = torch.matmul(norm_user_emb1, norm_user_emb1.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)
        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        context_item_emb = context_item_emb_all[item]
        initial_item_emb = initial_item_emb_all[item]
        norm_item_emb1 = F.normalize(context_item_emb)
        norm_item_emb2 = F.normalize(initial_item_emb)
        norm_all_item_emb = F.normalize(initial_item_emb_all)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        # ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        ttl_score_item = torch.matmul(norm_item_emb1, norm_item_emb1.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        ssl_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
        return ssl_loss

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            n_negs = 1024
            strategy = 'mns'
            rec_temp = 0.2
            rec_norm = True
            bpr = False
            print('strategy: ', strategy, 'n_negs: ', n_negs, 'rec_temp: ', rec_temp, 'rec_norm: ', rec_norm, 'bpr: ', bpr)
            self.e_step()
            for n, batch in enumerate(sampler_dual(self.data, self.batch_size, n_negs, strategy=strategy)):
                if strategy == 'mns':
                    if n_negs <= 1:
                        print('one negative sample not suitable for mns sampling strategy')
                        break
                    user_idx, pos_idx, neg_idx, freq_pos, freq_neg, neg_idx2 = batch
                    rec_user_emb, rec_item_emb, emb_list = model()
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
                    rec_user_emb, rec_item_emb, emb_list = model()
                    user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                    neg_item_idx = torch.Tensor(neg_idx).type(torch.long).view(-1, n_negs)
                    freq_pos = torch.Tensor(freq_pos).cuda()
                    freq_neg = torch.Tensor(freq_neg).view(-1, n_negs).cuda()
                    rec_loss = SSM(user_emb, rec_item_emb[pos_idx], rec_item_emb[neg_item_idx], rec_temp, rec_norm, freq_pos, freq_neg)
                elif strategy == 'inbatch':
                    user_idx, pos_idx, neg_idx, neg_idx2 = batch
                    rec_user_emb, rec_item_emb, emb_list = model()
                    user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                    neg_item_idx = torch.Tensor(neg_idx).type(torch.long).view(-1, n_negs)
                    if bpr:
                        neg_item_idx = neg_item_idx[:,0].view(-1,1)
                        rec_loss = SSM(user_emb, rec_item_emb[pos_idx], rec_item_emb[neg_item_idx], rec_temp, rec_norm)
                    else:
                        rec_loss = SSM(user_emb, rec_item_emb[pos_idx], rec_item_emb[neg_item_idx], rec_temp, rec_norm)
                elif strategy == 'random':
                    user_idx, pos_idx, neg_idx, neg_idx2 = batch
                    rec_user_emb, rec_item_emb, emb_list = model()
                    user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                    neg_item_idx = torch.Tensor(neg_idx).type(torch.long).view(-1, n_negs)
                    if bpr:
                        neg_item_idx = neg_item_idx[:,0].view(-1,1)
                        rec_loss = SSM(user_emb, rec_item_emb[pos_idx], rec_item_emb[neg_item_idx], rec_temp, rec_norm)
                    else:
                        rec_loss = SSM(user_emb, rec_item_emb[pos_idx], rec_item_emb[neg_item_idx], rec_temp, rec_norm)

                # rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                initial_emb = emb_list[0]
                context_emb = emb_list[self.hyper_layers*2]
                # ssl_loss = self.ssl_layer_loss(context_emb,initial_emb,user_idx,pos_idx)
                context_user_emb_all, context_item_emb_all = torch.split(context_emb, [self.data.user_num, self.data.item_num])
                initial_user_emb_all, initial_item_emb_all = torch.split(initial_emb, [self.data.user_num, self.data.item_num])
                ssl_loss_user = SInfoNCE(context_user_emb_all[user_idx], initial_user_emb_all[user_idx], initial_user_emb_all[neg_idx2], 0.2, True)
                ssl_loss_item = SInfoNCE(context_item_emb_all[pos_idx], initial_item_emb_all[pos_idx], initial_item_emb_all[neg_idx], 0.2, True)
                ssl_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
                warm_up_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb)/self.batch_size  + ssl_loss

                # Backward and optimize
                proto_loss = self.ProtoNCE_loss(initial_emb, user_idx, pos_idx)
                self.writer.add_scalars('NCL', {'rec_loss:':rec_loss.item(), 'ssl_loss:':ssl_loss.item(), 'proto_loss':proto_loss.item()}, epoch*self.batch_size+n)
                batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb) / self.batch_size + ssl_loss + proto_loss
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'ssl_loss', ssl_loss.item(), 'proto_loss', proto_loss.item())
            model.eval()
            with torch.no_grad():
                self.user_emb, self.item_emb, _ = model()
            if(self.fast_evaluation(epoch)):
                break
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb, _ = self.model()

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
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

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
        lgcn_all_embeddings = torch.stack(all_embeddings, dim=1)
        lgcn_all_embeddings = torch.mean(lgcn_all_embeddings, dim=1)
        user_all_embeddings = lgcn_all_embeddings[:self.data.user_num]
        item_all_embeddings = lgcn_all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings, all_embeddings