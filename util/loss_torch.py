import torch
import torch.nn.functional as F
import torch.nn as nn
import random

def bpr_k(user_emb, pos_item_emb, k_neg_item_emb):
    """bpr+ k negative_item

    Args:
        user_emb (_type_): (bsz, emb)
        pos_item_emb (_type_): (bsz, emb)
        k_neg_item_emb (_type_): (k*bsz , emb)
        TODO: k_neg_item_emb不是直接传入, 而是用embedding + idx方式进来,否则太占显存

    Returns:
        _type_: _description_
    """
    #(bsz, k, emb)
    k_neg_item_emb = k_neg_item_emb.view(pos_item_emb.shape[0], -1, pos_item_emb.shape[1])
    pos_score = (user_emb * pos_item_emb).sum(dim=-1)
    pos_score = torch.exp(pos_score)
    ttl_score = pos_score
    for i in range(k_neg_item_emb.shape[1]):
        neg_score = (user_emb * k_neg_item_emb[:,i,:]).sum(dim=-1)
        ttl_score = ttl_score + torch.exp(neg_score)
    cl_loss = -torch.log(10e-8 + pos_score / ttl_score)
    return torch.mean(cl_loss)

def kssm3(anchor_emb, pos_emb, neg_all_emb, neg_sample_idx, temperature=1, normalized=False):
    """bpr_k

    Args:
        anchor_emb (_type_): (bsz, emb)
        pos_emb (_type_): (bsz, emb)
        neg_all_emb (_type_): (bsz, emb)
        neg_sample_idx (_type_): (bsz, k) k is the number of sample strategy
    """
    if normalized:
        anchor_emb = F.normalize(anchor_emb, dim=1)
        pos_emb = F.normalize(pos_emb, dim=1)
        neg_all_emb = F.normalize(neg_all_emb, dim=1)
    pos_score = (anchor_emb * pos_emb).sum(dim=-1)
    pos_score = torch.exp(pos_score/temperature)
    ttl_score = pos_score
    for i in range(neg_sample_idx.shape[1]):
        neg_score = (anchor_emb * neg_all_emb[neg_sample_idx[:,i]]).sum(dim=-1)
        ttl_score = ttl_score + torch.exp(neg_score/temperature)
    cl_loss = -torch.log(10e-8 + pos_score / ttl_score)
    return torch.mean(cl_loss)

def kssm2(anchor_emb, pos_emb, neg_all_emb, neg_sample_idx, temperature=1, normalized=False):
    """bpr_k

    Args:
        anchor_emb (_type_): (bsz, emb)
        pos_emb (_type_): (bsz, emb)
        neg_all_emb (_type_): (bsz, emb)
        neg_sample_idx (_type_): (bsz, k) k is the number of sample strategy
    """
    if normalized:
        anchor_emb = F.normalize(anchor_emb, dim=1)
        pos_emb = F.normalize(pos_emb, dim=1)
        neg_all_emb = F.normalize(neg_all_emb, dim=1)
    pos_score = (anchor_emb * pos_emb).sum(dim=-1)
    pos_score = torch.exp(pos_score/temperature)
    ttl_score = pos_score
    neg_emb = neg_all_emb[neg_sample_idx.t()] #(k, bsz, emb)
    neg_score = (neg_emb * anchor_emb).sum(dim=-1) #(k, bsz)
    neg_score = torch.exp(neg_score/temperature).sum(dim=0)
    ttl_score = ttl_score + neg_score
    cl_loss = -torch.log(10e-8 + pos_score / ttl_score)
    return torch.mean(cl_loss)

def kssm(anchor_emb, pos_emb, neg_all_emb, neg_sample_idx, temperature=1, normalized=False):
    """bpr_k

    Args:
        anchor_emb (_type_): (bsz, emb)
        pos_emb (_type_): (bsz, emb)
        neg_all_emb (_type_): (bsz, emb)
        neg_sample_idx (_type_): (bsz, k) k is the number of sample strategy
    """
    if normalized:
        anchor_emb = F.normalize(anchor_emb, dim=1)
        pos_emb = F.normalize(pos_emb, dim=1)
        neg_all_emb = F.normalize(neg_all_emb, dim=1)
    pos_score = (anchor_emb * pos_emb).sum(dim=-1)
    pos_score = torch.exp(pos_score/temperature)
    # print("kssm pos_score: ", pos_score)
    ttl_score = pos_score
    all_neg_score = torch.Tensor().to(neg_all_emb.device) #(k,n)
    split_size = neg_sample_idx.t().shape[0]//10 if neg_sample_idx.t().shape[0]//10 !=0 else 1
    for sub_idx in neg_sample_idx.t().split(split_size, dim=0): #防止k太大导致显存不够, 对比学习k=bsz-1
        neg_score = (neg_all_emb[sub_idx] * anchor_emb).sum(dim=-1) #(sub_k, bsz)
        neg_score = torch.exp(neg_score/temperature)
        all_neg_score = torch.cat((all_neg_score, neg_score), dim=0)
    ttl_score = ttl_score + all_neg_score.sum(dim=0)
    # print("kssm ttl_scoree: ", ttl_score)
    cl_loss = -torch.log(10e-8 + pos_score / ttl_score)
    return torch.mean(cl_loss)

def kssm_p(anchor_emb, pos_all_emb, pos_sample_idx, neg_all_emb, neg_sample_idx, temperature=1, normalized=False, cos_sim=False):
    if normalized:
        anchor_emb = F.normalize(anchor_emb, dim=1)
        pos_all_emb = F.normalize(pos_all_emb, dim=1)
        neg_all_emb = F.normalize(neg_all_emb, dim=1)
    if cos_sim:
        cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        pos_score = cos(pos_all_emb[pos_sample_idx.t()], anchor_emb)
        pos_score = torch.exp(pos_score/temperature).sum(dim=0)
    else:
        pos_score = (pos_all_emb[pos_sample_idx.t()] * anchor_emb).sum(dim=-1)
        pos_score = torch.exp(pos_score/temperature).sum(dim=0)
    # print("kssm_p pos_score: ", pos_score)
    ttl_score = pos_score
    all_neg_score = torch.Tensor().to(neg_all_emb.device) #(k,n)
    split_size = neg_sample_idx.t().shape[0]//10 if neg_sample_idx.t().shape[0]//10 !=0 else 1
    for sub_idx in neg_sample_idx.t().split(split_size, dim=0): #防止k太大导致显存不够
        if cos_sim:
            cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
            neg_score = cos(neg_all_emb[sub_idx], anchor_emb) #(sub_k, bsz)
            neg_score = torch.exp(neg_score/temperature)
        else:
            neg_score = (neg_all_emb[sub_idx] * anchor_emb).sum(dim=-1) #(sub_k, bsz)
            neg_score = torch.exp(neg_score/temperature)
        all_neg_score = torch.cat((all_neg_score, neg_score), dim=0)
    ttl_score = ttl_score + all_neg_score.sum(dim=0)
    # print("kssm_p ttl_score: ", ttl_score)
    cl_loss = -torch.log(10e-8 + pos_score / ttl_score)
    return torch.mean(cl_loss)

def kssm_dict(anchor_emb, pos_dict, neg_dict, temperature, normalized, rate):
    """_summary_

    Args:
        anchor_emb (_type_): (bsz, emb)
        pos_dict (_type_): {all_pos_embedding: pos_idx} = {(N,emb) : (bsz, 1)}
        neg_dict (_type_): {all_neg_embedding: neg_idx} = {(N,emb) : (bsz, k)}
        temperature (_type_): list[] each strategy's temperature
        normalized (_type_): list[] each strategy's normalized (True, False)

    Returns:
        _type_: result
    """

    pos_score_list = []
    idx = 0
    for pos_all_emb, pos_sample_idx in pos_dict.items():
        pos_score_list.append(p_score(anchor_emb, pos_all_emb, pos_sample_idx, temperature[idx], normalized[idx]))
        idx += 1
    neg_score_list = []
    idx = 0
    for neg_all_emb, neg_sample_idx in neg_dict.items():
        neg_score_list.append(n_score(anchor_emb, neg_all_emb, neg_sample_idx, temperature[idx], normalized[idx]))
        idx += 1
    assert len(pos_score_list) == len(neg_score_list)
    ttl_score_list = []
    for i in range(len(pos_score_list)):
        ttl_score_list.append(pos_score_list[i]+neg_score_list[i])
    pos_score = torch.ones(anchor_emb.shape[0]).to(anchor_emb.device)
    ttl_score = torch.ones(anchor_emb.shape[0]).to(anchor_emb.device)
    for i, p in enumerate(pos_score_list):
        pos_score = pos_score * torch.pow(p, rate[i])
    for i, ttl in enumerate(ttl_score_list):
        ttl_score = ttl_score * torch.pow(ttl, rate[i])

    if rate:
        kssm_loss = -torch.log(10e-8 + pos_score / ttl_score)
    else:
        kssm_loss = -torch.log(10e-8 + pos_score / ttl_score)
    return torch.mean(kssm_loss)

def p_score(anchor_emb, pos_all_emb, pos_sample_idx, temperature=1, normalized=False):
    """numerator

    Args:
        anchor_emb (_type_): (bsz, emb)
        pos_all_emb (_type_): (N, emb)
        pos_sample_idx (_type_): (bsz, k)
        temperature (int, optional): _description_. Defaults to 1.
        normalized (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if normalized:
        anchor_emb = F.normalize(anchor_emb, dim=1)
        pos_all_emb = F.normalize(pos_all_emb, dim=1)
    pos_score = (pos_all_emb[pos_sample_idx.t()] * anchor_emb).sum(dim=-1) #((k, bsz, emb) * (bsz, emb)).sum(dim=-1)=>(k, bsz)
    pos_score = torch.exp(pos_score/temperature).sum(dim=0)#(k,bsz).sum(dim=0) => (bsz)
    return pos_score

def n_score(anchor_emb, neg_all_emb, neg_sample_idx, temperature=1, normalized=False):
    """denominator

    Args:
        anchor_emb (_type_): (bsz, emb)
        neg_all_emb (_type_): (N, emb)
        neg_sample_idx (_type_): (bsz, k)
        temperature (int, optional): _description_. Defaults to 1.
        normalized (bool, optional): _description_. Defaults to False.
    """
    if normalized:
        anchor_emb = F.normalize(anchor_emb, dim=1)
        neg_all_emb = F.normalize(neg_all_emb, dim=1)
    all_neg_score = torch.Tensor().to(neg_all_emb.device)
    split_size = neg_sample_idx.t().shape[0]//4 if neg_sample_idx.t().shape[0]//4 !=0 else 1
    for sub_idx in neg_sample_idx.t().split(split_size, dim=0): #防止k太大导致显存不够
        neg_score = (neg_all_emb[sub_idx] * anchor_emb).sum(dim=-1) #(sub_k, bsz)
        neg_score = torch.exp(neg_score/temperature)
        all_neg_score = torch.cat((all_neg_score, neg_score), dim=0)
    return all_neg_score.sum(dim=0)

def kssm_dict_bp(anchor_emb, pos_dict, neg_dict, temperature, normalized):
    """_summary_

    Args:
        anchor_emb (_type_): (bsz, emb)
        pos_dict (_type_): {all_pos_embedding: pos_idx} = {(N,emb) : (bsz, 1)}
        neg_dict (_type_): {all_neg_embedding: neg_idx} = {(N,emb) : (bsz, k)}
        temperature (_type_): list[] each strategy's temperature
        normalized (_type_): list[] each strategy's normalized (True, False)

    Returns:
        _type_: result
    """

    pos_score = torch.ones(anchor_emb.shape[0]).to(anchor_emb.device)
    idx = 0
    for pos_all_emb, pos_sample_idx in pos_dict.items():
        pos_score = pos_score * p_score(anchor_emb, pos_all_emb, pos_sample_idx, temperature[idx], normalized[idx])
        idx += 1
    all_neg_score = torch.ones(anchor_emb.shape[0]).to(anchor_emb.device)
    idx = 0
    for neg_all_emb, neg_sample_idx in neg_dict.items():
        all_neg_score = all_neg_score * n_score(anchor_emb, neg_all_emb, neg_sample_idx, temperature[idx], normalized[idx])
        idx += 1
    ttl_score = pos_score+all_neg_score
    cl_loss = -torch.log(10e-8 + pos_score / ttl_score)
    return torch.mean(cl_loss)


def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = -torch.log(10e-8 + torch.sigmoid(pos_score - neg_score))
    return torch.mean(loss)

def triplet_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = F.relu(neg_score+1-pos_score)
    return torch.mean(loss)

def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb_loss += torch.norm(emb, p=2)
    return emb_loss * reg


def batch_softmax_loss(user_emb, item_emb, temperature):
    user_emb, item_emb = F.normalize(user_emb, dim=1), F.normalize(item_emb, dim=1)
    pos_score = (user_emb * item_emb).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(user_emb, item_emb.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    loss = -torch.log(pos_score / ttl_score)
    return torch.mean(loss)


def InfoNCE(view1, view2, temperature):
    view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    # print("InfoNCE pos_score: ", pos_score)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    # print("InfoNCE ttl_score: ", ttl_score)
    cl_loss = -torch.log(10e-8 + pos_score / ttl_score)
    return torch.mean(cl_loss)

def SSSM(anchor, pos, neg, temperature, normalize=False):
    if normalize:
        anchor, pos, neg = F.normalize(anchor, dim=-1), F.normalize(pos, dim=-1), F.normalize(neg, dim=-1)
    pos_score = (anchor * pos).sum(dim=-1)
    pos_score = torch.exp(pos_score/temperature)
    ttl_score = pos_score
    neg_score = (anchor.unsqueeze(dim=1) * neg).sum(dim=-1)
    neg_score = torch.exp(neg_score/temperature).sum(dim=-1)
    ttl_score = ttl_score + neg_score
    cl_loss = -torch.log(10e-8 + pos_score / ttl_score)
    return torch.mean(cl_loss)



def kl_divergence(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1) - F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(kl)

def js_divergence(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    q = F.softmax(q_logit, dim=-1)
    kl_p = torch.sum(p * (F.log_softmax(p_logit, dim=-1) - F.log_softmax(q_logit, dim=-1)), 1)
    kl_q = torch.sum(q * (F.log_softmax(q_logit, dim=-1) - F.log_softmax(p_logit, dim=-1)), 1)
    return torch.mean(kl_p+kl_q)

def sample_cl_negtive_idx(idx, sample_num):
    if sample_num > len(idx):
        sample_num = len(idx)
    u_idx = torch.unique(torch.Tensor(idx).type(torch.long))
    neg_sample_idx = (u_idx[1:])[:sample_num].unsqueeze(dim=0)
    for i in range(1, len(u_idx)):
        neg_idx = u_idx[:i]
        neg_idx = torch.cat((neg_idx, u_idx[(i+1):]))[:sample_num].unsqueeze(dim=0)
        neg_sample_idx = torch.cat((neg_sample_idx, neg_idx), dim=0)
    return u_idx.view(-1,1), neg_sample_idx

if __name__=="__main__":
    initializer = nn.init.xavier_uniform_
    idx = list(range(2048))
    u_idx, neg_sample_idx = sample_cl_negtive_idx(idx, 2048)
    tau = 0.2

    #测试kssm_dict分开写是否结果一致
    count = 0.0
    for i  in range(600):
        a = torch.rand(2048, 16)
        p = torch.rand(6666, 16)
        neg = torch.rand(8888, 16)

        # res1 = kssm_dict(a, {p: u_idx}, {neg: neg_sample_idx}, [1], [False],[1])
        res1 = kssm2(a,p[idx],neg,neg_sample_idx,1,False)
        # res2 = kssm_p(a, p, u_idx, neg, neg_sample_idx)
        res2 = SSSM(a, p[idx], neg[neg_sample_idx], 1, False)
        if res1 != res2:
            print(res1-res2)
            count += (res1-res2)
    print(count)

    # 测试个体判别
    # a = initializer(torch.empty(2048, 64))
    # res1 = InfoNCE(a,a,tau)
    # # res2 = kssm(a[u_idx],a[u_idx],a, neg_sample_idx, tau, normalized=True)
    # res2 = SSSM(a,a[idx], a[neg_sample_idx], 0.2, True)
    # print(res1 == res2)

    # 测试对比学习不同矩阵
    # view_1 = initializer(torch.empty(2048, 64)).cuda()
    # view_2 = initializer(torch.empty(2048, 64)).cuda()
    # res1 = InfoNCE(view_1,view_2,tau)
    # pos_sample_idx = torch.Tensor(idx).type(torch.long).view(-1, 1)
    # res2 = kssm_p(view_1[u_idx], view_2, pos_sample_idx, view_2, neg_sample_idx, tau, normalized=True)
    # res3 = kssm(view_1[u_idx],view_2[u_idx],view_2, neg_sample_idx, tau, normalized=True)
    # print(res1 == res2, res1-res2,res1, res2)
    # print(res2 == res3, )

    # 多次测试
    # count = 0.0
    # for i in range(600):
    #     a = initializer(torch.empty(2048, 64)).cuda()
    #     b = initializer(torch.empty(2048, 64)).cuda()
    #     res1 = InfoNCE(a,b,tau)
    #     res2 = kssm(a[u_idx],b[u_idx],b, neg_sample_idx, tau, normalized=True)
    #     if res1 != res2 :
    #         print(res1-res2)
    #         count += (res1-res2).item()
    #     # else:
    #     #     print(res1 == res2)
    # print("count = ", count)

    # count = 0.0
    # for i in range(600):
    #     user_emb = torch.rand(2048, 64).cuda()
    #     inter_emb = torch.rand(2048, 64).cuda()
    #     neg_emb = torch.rand(2048, 64).cuda()
    #     neg_idx = list(range(2048))
    #     neg_sample_idx = torch.Tensor(neg_idx).type(torch.long).view(-1, 1)
    #     res1 = bpr_k(user_emb,inter_emb,neg_emb)
    #     res2 = kssm_p(user_emb[neg_idx], inter_emb, torch.Tensor(neg_idx).type(torch.long).view(-1,1), neg_emb, neg_sample_idx,1,False)
    #     # res2 = kssm(user_emb[neg_idx],inter_emb[neg_idx],neg_emb, neg_sample_idx, 1, normalized=False)
    #     if res1 != res2:
    #         count += (res1-res2).item()
    # print(count)


    # epoch = 2000
    # import time
    # start_time = time.time()
    # for i in range(epoch):
    #     kssm(a[idx],a[idx],a, neg_sample_idx, tau, normalized=True)
    # end_time = time.time()
    # print(end_time - start_time)
    # start_time = time.time()
    # for i in range(epoch):
    #     kssm2(a[idx],a[idx],a, neg_sample_idx, tau, normalized=True)
    # end_time = time.time()
    # print(end_time - start_time)

    # kssm2(a[idx],a[idx],a, neg_sample_idx, tau, normalized=True)