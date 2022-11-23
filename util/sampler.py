from random import shuffle,randint,choice, sample
import torch
import numpy as np

def next_batch_pairwise(data,batch_size,n_negs=1):
    training_data = data.training_data
    shuffle(training_data)
    batch_id = 0
    data_size = len(training_data)
    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            for m in range(n_negs):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(data.item[neg_item])
        yield u_idx, i_idx, j_idx

def next_batch_pairwise2(data,batch_size,n_negs=1):
    training_data = data.training_data
    shuffle(training_data)
    batch_id = 0
    data_size = len(training_data)
    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, j_idx, k_idx = [], [], [], []
        item_list = list(data.item.keys())
        user_list = list(data.user.keys())
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            for m in range(n_negs):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(data.item[neg_item])
            for m in range(n_negs):
                neg_user = choice(user_list)
                while neg_user in data.training_set_i[items[i]]:
                    neg_user = choice(user_list)
                k_idx.append(data.user[neg_user])
        yield u_idx, i_idx, j_idx, k_idx


def negative_sample(data, user, n_negs, sample_strategy='uniform'):
    res = []
    choosed_items = list(data.training_set_u[user].keys())
    if sample_strategy == 'uniform':
        item_list = list(data.item.keys())
        neg_items = sample(item_list, n_negs) #unique
        for i in neg_items:
            if i not in choosed_items:
                res.append(data.item[i])
        while(len(res) != n_negs):
            neg_item = choice(item_list)
            while (neg_item in choosed_items) or (data.item[neg_item] in res):
                neg_item = choice(item_list)
            res.append(data.item[neg_item])
        return res
    if sample_strategy == 'popularity':
        all_item = list(np.array(data.training_data)[:,1])
        random_index_list = np.random.randint(0, len(all_item), n_negs).tolist()
        for idx in random_index_list:
            if all_item[idx] not in choosed_items and all_item[idx] not in res:
                res.append(data.item[all_item[idx]])
        while(len(res)!=n_negs):
            random_index = np.random.randint(0, len(all_item), 1)[0]
            neg_item = all_item[random_index]
            while (neg_item in choosed_items) or (data.item[neg_item] in res):
                random_index = np.random.randint(0, len(all_item), 1)[0]
                neg_item = all_item[random_index]
            res.append(data.item[neg_item])
        return res

def next_batch_pairwise_in_batch(data, batch_size):
    training_data = data.training_data
    shuffle(training_data)
    batch_id = 0
    data_size = len(training_data)
    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, j_idx = [], [], []
        # item_list = list(data.item.keys())
        #inbatch_neg_sample
        min_len = batch_size
        tmp_j_list = []
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            neg_items = list(set(items)-set(data.training_set_u[user].keys()))
            min_len = min(min_len, len(neg_items))
            tmp_j_list.append(neg_items)
        for neg_items in tmp_j_list:
            j_idx += [data.item[neg] for neg in sample(neg_items, min_len)]
            # neg_item = choice(item_list)
            # while neg_item in data.training_set_u[user]:
            #     neg_item = choice(item_list)
            # j_idx.append(data.item[neg_item])

        yield u_idx, i_idx, j_idx, min_len

def next_batch_pairwise_k(data,batch_size,n_negs=512):
    training_data = data.training_data
    shuffle(training_data)
    batch_id = 0
    data_size = len(training_data)
    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, j_idx = [], [], []

        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            res = negative_sample(data, user, n_negs, sample_strategy='uniform')
            j_idx += res
        yield u_idx, i_idx, j_idx


def next_batch_pairwise_cluster(data,batch_size, item_2cluster, n_negs=1):
    training_data = data.training_data
    shuffle(training_data)
    batch_id = 0
    data_size = len(training_data)
    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            neg_cluster = item_2cluster[data.item[items[i]]]
            neg_idx = torch.where(item_2cluster==neg_cluster)
            while True:
                neg_item = choice(neg_idx[0])
                if neg_item in data.training_set_u[user]:
                    continue
                j_idx.append(data.item[str(neg_item.item())])
                break
        yield u_idx, i_idx, j_idx

def sample_cl_negtive_idx(idx, sample_num):
    if sample_num > len(idx):
        sample_num = len(idx)
    u_idx = torch.Tensor(idx).type(torch.long)
    neg_sample_idx = (u_idx[1:])[:sample_num].unsqueeze(dim=0)
    for i in range(1, len(u_idx)):
        neg_idx = u_idx[:i]
        neg_idx = torch.cat((neg_idx, u_idx[(i+1):]))[:sample_num].unsqueeze(dim=0)
        neg_sample_idx = torch.cat((neg_sample_idx, neg_idx), dim=0)
    return u_idx.view(-1,1), neg_sample_idx

def sample_cl_negtive_idx_with_rec_neg(idx, neg_idx, sample_num):
    u_idx = torch.Tensor(idx).type(torch.long)
    neg_idx = torch.Tensor(neg_idx).type(torch.long).view(u_idx.shape[0], -1)
    if sample_num > neg_idx.shape[1]:
        sample_num = neg_idx.shape[1]
    neg_sample_idx = neg_idx[:,:sample_num]
    return u_idx.view(-1,1), neg_sample_idx

if __name__=="__main__":
    idx = [21,65,45]
    neg = [12,12,12,23,23,23,78,89,1]
    print(sample_cl_negtive_idx_with_rec_neg(idx, neg, 2)[0])
    print(sample_cl_negtive_idx_with_rec_neg(idx, neg, 2)[1])

def next_batch_pairwise_cluster_middle(data,batch_size, item_2cluster, item_centroids, n_negs=1):
    training_data = data.training_data
    shuffle(training_data)
    batch_id = 0
    data_size = len(training_data)
    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            item2cluster = item_2cluster[data.item[items[i]]]
            item_centroid = item_centroids[item2cluster]
            similarity = torch.sparse.mm(item_centroid.unsqueeze(0), item_centroids.transpose(1,0))
            _, top_k_idx = torch.sort(similarity, descending=True)
            idx = len(top_k_idx)//2
            while True:
                neg_cluster = top_k_idx[0][idx]
                if neg_cluster == item2cluster:
                    idx += 1
                    continue
                neg_idx = torch.where(item_2cluster==neg_cluster)
                neg_item = choice(neg_idx[0])
                if neg_item in data.training_set_u[user]:
                    idx += 1
                    continue
                j_idx.append(data.item[str(neg_item.item())])
                break
        yield u_idx, i_idx, j_idx

def next_batch_pairwise_cluster_argmin(data,batch_size, item_2cluster, item_centroids, n_negs=1):
    training_data = data.training_data
    shuffle(training_data)
    batch_id = 0
    data_size = len(training_data)
    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            item2cluster = item_2cluster[data.item[items[i]]]
            item_centroid = item_centroids[item2cluster]
            similarity = torch.sparse.mm(item_centroid.unsqueeze(0), item_centroids.transpose(1,0))
            idx = torch.argmin(similarity)
            while True:
                neg_cluster = idx
                if neg_cluster == item2cluster:
                    idx += 1
                    continue
                neg_idx = torch.where(item_2cluster==neg_cluster)
                neg_item = choice(neg_idx[0])
                if neg_item in data.training_set_u[user]:
                    idx += 1
                    continue
                j_idx.append(data.item[str(neg_item.item())])
                break
        yield u_idx, i_idx, j_idx

def next_batch_pointwise(data,batch_size):
    training_data = data.training_data
    data_size = len(training_data)
    batch_id = 0
    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, y = [], [], []
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            y.append(1)
            for instance in range(4):
                item_j = randint(0, data.item_num - 1)
                while data.id2item[item_j] in data.training_set_u[user]:
                    item_j = randint(0, data.item_num - 1)
                u_idx.append(data.user[user])
                i_idx.append(item_j)
                y.append(0)
        yield u_idx, i_idx, y