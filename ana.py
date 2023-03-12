import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from random import choices
#amazon
max_item = 98572
max_user = 138333
sample_num = 1527913
f = open("dataset/amazon-kindle/train.txt")
#yelp2018
max_item = 38048
max_user = 31668
sample_num = 1237259
f = open("dataset/yelp2018/train.txt")
#iFashion
max_item = 81614
max_user = 300000
sample_num = 1255447
f = open("dataset/iFashion/train.txt")
#douban
max_item = 22222
max_user = 12638
sample_num = 478730
f = open("dataset/douban-book/train.txt")
data = []
item_freq = dict()
item_list = []
for line in f:
    user = line.split(" ")[0]
    item = line.split(" ")[1]
    if item not in item_freq:
        item_freq[item] = 1
    else:
        item_freq[item] += 1
    item_list.append(item)
    data.append((user, item))


minibatch = choices(data, k=sample_num)
minibatch_freq = dict()
for neg_pair in minibatch:
    if neg_pair[1] not in minibatch_freq:
        minibatch_freq[neg_pair[1]] = 1
    else:
        minibatch_freq[neg_pair[1]] += 1

corpus = choices(list(range(max_item)), k=sample_num)
corpus_freq = dict()
for neg_item in corpus:
    if neg_item not in corpus_freq:
        corpus_freq[neg_item] = 1
    else:
        corpus_freq[neg_item] += 1

y1 = []
for k, v in minibatch_freq.items():
    y1.append(v)

y2 = []
for k, v in corpus_freq.items():
    y2.append(v)

y3 = []
for e in Counter(item_list).most_common():
    y3.append(e[1])

x1 = list(range(len(y1)))
x2 = list(range(len(y2)))

# assert len(x1) == len(x2)
plt.scatter(x1,y1,s=1, label='item frequencies sampled by minibatch')
plt.scatter(x2,y2,s=1, label='item frequencies sampled by corpus')

x3 = list(range(len(y3)))
plt.plot(x3,y3, linewidth=3, label='item frequencies',color='red')
plt.xlabel('item id')
plt.ylabel('item frequency')
plt.legend()
plt.grid()
plt.savefig("./1.eps",format='eps',dpi=1000)
plt.savefig("./1.jpg")
