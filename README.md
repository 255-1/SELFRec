<img src="https://i.ibb.co/54vTYzk/ssl-logo.png" alt="ssl-logo" border="0">

<p float="left"><img src="https://img.shields.io/badge/python-v3.7+-red"> <img src="https://img.shields.io/badge/pytorch-v1.7+-blue"> <img src="https://img.shields.io/badge/tensorflow-v1.14+-green">  <br>

**SELFRec** is a Python framework for self-supervised recommendation (SSR) which integrates commonly used datasets and metrics, and implements many state-of-the-art SSR models. SELFRec has a lightweight architecture and provides user-friendly interfaces. It can facilitate model implementation and evaluation.
<br>
**Founder and principal contributor**: [@Coder-Yu ](https://github.com/Coder-Yu) [@xiaxin1998](https://github.com/xiaxin1998) <br>
**Supported by**: [@AIhongzhi](https://github.com/AIhongzhi) (<a href="https://sites.google.com/view/hongzhi-yin/home">A/Prof. Hongzhi Yin</a>, UQ)

This repo is released with our [survey paper](https://arxiv.org/abs/2203.15876) on self-supervised learning for recommender systems. We organized a tutorial on self-supervised recommendation at WWW'22. Visit the [tutorial page](https://ssl-recsys.github.io/) for more information.
  
<h2>Architecture<h2>
<img src="https://raw.githubusercontent.com/Coder-Yu/SELFRec/main/selfrec.jpg" alt="ssl-logo" border="0">


<h2>Features</h2>
<ul>
<li><b>Fast execution</b>: SELFRec is developed with Python 3.7+, Tensorflow 1.14+ and Pytorch 1.7+. All models run on GPUs. Particularly, we optimize the time-consuming procedure of item ranking, drastically reducing the ranking time to seconds (less than 10 seconds for the scale of 10,000×50,000). </li>
<li><b>Easy configuration</b>: SELFRec provides a set of simple and high-level interfaces, by which new SSR models can be easily added in a plug-and-play fashion.</li>
<li><b>Highly Modularized</b>: SELFRec is divided into multiple discrete and independent modules/layers. This design decouples the model design from other procedures. For users of SELFRec, they just need to focus on the logic of their method, which streamlines the development.</li>
<li><b>SSR-Specific</b>:  SELFRec is designed for SSR. For the data augmentation and self-supervised tasks, it provides specific modules and interfaces for rapid development.</li>
</ul>

<h2>Requirements</h2>
	
```
numba==0.53.1
numpy==1.20.3
scipy==1.6.2
tensorflow==1.14.0
torch>=1.7.0
```

