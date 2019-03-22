## 论文解读与实现

记录一些论文阅读的心得并利用 python 实现论文中的算法。论文被放置在 [papers](./papers/README.md) 里面，而相应的代码放置在 [lab](./notebook/lab/README.md) 里面，解读则放在 [notebook](./notebook/README.md) 里面。 `notebook` 也可以看作是 `lab` 的 demo。

如果觉得有用，记得 star 支持！

## Graph-based Semi-Supervised Classification

直推式学习: transductive learning

参考: [Graph-based Semi-Supervised Classification](https://paperswithcode.com/search?q=Graph-based+Semi-Supervised+Classification) 与 [半监督学习](https://www.cnblogs.com/kamekin/p/9683162.html).

- LLGC: [Learning with Local and Global Consistency](notebook/lgc.md)[^1]
- Mean Teacher: [Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results](notebook/mean_teacher.md)[^2]
- GRFH: [Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions](notebook/GRFH.md)[^3]
- RMGT: [Robust Multi-Class Transductive Learning with Graphs](notebook/RMGT.md)[^4]
- GCN: [Semi-Supervised Classification With Graph Convolutional Networks](notebook/GCN.md)[^5]

[^1]: Zhou D, Bousquet O, Lal T N, et al. Learning with Local and Global Consistency[C]. neural information processing systems, 2003: 321-328.
[^2]: Tarvainen A, Valpola H. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results[J]. neural information processing systems, 2017: 1195-1204.
[^3]: Zhu X, Ghahramani Z, Lafferty J D, et al. Semi-supervised learning using Gaussian fields and harmonic functions[C]. international conference on machine learning, 2003: 912-919.
[^4]: Liu W, Chang S. Robust multi-class transductive learning with graphs[C]. computer vision and pattern recognition, 2009: 381-388.
[^5]: Kipf T N, Welling M. Semi-Supervised Classification with Graph Convolutional Networks[J]. international conference on learning representations, 2017.

## K 近邻 + CNN

- Condensed NN 综述: [Survey of Nearest Neighbor Condensing Techniques](notebook/fcn_rule.md)[^6]

[^6]: Amal M, Ahmed B. Survey of Nearest Neighbor Condensing Techniques[J]. International Journal of Advanced Computer Science and Applications, 2011, 2(11).