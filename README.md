# 一层网络拟合任意连续函数(HWlayer)
# One-layer network fitting any continuous function (HWlayer)
## 说明
## Instruction
* 这是HWnet(https://github.com/FFiot/HWnet)的升级版本。
* This is an upgraded version of HWnet (https://github.com/FFiot/HWnet).
![avatar](https://github.com/FFiot/HWnet/blob/master/HWnet_base/Image/gradient.gif?raw=true)
* 稀疏矩阵使网络训练速度加快，但是会引起过拟合，为了兼顾训练速度与泛化能力，设计了HWlayer。
* The sparse matrix makes the network training faster, but it will cause overfitting. In order to take into account the training speed and generalization ability, HWlayer is designed.
* 网络调整后的重新训练，也会耗费大量的资源。
* Retraining after network adjustment will also consume a lot of resources.
## 方法
## Method
* 对于一个维度的样本集合 [x]，通过指定数量的分位数，获得 evaluate_list。
* For a sample set [x] of one dimension, get evaluate_list with the specified number of quantiles.
* 通过 softmax(abs(x - evaluate_list) * -1.0 * focus)，获得某一个值 x 在 evaluate_list上的分布。
* Obtain the distribution of a value x on evaluate_list by softmax(abs(x - evaluate_list) * -1.0 * focus).
  * focus 越大，对距离最近的 evaluate 集中度越高；越小，则会关注周边的 evaluate。
  * The larger the focus, the higher the concentration of the nearest evaluate; the smaller the focus, the surrounding evaluate.
  * 解决了归一化问题，每一个数值转换为 一组 0 ~ 1.0 的概率值。
  * Solved the normalization problem, where each value is converted into a set of probability values ​​from 0 to 1.0.
  * 相对于简单的嵌入式向量简单的 0 或者 1，这组概率值同样增加了维度，但是也兼顾了“线性”。
  * Compared to the simple 0 or 1 of the simple embedded vector, this set of probability values ​​also increases the dimension, but also takes into account the "linearity".
  * 基于分位数，使得网络参数使用率基本相同。
  * Based on quantiles, the network parameter usage rates are basically the same.
## 升级
## Upgrade
* 对于一个维度，数据的分布是不均匀的，如使用相同的 focus 值会导致网络异常。
* For a dimension, the distribution of data is not uniform, such as using the same focus value will cause network exceptions.
* 对每一个 evaluate 值，计算 focus，使得每一个 evaluate 计算获得的概率值大致相同。
* For each evaluate value, calculate focus, so that the probability value obtained by each evaluate calculation is roughly the same. 
* 累计各值域损失: 大于预期的值域进行分裂，小于预期的值域与周围值域合并。
* By accumulating the loss of each value field: larger than expected will be split, less than expected will be merged.