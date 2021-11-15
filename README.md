# HWlayer_V2
HWlayer_V2
## 说明
* 这是HWnet的升级版本，网址：https://github.com/FFiot/HWnet。
![avatar](https://github.com/FFiot/HWnet/blob/master/HWnet_base/Image/gradient.gif?raw=true)
* 特征工程为前期工作，改变特征工程后进行尝试训练，会消耗大量的资源。使用拟合能力强的网络，期望能大幅度减少特征工程的工作。
* 网络调整后的重新训练，也会耗费大量的资源。
* 稀疏矩阵使网络训练速度加快，但是会引起过拟合，为了兼顾训练速度与泛化能力，设计了HWlayer。
## 概念
* 对于一个维度的样本集合 [x]，通过指定数量的分位数，获得 evaluate_list。
* 通过 softmax(abs(x - evaluate_list) * -1.0 * focus)，获得某一个值 x 在 evaluate_list上的分布。
  * focus 越大，对距离最近的 evaluate 集中度越高；越小，则会关注周边的 evaluate。
  * 解决了归一化问题，每一个数值转换为 一组 0 ~ 1.0 的概率值。
  * 相对于简单的嵌入式向量简单的 0 或者 1，这组概率值同样增加了维度，但是也兼顾了“线性”。
  * 基于分位数，使得网络参数使用率基本相同。
* 需要验证
  * 离群点的影响。
  * 使用测试集、测试集获得 evaluate 的性能差异。
## 升级
* 对于一个维度，数据的分布是不均匀的，如使用相同的 focus 值会导致网络异常。
  * 对每一个 evaluate 值，计算 focus，使得每一个 evaluate 计算获得的概率值大致相同。
* 无限叠加模型（有些失败，再议）。
  * 简单地增加模型规模，性能提升有限。
  * 性能较强，如能在模型中多次使用，理论上会多次增加模型性能。
* 将获得的概率值作为 “激活层” 使用。
  * x  -> hw_layer -> x1 
  * x1 -> Linear   -> x2
  * x3 = x2 * x 
  * 由于过于稀疏，导致模型很少的参数在工作，需要非常大的网络。
## 训练
* focus 设定为0.6时，较小的模型表现力较好。
* 较大的 batch_size，开始阶段，会使loss下降非常快，后期性能较差。
* 较大的 batch_size, 开始阶段，loss下降较慢，后期性能较好。
* 超级大的batch_size, 需要测试。
## 使用
* 对于已有模型
  * 增加在模型前端，使用线性网络与已有模型连接，固定已有模型参数，训练。
  * 增加在模型后端，代替最后的全连接成，固定已有模型参数，训练。
## 备注
* 感谢 Google、Pyorch、FastAI...
* 抱歉：学习AI的时间较少，投入时间较少，没有稳定的显卡，未能找到基于线性输入的数据集......
## 对比
*  使用HW_layer, History/F5-E128-F60_Linear-LSTM128x4-FC-SELU-FC
  * 参数：150K 
  * Private Score: 0.1972
  * Public Score  0.1918
* 不使用HW_layer, History/F25_Linear-LSTM128x4-FC-SELU-FC
  * 参数：150K 
  * Private Score: 0.2234
  * Public Score:  0.2189
* 增加很少的参数，并且做特征工程，获得了提升。