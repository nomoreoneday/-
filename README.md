# -文本抄袭自动检测分析

主要技术：Python / jieba / TF-IDF / MultinomialNB / KMeans / editdistance / TopN
项目简介：通过分析不同机构发布的文章，判断是否有文章抄袭的情况，并找到原文和抄袭的文章，以及具体相似的句子。可以应用于毕业论文查重，IP作品及文本抄袭检测
主要工作：对采集的文档进行数据清洗，采用TF-IDF提取文本特征，使用朴素贝叶斯分类器进行写作风格分类，并针对模仿自己写作风格的文章进行抄袭检测。先采用聚类算法对文档进行聚类降维，针对预测写作风格一致的作品，进行相似度检测及编辑距离检测

# -Thinking about Machine Learning

***Thinking1: 什么是监督学习，无监督学习，半监督学习？***

学习有没有监督区别在于数据有没有标签，有标签就是有监督学习，没有标签就是无监督学习，半监督学习位于二者之间，即数据一部分是有标签的，一部分是没有标签的。而没有标签的数据量往往大于有标签的数据量。具体来说：

* 监督学习：是一个机器学习中的方法，可以由训练资料中学到或建立一个模式（ learning model），并依此模式推测新的实例。
训练资料是由输入物件（通常是向量）和预期输出所组成。函数的输出可以是一个连续的值（称为回归分析），或是预测一个分类标签（称作分类）

* 无监督式学习(Unsupervised Learning )：是人工智能网络的一种算法(algorithm)，其目的是去对原始资料进行分类，以便了解资料内部结构。有别于监督式学习网络，无监督式学习网络在学习时并不知道其分类结果是否正确，亦即没有受到监督式增强(告诉它何种学习是正确的)。其特点是仅对此种网络提供输入范例，而它会自动从这些范例中找出其潜在类别规则。当学习完毕并经测试后，也可以将之应用到新的案例上。无监督学习里典型的例子就是聚类了。聚类的目的在于把相似的东西聚在一起，而我们并不关心这一类是什么。因此，一个聚类算法通常只需要知道如何计算相似度就可以开始工作了。

* 半监督学习(Semi-supervised learning)：其基本思想是利用数据分布上的模型假设, 建立学习器对未标签样本进行标签。
形式化描述为：给定一个来自某未知分布的样本集S=L∪U, 其中L 是已标签样本集L={(x1,y1),(x2,y2), … ,(x |L|,y|L|)}, U是一个未标签样本集U={x’1,x’2,…,x’|U|},希望得到函数f:X → Y可以准确地对样本x预测其标签y，这个函数可能是参数的，如最大似然法；可能是非参数的，如最邻近法、神经网络法、支持向量机法等；也可能是非数值的，如决策树分类。其中, x与x’  均为d 维向量, yi∈Y 为样本x i 的标签, |L| 和|U| 分别为L 和U 的大小, 即所包含的样本数。半监督学习就是在样本集S 上寻找最优的学习器。如何综合利用已标签样例和未标签样例,是半监督学习需要解决的问题。半监督学习问题从样本的角度而言是利用少量标注样本和大量未标注样本进行机器学习，从概率学习角度可理解为研究如何利用训练样本的输入边缘概率 P( x )和条件输出概率P ( y | x )的联系设计具有良好性能的分类器。这种联系的存在是建立在某些假设的基础上的，即聚类假设(cluster  assumption)和流形假设(maniford assumption)。

***Thinking2:K-means中的K值如何选取？***

1.手肘法

手肘法的核心指标是SSE(sum of the squared errors，误差平方和)


其中，Ci是第i个簇，p是Ci中的样本点，mi是Ci的质心（Ci中所有样本的均值），SSE是所有样本的聚类误差，代表了聚类效果的好坏。

手肘法的核心思想是：随着聚类数k的增大，样本划分会更加精细，每个簇的聚合程度会逐渐提高，那么误差平方和SSE自然会逐渐变小。并且，当k小于真实聚类数时，由于k的增大会大幅增加每个簇的聚合程度，故SSE的下降幅度会很大，而当k到达真实聚类数时，再增加k所得到的聚合程度回报会迅速变小，所以SSE的下降幅度会骤减，然后随着k值的继续增大而趋于平缓，也就是说SSE和k的关系图是一个手肘的形状，而这个肘部对应的k值就是数据的真实聚类数。当然，这也是该方法被称为手肘法的原因。





