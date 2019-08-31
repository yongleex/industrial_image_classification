# 1.比赛基本信息
[AOI 瑕疵分類官方网址](https://aidea-web.tw/topic/a49e3f76-69c9-4a4a-bcfc-c882840b3f27)

简言之,“数百张图像/类别”  图像数据的6个类别的分类任务。
（0 表示 normal，1 表示 void，2 表示 horizontal defect，3 表示 vertical defect，4 表示 edge defect，5 表示 particle）

# 2. 数据分析

| 类别 |训练集 |测试集 |
| :------: | :------: | :------: |
| 0 |674 | ~2552 |
| 1 | 492 | ~2049 |
| 2 | 100 | ~592 |
| 3 | 378 | ~1530 |
| 4 |240 | ~914 |
| 5 | 644 | ~2505 |
| 合计| 2528 | 10142 |
- 数据量少。1.需要数据增强；2.半监督的方式利用测试集的数据.
- 样本分布不均衡。1.采用复制“重复使用”的方式，使得数据“看起来”均衡。
- 测试集的分布是根据我个人的结果统计分析出的。
# 3. 代码说明
- preprocessing，数据预处理；
- train_nets,训练网络；
- predict，用网络进行预测；
- result_analysis.ipynb，预测结果的分析，展示，合成最后提交的数据。大致参考就可以。

# 4.结果排名
暂时还没训练完成，等结果提交后，再列举出来.准确率应该大于98%.

| 方案|准确率|排名|
|:-----:|:-------:|:-------:|
|最初五个网络预测|0.9829839|~56/147|
|新的方案|0.9807644|61/147|

为何二次训练会导致准确率下降呢？可能是第一次的错误结果在第二次得到了放大。


# 5.联系我
如有任何问题和疑问，邮件联系 [yongli.cv@gmail.com](yongli.cv@gmail.com)


