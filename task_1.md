"""
学习内容 1. 机器学习的一些概念 有监督、无监督、泛化能力、过拟合欠拟合(方差和偏差以及各自解决办法)、交叉验证 2. 线性回归的原理 3. 线性回归损失函数、代价函数、目标函数 4. 优化方法(梯度下降法、牛顿法、拟牛顿法等) 5、线性回归的评估指标 6、sklearn参数详解
"""
## 一 基本概念:

1. 有监督:Supervised Learning 通过标注数据进行学习的方法(回归和分类)
2. 无监督:Unsupervised Learning 用于学习的数据只有样本，没有标签，那么通过这种无标注数据进行学习的方法(聚类)
3. 泛化能力:Generalization Ability）机器学习算法对新鲜样本的适应能力
4. 过拟合:Overfitting 某模型训练集上指标很好，而在验证/测试集上指标偏低
5. 欠拟合:Underfitting 某模型在训练集上的性能不佳
6. 交叉验证:Cross-Validation 数据集分为训练集和测试集这一步骤。但是不同的是，我们现在只用一个数据作为测试集，其他的数据都作为训练集，并将此步骤重复N次 ![imag](https://pic4.zhimg.com/80/v2-fcb843dd06c15a515d03a543864bbb77_hd.png)

## 二 线性回归:
1. 原理: 利用线性函数对一个或多个自变量 （x 或 (x1,x2,...xk)）和因变量（y）之间的关系进行拟合的模型
        线性函数的定义是：一阶（或更低阶）多项式，或零多项式
2. 损失函数:Loss Function 是定义在单个样本上的，算的是一个样本的误差 
                   ![img](https://images0.cnblogs.com/blog/312753/201403/261738509057108.png)
3. 代价函数: Cost Function 是定义在整个训练集上的，是所有样本误差的平均，也就是损失函数的平均
3. 目标函数: Object Function（目标函数 ）定义为：Cost Function + 正则化项
4. 优化方法: 梯度下降法
            ![img](https://images2018.cnblogs.com/blog/697102/201803/697102-20180308101413881-932496454.png)
5. 评估指标: 均方误差(MSE)
            ![img](https://file.ai100.com.cn/files/sogou-articles/original/4c81c092-4a9d-4a0a-80ee-1963facd990e/640.png)

## 三 sklearn参数详解:
LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)

fit_intercept : boolean, optional, default True

    whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (e.g. data is expected to be already centered).
    
normalize : boolean, optional, default False

    This parameter is ignored when fit_intercept is set to False. If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm. If you wish to standardize, please use sklearn.preprocessing.StandardScaler before calling fit on an estimator with normalize=False.
    
copy_X : boolean, optional, default True

    If True, X will be copied; else, it may be overwritten.
    
n_jobs : int or None, optional (default=None)

    The number of jobs to use for the computation. This will only provide speedup for n_targets > 1 and sufficient large problems. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.



