import numpy as np
from sklearn.neighbors import NearestNeighbors


def SMOTE(X, y, N, k=3):

    # 分离多数类和少数类样本
    X_majority = X[y == 0]
    X_minority = X[y == 1]

    # 计算每个少数类样本需要生成的合成样本数量
    N_per_sample = N // len(X_minority)

    # 如果k大于少数样本数量，则将其减少到可能的最大值
    k = min(k, len(X_minority) - 1)

    # 初始化列表以存储合成样本和相应的标签
    synthetic_samples = []
    synthetic_labels = []

    # 在少数类样本上拟合k近邻
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X_minority)

    for minority_sample in X_minority:
        # 查找当前少数类样本的k个最近邻居
        _, indices = knn.kneighbors(minority_sample.reshape(1, -1), n_neighbors=k)

        # 随机选择k个邻居并创建合成样本
        for _ in range(N_per_sample):
            neighbor_index = np.random.choice(indices[0])
            neighbor = X_minority[neighbor_index]

            # 计算当前少数类样本和邻居之间的差异
            difference = neighbor - minority_sample

            # 生成一个0到1之间的随机数
            alpha = np.random.random()

            # 创建一个合成样本作为少数类样本和邻居的线性组合
            synthetic_sample = minority_sample + alpha * difference

            # 将合成样本及其标签追加到列表中
            synthetic_samples.append(synthetic_sample)
            synthetic_labels.append(1)

    # 将列表转换为numpy数组
    X_synthetic = np.array(synthetic_samples)
    y_synthetic = np.array(synthetic_labels)

    # 将原始多数类样本与合成样本合并
    X_balanced = np.concatenate((X_majority, X_synthetic), axis=0)
    y_balanced = np.concatenate((np.zeros(len(X_majority)), y_synthetic), axis=0)

    return X_balanced, y_balanced


def datapro(x, y):
    xx = []
    for i in x:
        xx.append(i.flatten().tolist())
    xx = np.array(xx)
    yy = y.reshape(1, -1)[0]
    X_balanced, y_balanced = SMOTE(xx, yy, N=1818)
    X_train = np.concatenate((X_balanced, xx[:1818]), axis=0)
    y_train = np.concatenate((y_balanced, yy[:1818]), axis=0)
    XX = []
    for i in X_train:
        XX.append(i.reshape(49, 96))
    XX = np.array(XX)
    y_train = np.array([int(i) for i in y_train])
    y_train = y_train.reshape(8205, 1)
    return XX, np.array(y_train)
