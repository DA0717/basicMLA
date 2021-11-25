import numpy as np
from scipy.spatial import distance


def kNN(data, cate, udata, k):
    """
    kNN algorithm with
    :param data: known dataset(m, x). x--length of characters. m--#m samples.
    :param cate: category of samples in data. Shape: (m, 1)
    :param udata: cate-unknown data(n, x).
    :param k: k of "kNN".
    :return: ucate(n, 1), category of udata.
    """
    # step0: 数据扩充——需要先将 udata 转化为高维的数据(n, m, x)，将 data 转化为 (n, m, x）
    m = data.shape[0]
    n = udata.shape[0]
    ncate = np.expand_dims(cate, axis=0).repeat(n, axis=0)  # shape(n,m,1)
    # 删除最后一个维度
    ncate = np.squeeze(ncate, axis=2)
    expanddata = np.expand_dims(data, axis=0).repeat(n, axis=0)
    expandudata = np.expand_dims(udata, axis=1).repeat(m, axis=1)

    # step1: 计算所有距离，返回数据，dis (n, m)
    dis = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            # 在这儿修改距离计算方式，此处采用欧式距离
            dis[i][j] = distance.euclidean(expandudata[i][j], expanddata[i][j])

    # step2: 按照距离对相似的 data 对应的 cate 进行排序
    indices = np.argsort(dis, axis=1)
    first_indices = np.arange(n).reshape(n, -1)
    ocate = ncate[first_indices, indices] # 按照距离进行 cate 排序

    # step3: 获取 ocate 前 k 个 cate 并统计个数
    kcate = ocate[:, :k]
    ucate = np.zeros(n, dtype=int)
    for i in range(n):
        ucate[i] = np.argmax(np.bincount(kcate[i]))
    return ucate