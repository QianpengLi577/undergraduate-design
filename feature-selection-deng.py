from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.cluster import KMeans
import math as m
import warnings
warnings.filterwarnings("ignore")
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score

#ACC function
def acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
        ind = linear_sum_assignment(w.max() - w)
        ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

#SC part
def get_dis_matrix(data,sigma):
    nPoint = data.shape[0]
    dis_matrix = np.zeros((nPoint, nPoint))
    for i in range(nPoint):
        for j in range(i + 1, nPoint):
            dis_matrix[i][j] = dis_matrix[j][i] = m.sqrt(np.power(data[i,:] - data[j,:], 2).sum())
    # dis_matrix=np.exp(dis_matrix/2.0/sigma/sigma*(-1.0))
    return dis_matrix

def getW(data, k,sigma):
    dis_matrix = get_dis_matrix(data,sigma)
    W = np.zeros((data.shape[0], data.shape[0]))
    for idx, each in enumerate(dis_matrix):
        index_array = np.argsort(each)
        W[idx][index_array[1:k+1]] = 1
    tmp_W = np.transpose(W)
    W = (tmp_W+W)/2
    return W

def getD(W):
    D = np.diag(sum(W))
    return D

def getL(D, W):
    return D - W

def getEigen(L,cluster_num):
    eigval, eigvec = np.linalg.eig(L)
    ix = np.argsort(eigval)[0:cluster_num]
    return eigvec[:, ix]
#

dataname = r'data/deng.csv'
with open(dataname,encoding = 'utf-8') as f:
    x = np.loadtxt(f,delimiter = ",")
labelname = r'label//deng.csv'
with open(labelname,encoding = 'utf-8') as f:
    y = np.loadtxt(f,delimiter = ",")
y=y.astype(np.int)
n=y.max()-y.min()+1
x=np.log(x+1)

clf = KMeans(n_clusters=n)
clustering = clf.fit(x)
yp = clustering.predict(x)
from sklearn import metrics
print('before process ARI:', metrics.adjusted_rand_score(y, yp))
# x=x.tolist()
# y=y.tolist()
selector=SelectFromModel(LogisticRegression()).fit(x, y)
# print("estimator的模型参数",selector.estimator_.coef_)

# 根据estimator中特征重要性均值获得阈值
# print("用于特征选择的阈值；",selector.threshold_)

# 哪些特征入选最后特征，true表示入选
# print("特征是否保留",selector.get_support())
# 获得最后结果
# print("特征提取结果",selector.transform(x))
x1=selector.transform(x)

# np.savetxt('dengimproved.csv',x1,fmt='%f',delimiter=',')
clustering = clf.fit(x1)
yp = clustering.predict(x1)
from sklearn import metrics
print('after process ARI:', metrics.adjusted_rand_score(y, yp))

#cluster using sc
KNN_k = 8#5
W = getW(x1, KNN_k,0.7)
D = getD(W)
L = getL(D, W)
eigvec = getEigen(L,n)
eigvec=np.real(eigvec)
clf = KMeans(n_clusters=n)
s = clf.fit(eigvec)
C = s.labels_
print('processed data using sc ARI:', metrics.adjusted_rand_score(y, C))
print('NMI:', normalized_mutual_info_score(y, C))
print('ACC:',acc(y, C))
c='ARI:'+ str(metrics.adjusted_rand_score(y,C))+'\n'+'NMI:'+ str(normalized_mutual_info_score(y, C))+'\n'
c=c+'ACC:'+str(acc(y, C))
fh = open('performancedeng.txt', 'w', encoding='utf-8')
fh.write(c)
fh.close()