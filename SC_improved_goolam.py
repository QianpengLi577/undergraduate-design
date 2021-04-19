from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.cluster import KMeans
import math as m
import warnings
warnings.filterwarnings("ignore")
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn import metrics

#SC part
def get_dis_matrix(data,sigma):
    nPoint = data.shape[0]
    dis_matrix = np.zeros((nPoint, nPoint))
    for i in range(nPoint):
        for j in range(i + 1, nPoint):
            dis_matrix[i][j] = dis_matrix[j][i] = m.sqrt(np.power(data[i,:] - data[j,:], 2).sum())
    # dis_matrix=np.exp(dis_matrix/2.0/sigma/sigma*(-1.0))
    return dis_matrix

def getW(data, k1, k2, sigma):
    dis_matrix = get_dis_matrix(data,sigma)
    W1 = np.zeros((data.shape[0], data.shape[0]))
    W2 = np.zeros((data.shape[0], data.shape[0]))
    for idx, each in enumerate(dis_matrix):
        index_array = np.argsort(each)
        W1[idx][index_array[1:k1 + 1]] = 1
        W2[idx][index_array[1:k2 + 1]] = 1
    tmp_W1 = np.transpose(W1)
    tmp_W2 = np.transpose(W2)
    W1 = (tmp_W1 + W1) / 2
    W2 = (tmp_W2 + W2) / 2
    return W1, W2, dis_matrix

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

def dijkstra(graph,src):
    length = len(graph)
    type_ = type(graph)
    if type_ == list:
        nodes = [i for i in range(length)]
    elif type_ == dict:
        nodes = list(graph)

    visited = [src]
    path = {src:{src:[]}}
    nodes.remove(src)
    distance_graph = {src:0}
    pre = next = src

    while nodes:
        distance = float('inf')
        for v in visited:
             for d in nodes:
                new_dist = graph[src][v] + graph[v][d]
                if new_dist <= distance:
                    distance = new_dist
                    next = d
                    pre = v
                    graph[src][d] = new_dist


        path[src][next] = [i for i in path[src][pre]]
        path[src][next].append(next)

        distance_graph[next] = distance

        visited.append(next)
        nodes.remove(next)

    return distance_graph, path

def findindex(distance,index):
    index1=[]
    for i in range(len(index)):
        index1.append(np.argwhere(distance[0,:]==index[i])[0][0])
    return np.array(index1)

def count_percent(x1,x2):
    row,col=x1.shape
    percent = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            if x1[i][j]!=0:
                percent[i][j]=x1[i][j]/x2[i][j]
    return percent

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

import scipy.io as sio
data_all = sio.loadmat('Goolamexpression.mat')
a=data_all["Goolamexpression"]
dataname = r'data_improved/goolamimproved.csv'
with open(dataname,encoding = 'utf-8') as f:
    x1 = np.loadtxt(f,delimiter = ",")
y=a[:,a.shape[1]-1]
y=y.astype(np.int)
n=y.max()-y.min()+1


#feature selector

#cluster using sc
KNN_for_distance = 50#6
KNN_for_neighbord = 43#12
W1, W2, EUdistance= getW(x1, KNN_for_distance, KNN_for_neighbord, 0.7)
W1[W1 > 0] = 1
W2[W2 > 0] = 1
print(W1.sum(),',',W2.sum())
D1=W1*EUdistance#for Dijkstra
D2=W2*EUdistance#for percent
D3=W2*EUdistance#for percent

inf = 100000
D1[D1 == 0] = 100000
length=D1.shape[0]
D1=D1*(np.ones((D1.shape[0],D1.shape[0]))-np.identity(D1.shape[0]))
D1=D1.tolist()
graph_list = D1
# for j in range(length):
#     distance, path = dijkstra(graph_list, j)
#     # print (distance, '\n', path)
#     c1 = []
#     for k in distance.items():
#         c1.append(k)
#     c1 = np.transpose(np.array(c1))
#     c1=c1[:,0:len(np.argwhere(c1[1,:]<100000))]  #used for Dijkstra distance change   shape(2*M) index0 means target  index1 means distance
#
#     index1=np.argwhere(W2[j,:]==1)
#     index2=findindex(c1,index1)
#     for i in range(len(index1)):
#         D2[j,index1[i]]=c1[1,index2[i]]
# print('done')

aaa=np.amax(D3, axis=1)
aaa=np.log10(aaa+1)
aaa=1/aaa
# aaa=(aaa-aaa.min())/(aaa.max()-aaa.min())
aaa= np.concatenate([aaa] * length, axis=0)
aaa=aaa.reshape(length,length)
aaa=np.transpose(aaa)
# aaa=np.log(aaa+1)
# aaa=(aaa-aaa.min())/(aaa.max()-aaa.min())

p=count_percent(D3,D2)
p=p*aaa
D = getD(p)
L = getL(D, p)
eigvec = getEigen(L,n)
eigvec=np.real(eigvec)
clf = KMeans(n_clusters=n)
s = clf.fit(eigvec)
C = s.labels_
print('processed data using sc ARI:', metrics.adjusted_rand_score(y, C))
print('NMI:', normalized_mutual_info_score(y, C))
print('ACC:',acc(y, C))


from sklearn.cluster import SpectralClustering
sc1 = SpectralClustering(n_clusters=n,affinity='nearest_neighbors')
print('SC KNN ARI:', metrics.adjusted_rand_score(y, sc1.fit_predict(x1)))
c='ARI:'+ str(metrics.adjusted_rand_score(y,C))+'\n'+'NMI:'+ str(normalized_mutual_info_score(y, C))+'\n'
c=c+'ACC:'+str(acc(y, C))+'\n'+'SKARI'+str(metrics.adjusted_rand_score(y, sc1.fit_predict(x1)))
fh = open('performancegoolamimproved.txt', 'w', encoding='utf-8')
fh.write(c)
fh.close()