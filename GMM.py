from __future__ import division, print_function, unicode_literals 
import numpy as np
import scipy.cluster.vq as vq
from utils import *
import pickle as pkl
import os
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture as BGM
import time
from memory_profiler import profile

class GMM(object):
    def __init__(self, dim = 0, K = None, filename = None, data = None, init_method="random", kmeans_iter = 10, iters = 10, run_method="loaded"):
        # fix sigma as a PSD matrix, to reduce memory needed
        # self.sigma = np.random.rand(dim, dim)
        # self.sigma = self.sigma.dot(self.sigma.T)
        self.data = data

        if filename: 
            self.load_model(filename)

        elif not data is None: # have data
            self.K = K
            self.dim = dim
            self.clusters = []
            N = len(data)

            if init_method == "kmeans":
                print("Initilizing with kmeans ...")
                # run kmeans
                (centroids, labels) = vq.kmeans2(data, K, minit="points", iter=kmeans_iter)
                plot_data(self.data, labels)
                # assign data for each clusterd
                clusters = [[] for i in range(K)]
                for i in range(len(labels)):
                    clusters[labels[i]].append(i)
                # estimate distribution parameters for each cluster
                for cluster in clusters:
                    self.clusters.append(NormalDistribution(dim = dim, data = data[cluster, :]))
                # calculate priors for model
                self.pi = [len(c)/N for c in clusters]

            elif init_method == "uniform":
                np.random.shuffle(data)
                batch = int(N/K)
                for i in range(K):
                    self.clusters.append(NormalDistribution(dim = dim, data = data[(i * batch):((i+1) * batch)]))

                self.pi = np.ones(K, dtype="double")/K
                print("Init uniform: ")
                for cluster in self.clusters:
                    print(cluster)
                # print("Priors: ", self.pi)
        else: # no data
            print('Initializing with no data...')
            self.K = K
            self.dim = dim
            self.clusters = [NormalDistribution(dim) for i in range(K)]
            self.pi = np.ones(K, dtype="double")/K
        
        if run_method=="loaded":
            self.load_model(model)
        if run_method=="EM":
            self.EM_alg(self.data, iters = iters)

    def load_model(self, filename):
        savedir = '/media/hieu/DATA/ShareOS/PythonVirtualEnv/WorkSpace/test/topic model/GMM/generated_files'        
        with open(os.path.join(savedir, filename), 'r+b') as f:
            self.dim = pkl.load(f)
            self.pi = pkl.load(f)
            self.clusters = pkl.load(f)
        
        self.K = len(self.pi)
        pass
    
    def save_model(self, filename):
        savedir = '/media/hieu/DATA/ShareOS/PythonVirtualEnv/WorkSpace/test/topic model/GMM/generated_files'
        with open(os.path.join(savedir, filename), 'w+b') as f:
            pkl.dump(self.dim, f)
            pkl.dump(self.pi, f)
            pkl.dump(self.clusters, f)
        
    def EM_alg(self, data, iters = 50):
        data = np.array(data)
        K = self.K
        N = len(data)
        d = self.dim

        for i in range(iters):
            print("EM ", i)
            # E-step
            T = np.zeros((N, K), dtype="float") # T_nk = P(z_n = k | x_n, Theta)

            for n in range(N):
                for k in range(K):
                    density = self.clusters[k].pdf(data[n, :])
                    T[n, k] = self.pi[k] * density
            
            T = T / np.sum(T, axis= 1).reshape(N, 1)
            # M-step

            M = np.sum(T, axis = 0)

            for k in range(K):
                # update priors
                self.pi[k] = M[k] / N

                # update mu_k
                mu_k = np.dot(T.T[k, :], data) / M[k] # (1, N) * (N, d) = (1, d)

                # update sigma_k
                Xbar_k = data - mu_k # (N,d)

                sigma_k = (T.T[k, :]*Xbar_k.T).dot(Xbar_k) # (1,N)*(d,N) = (d,N) . (N,d) = (d,d)
                # sigma_k = np.zeros((d,d))
                # for n in range(N):
                #     vx_ = (data[n] - mu_k).reshape(d, 1)
                #     vx = vx_.dot(vx_.T) # (1,d).T * (1,d) = (d,d)
                #     sigma_k += T[n, k] * vx
                sigma_k = sigma_k/M[k]
                # print("sigma_k:", sigma_k)
                self.clusters[k].update(mu_k, sigma_k)
            
            if (i + 1) % 10 == 0 and i > 0:
                self.save_model('gmm_model_' + str(i + 1) + '.dat')
        
        self.save_model('gmm_model_final.dat')
    
    def __str__(self):
        return "means:\n {}\ncovariance:\n {}".format([c.mu for c in self.clusters], [c.sig for c in self.clusters])

def demo_data():
    np.random.seed(11)
    means = [[2, 2], [8, 3], [3, 6]]
    cov = [[1, 0], [0, 1]]
    N = 500
    X0 = np.random.multivariate_normal(means[0], cov, N)
    X1 = np.random.multivariate_normal(means[1], cov, N)
    X2 = np.random.multivariate_normal(means[2], cov, N)
    X = np.concatenate((X0, X1, X2), axis = 0)
    original_label = np.asarray([0]*N + [1]*N + [2]*N).T
    return X, original_label

def plot_data(X, label):
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]

    plt.figure()

    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8)

    plt.axis('equal')

def plot_sample(X, label, clusters):

    plt.figure()

    for x in X:
        l = np.argmax([cluster.pdf(x) for cluster in clusters])
        if l == 0: plt.plot(x[0], x[1], 'b^', markersize=4, alpha = .8)
        elif l == 1: plt.plot(x[0], x[1], 'go', markersize = 4, alpha = .8)
        else: plt.plot(x[0], x[1], 'rs', markersize = 4, alpha = .8)

    plt.axis('equal')

def plot_contour(X, label, clusters):
    indices = [np.random.randint(X.shape[0]) for _ in range(100)]
    x = X[indices, 0]
    y = X[indices, 1]
    xv, yv = np.meshgrid(x, y)
    plt.figure()
    z = np.zeros((xv.shape[0], xv.shape[1]))
    for cluster in clusters:
        for i in range(xv.shape[0]):
            for j in range(xv.shape[1]):
                z[i, j] = cluster.pdf(np.array([xv[i, j], yv[i, j]]))

        maxval = np.max(np.max(z, axis=0))
        cs = plt.contour(xv, yv, z, levels=np.arange(0, maxval, maxval/10))
        plt.clabel(cs, inline=1, fontsize=4)

def test_clustering(X, origin_labels, clusters):
    labels = []
    for x in X:
        l = np.argmax([cluster.pdf(x) for cluster in clusters])
        labels.append(l)
    print(origin_labels)
    print(labels)
    
def purity(confusion_table, N):
    """
    Parameters
    ----------
        `confusion_table:list` - each element is a dict has form: `dict[label] = number of points`
        `N:integer` - number of points used to train model
    
    Returns
    -------
        `purity = 1/N * sum(max(confusion_table[k].values()))`
            To compute purity, each cluster is assigned to the class which 
            is most frequent in the cluster, and then the accuracy of this 
            assignment is measured by counting the number of correctly assigned 
            documents and dividing by $N$
    """
    majority_labels = []
    for cluster in range(len(confusion_table)):
        majority_labels.append(max(confusion_table[cluster].values()))
        
    return sum(majority_labels)/N

def NMI(confusion_table, labels, N):
    """
    Normalized Mutual Information

    Parameters
    ----------
        `confusion_table:list` - each element is a dict has form: `dict[label] = number of points`
        `labels:dict` - `labels[label] = number of data points have this label`
        `N:integer` - number of points used to train model
    
    Returns
    -------
        `NMI(L, C) = I(L, C) / ((H(L) + H(C)) / 2)`
        `I(L, C)` is mutual information
            `I(L, C) = SUM_k(SUM_m( p(L_m_C_k) * log(p(L_m_C_k) / (p(L_m) * p(C_k)) )))`
        `H` is entropy of Labels set or Cluster set
            `H(C) = -SUM_k( p(C_k) * log(p(C_k)) )`
            
        `p(x)` is probability of x according to the whole data points set (N points).
    """
    I = 0
    H_C = 0
    H_L = 0
    for cluster in range(len(confusion_table)):
        p_C_k = sum(confusion_table[cluster].values()) / N
        H_C -= p_C_k * np.log(p_C_k)
        for label in labels.keys():
            if confusion_table[cluster].has_key(label):
                p_L_m_C_k = confusion_table[cluster][label] / N
                p_L_m = labels[label] / N
                I +=  p_L_m_C_k * np.log( p_L_m_C_k / (p_L_m * p_C_k))
    for label in labels.keys():
        p_L_m = labels[label] / N
        H_L -= p_L_m * np.log(p_L_m)
    
    NMI = I/((H_C + H_L)/2)
    return NMI

if __name__ == "__main__":
    start = time.time()
    # savedir = '/media/hieu/DATA/ShareOS/PythonVirtualEnv/WorkSpace/test/topic model/GMM/generated_files'

    # len_doc, len_bag, labels = pkl.load(open(os.path.join(savedir, 'meta.data'), 'rb'))
    # labels = np.array(labels)
    # print("Loading from file...")
    # X = np.zeros((len_doc, len_bag))
    # with open(savedir + '/training_set.data', 'r+b') as f:
    #     for i in range(len_doc):
    #         X[i] = pkl.load(f)

    # print("X.shape = ", X.shape)

    X, origin_labels = demo_data()
    K = 3
    gmm = GMM(dim = X.shape[1], K= K, data = X, init_method="kmeans", kmeans_iter=1, iters = 10, run_method="EM")
    print(gmm)
    plot_sample(X, origin_labels, gmm.clusters)

    # make confusion table
    labels = []
    for x in X:
        l = np.argmax([cluster.pdf(x) for cluster in gmm.clusters])
        labels.append(l)
    confusion_table = []
    for k in range(K):
        confusion_table.append({})
        # print("___________________________________________________________")
        for i in range(len(X)):
            label = labels[i]
            if label == k:
                if confusion_table[k].has_key(origin_labels[i]):
                    confusion_table[k][origin_labels[i]] += 1
                else:
                    confusion_table[k][origin_labels[i]] = 1
        # print("cluster {} has {} docs".format(k, len(X[labels[-1] == k])))
        # for label in confusion_table[k].keys():
        #     print("{} has {} docs".format(label, confusion_table[k][label]))
    labels_stats = {}
    for i in range(len(X)):
        if labels_stats.has_key(origin_labels[i]):
            labels_stats[origin_labels[i]] += 1
        else:
            labels_stats[origin_labels[i]] = 1

    print("Purity = ", purity(confusion_table, len(X)))
    print("NMI = ", NMI(confusion_table, labels_stats, len(X)))
    # bgm = BGM(n_components=20)
    # bgm.fit(X)
    # print(bgm.means_)
    # print(bgm.covariances_)
    plt.show()    
    print("Done in {} seconds".format(time.time() - start))
