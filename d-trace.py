# Chester Holtz - chholtz@ucsd.edu
# d-trace precision Estimator

import numpy as np
from numpy import linalg as la
from sklearn.linear_model import Lasso

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

MAXIT = 1000
TOL = 10**(-7)
EPS = 10**(-2)

def dict_stats(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        vals = [d[key] for d in dict_list]
        mean_dict[key] = (round(np.mean(vals),2), round(np.std(vals),2))
    return mean_dict

def gen_5folds(X):
    X_folds = np.split(X, 5, axis=1)
    S_folds = []
    for x_fold in X_folds:
        S_folds.append(np.cov(x_fold))
    return X_folds, S_folds

def risk(T,T_star):
    risk = {'fro':0, 'op':0, 'mat':0, 'tp':0, 'tn':0}
    # frobenius risk
    risk['fro'] = la.norm(T-T_star,'fro')
    # operator risk
    risk['op'] = la.norm(T-T_star,2)
    # matrix l1infty risk
    risk['mat'] = la.norm(la.norm((T-T_star),1,axis=0),np.inf)
    # sparsity accuracy
    risk['tp'] = np.sum(np.logical_and(T == 0, T_star == 0))/(T.shape[0]*T.shape[1])
    risk['tn'] = np.sum(np.logical_and(T != 1, T_star != 1))/(T.shape[0]*T.shape[1])
    return risk

def G(A, B):
    #l, U_A = la.eig(A) # numerical instabillity makes life miserable
    U_A, l, V_A = la.svd(A, full_matrices=True)
    l = l.real
    U_A = U_A.real
    S_A = np.diag(l)

    def c(i,j):
        i = i.astype(int)
        j = j.astype(int)
        return 2/(l[i] + l[j])
    C = np.fromfunction(lambda i, j: c(i,j), (l.shape[0],l.shape[0]))
    return U_A.dot( np.multiply(U_A.T.dot(B).dot(U_A), C) ).dot(U_A.T)

def S(A, l):
    def a(i,j):
        i = i.astype(int)
        j = j.astype(int)
        return np.where(i == j, A[i,j], 0) + \
               np.where((A[i,j] > l) & (i!=j), A[i,j] - l, 0) + \
               np.where((A[i,j] < -l)  & (i!=j), A[i,j] + l, 0)
    return np.fromfunction(lambda i, j: a(i,j), (A.shape[0],A.shape[1]))

def p(X):
    U_X, l, V_X = la.svd(X, full_matrices=True)
    e = EPS*np.ones(l.shape[0])
    return U_X.dot(np.diag(np.maximum(l, e))).dot(U_X.T)

def scad(l,th,a):
    for i in range(th.shape[0]):
        if th[i] <= 2*l[i]:
            np.sign(th[i])*(th[i] - l[i])
        elif 2*l[i] <= th[i]  and th[i] <= a*l[i]:
            th[i] = np.sign(th[i])*((a-1)*th[i] - a*l)/(a-2)
        elif a*l[i] <= th[i]:
            th[i] = th[i]
    return th

def obj(th, l_n, S_hat):
    return 1/2 * th.dot(th).dot(S_hat) - np.trace(th) + l_n * la.norm(th - np.diag(np.diag(th)),1)

def update_lambda(l, th, a):
    if not th.any():
        return np.zeros(th.shape[0])
    return l * np.less_equal(th,l) + np.clip(np.divide((a*l - th),(a-1)*l),0,np.inf) * np.greater(th,l)

def lla(L_init, Th_0_init, S_hat, rho, l_init):
    a = 3
    th_prev = Th_0_init
    l_n = l_init
    for i in tqdm(range(MAXIT), desc="lla"):
        th_k = alg1(L_init, Th_0_init, S_hat, rho, l_n)
        l_n = update_lambda(l_n, np.abs(th_k), a)
        if la.norm(th_k - th_prev, np.inf) <= TOL:
            break
        th_prev = th_k
    return th_k, l_n


def alg1(L_init, Th_0_init, S_hat, rho, l_n):
    Th_init = Th_0_init = la.inv(np.diag(np.diag(S_hat)))
    I = np.eye(S_hat.shape[0])
    L_init = np.zeros(S_hat.shape)

    Th = Th_init
    Th_0 = Th_1 = Th_0_init
    L_0 = L_1 = L_init

    t = tqdm(range(MAXIT))
    for k in t:
        Th_prev = Th
        Th_0_prev = Th_0
        Th_1_prev = Th_1

        Th = G(S_hat + 2*rho * I, I + rho * Th_0 + rho*Th_1 - L_0 - L_1)

        Th_0 = S(Th + 1.0/rho * L_0, 1.0/rho * l_n)
        Th_1 = p(Th + 1.0/rho * L_1)

        L_0 =  L_0 + rho*(Th - Th_0)
        L_1 =  L_1 + rho*(Th - Th_1)

        # convergence
        t.set_description(str(('alg1',la.norm(Th - Th_prev,'fro')/max(1, la.norm(Th, 'fro'), la.norm(Th_prev,'fro')), \
                               la.norm(Th_0 - Th_0_prev,'fro')/max(1, la.norm(Th_0, 'fro'), la.norm(Th_0_prev,'fro')), \
                               la.norm(Th_1 - Th_1_prev,'fro')/max(1, la.norm(Th_1, 'fro'), la.norm(Th_1_prev,'fro')))))

        if la.norm(Th - Th_prev,'fro')/max(1, la.norm(Th, 'fro'), la.norm(Th_prev,'fro')) < TOL and \
           la.norm(Th_0 - Th_0_prev,'fro')/max(1, la.norm(Th_0, 'fro'), la.norm(Th_0_prev,'fro')) < TOL and \
           la.norm(Th_1 - Th_1_prev,'fro')/max(1, la.norm(Th_1, 'fro'), la.norm(Th_1_prev,'fro')) < TOL:
           break
    return Th

def alg2(L_init, Th_0_init, S_hat, rho, l_n):
    Th_init = Th_0_init = la.inv(np.diag(np.diag(S_hat)))
    I = np.eye(S_hat.shape[0])
    L_init = np.zeros(S_hat.shape)

    Th = Th_init
    Th_0 = Th_0_init
    L = L_init

    t = tqdm(range(MAXIT))
    for k in t:
        Th_prev = Th
        Th_0_prev = Th_0

        Th = G(S_hat + rho * I, I + rho * Th_0 - L)
        Th_0 = S(Th + 1.0/rho * L, 1.0/rho * l_n)
        L =  L + rho*(Th - Th_0)

        # convergence
        t.set_description(str(('alg2',la.norm(Th - Th_prev,'fro')/max(1, la.norm(Th, 'fro'), la.norm(Th_prev,'fro')), \
                               la.norm(Th_0 - Th_0_prev,'fro')/max(1, la.norm(Th_0, 'fro'), la.norm(Th_0_prev,'fro')))))
        if la.norm(Th - Th_prev,'fro')/max(1, la.norm(Th, 'fro'), la.norm(Th_prev,'fro')) < TOL and \
           la.norm(Th_0 - Th_0_prev,'fro')/max(1, la.norm(Th_0, 'fro'), la.norm(Th_0_prev,'fro')) < TOL:
           break
    s = la.svd(Th, compute_uv=False)

    if min(s) < EPS:
        Th = alg1(L_init, Th, S_hat, rho, l_n)

    return Th

def cv_lambda(L_init, Th_0_init, X, rho, l_init):
    a = 3
    X_folds, S_folds = gen_5folds(X)
    errs = []
    for i, fold in enumerate(tqdm(X_folds, desc='cv')):
        X = np.concatenate(X_folds[:i] + X_folds[(i + 1):])
        S_hat = np.mean(S_folds[:i] + S_folds[(i + 1):], axis=0)
        th, l_n = lla(L_init, Th_0_init, S_hat, rho, l_init)
        errs.append(obj(th,l_n,S_folds[i]))
    return np.sum(errs)/len(X_folds)

# Experiments from sec. 4
n = 400 # samples size
p12 = 500 # dimension
p3 = 484

mean12 = np.zeros(p12)
mean3 = np.zeros(p3)

PR0 = np.eye(p12)
COV0 = la.inv(PR0)

def pr1(i,j):
    return np.where(i == j, 1, 0) + np.where((1 <= np.abs(i-j)) & (np.abs(i-j) <= 2), 0.2, 0)
PR1 = np.fromfunction(lambda i,j: pr1(i,j),(p12,p12))
COV1 = la.inv(PR1)
def pr2(i,j):
    return np.where(i == j, 1, 0) + np.where((1 <= np.abs(i-j)) & (np.abs(i-j) <= 4), 0.2, 0)
PR2 =  np.fromfunction(lambda i, j: pr2(i,j), (p12,p12))
COV2 = la.inv(PR2)
def pr3(i,j):
    return np.where(i == j, 1, 0) + \
           np.where((i % pow(p3,1/2) != 0) & (j == i+1), 0.2, 0) + \
           np.where((i % pow(p3,1/2) == 0) & (j == i + pow(p3,1/2)), 0.2, 0)
PR3 =  np.fromfunction(lambda i, j: pr3(i,j), (p3,p3))
COV3 = la.inv(PR3)

X0 = np.random.multivariate_normal(mean12, COV0, n).T
X1 = np.random.multivariate_normal(mean12, COV1, n).T
X2 = np.random.multivariate_normal(mean12, COV2, n).T
X3 = np.random.multivariate_normal(mean3, COV3, n).T

test_cases = [(X0, PR0), (X1, PR1), (X2, PR2), (X3, PR3)]
rho = 1
l_init = 0.25
R = []
for i, (X, S_star) in tqdm(enumerate(test_cases),desc="test cases"):
    S_hat = np.cov(X)

    Th_0_init = np.zeros(S_hat.shape)
    L_init = np.zeros(S_hat.shape)

    risks = []
    for j in tqdm(range(100), desc="trials"):
        th, l_n = lla(L_init, Th_0_init, S_hat, rho, l_init)
        th = np.around(th, 6)
        #th = alg2(L_init, Th_0_init, S_hat, rho, l_init)
        #print(cv_lambda(L_init, Th_0_init, X, rho, l_init))
        #tqdm.write(str(risk(th, PR1)))
        r = risk(th,S_star)
        risks.append(r)
    tqdm.write(str((i,dict_stats(risks))))
    R.append(dict_stats(risks))
