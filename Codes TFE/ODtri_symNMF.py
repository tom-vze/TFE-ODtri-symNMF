import numpy as np

from weighted_median import weighted_median
from Compute_alpha import Compute_alpha

"""
Computes a solution of
        min     ||X-WSW'||_OD,1 (except the diagonal elements)
    W,S >= 0

Inputs :
    X       : (n x n) symmetric matrix to factorize
    W       : (n x r) initial matrix W
    S       : (r x r) initial symmetric matrix S
    maxiter : maximum number of iterations performed
 
Note that X, W and S have to be numpy arrays

Outputs :
    W, S    : nonnegatives matrices s.t. WSW' approximates X
    error   : relative error at each iteration
    
"""

def ODtri_symNMF(X, W, S, maxiter):
    
    X = X.copy()
    W = W.copy()
    S = S.copy()
    
    n = np.shape(X)[0]
    r = np.shape(S)[0]
    error = np.zeros(maxiter)
    
    # Ignore the diagonal for further computations
    for x in range(n):
        X[x,x] = 0
    
    for it in range(maxiter):
        
        "----- W update -----"
        for i in range(n):
            for k in range(r):
                a = []
                b = []
                for j in range(n):
                     if j != i:
                        temp = 0
                        for k_bis in range(r):
                            if k_bis != k:
                                temp += W[i,k_bis]*S[k_bis]@W[j]
                        a.append(X[i,j]-temp)
                        b.append(S[k]@W[j])       
                W[i,k] = weighted_median(a, b)
                if W[i,k] < 0:
                    W[i,k] = 0
        
        "----- S update -----"
        for k in range(r):
            for p in range(r):
                a = []
                b = []
                for i in range(n):
                    for j in range(n):
                        if j != i : 
                            temp = 0
                            for k_bis in range(r):
                                for p_bis in range(r):
                                    if k_bis != k or p_bis != p:
                                        temp += W[i,k_bis]*S[k_bis,p_bis]*W[j,p_bis]
                            a.append(X[i,j]-temp)
                            b.append(W[i,k]*W[j,p])
                S[k,p] = weighted_median(a, b)
                if S[k,p] < 0:
                    S[k,p] = 0
                S[p,k] = S[k,p]
        
        "----- relative error computation -----"
        Sol = W@S@W.transpose()
        for d in range(n):
            Sol[d,d] = 0
        
        error[it] = 0
        for i in range(n - 1):
            R = X[i, i + 1:] - Sol[i, i + 1:]
            error[it] += np.sum(np.abs(R))
            
        error[it] = 2*error[it]/sum(sum(X))
        
        if error[it] == 0:
            print('Des matrices W et S optimales ont été trouvées en', it+1,'itérations')
            break
        
        else:
            if it > 1:
                progression = (error[it]-error[it-1])/error[it]
                if progression > -1e-15:
                    error[it+1:] = error[it]
                    print("L'erreur ne décroit plus significativement, algorithme stoppé prématurément après", it+1, 'itérations')
                    break
        
        "----- Scaling -----"
        alpha = Compute_alpha(X, Sol)
        W = W * (alpha**0.5)
        
    return W, S, error