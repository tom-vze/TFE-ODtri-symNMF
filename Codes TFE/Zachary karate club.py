import numpy as np
from sklearn.cluster import KMeans

from ODtri_symNMF import ODtri_symNMF
from ODtri_symNMF_2 import ODtri_symNMF_2
from Compute_alpha import Compute_alpha

"""
This script is used to test the algo on Zachary's karate club dataset
See "4.3 Club de karaté de Zachary"

Note : To compute the mean of the errors, 
uncomment declaration of related variables,
uncomment the for loop
and indent lignes 44 to 101

Note : As explained in "4.3.1 Analyse de l’erreur",
scaling is skiped for this experiment.
If you want to test it anyway,
just uncoment lines related to scaling
and change W in algo inputs
    
"""

X = np.loadtxt('X_karate.txt')

n = np.shape(X)[0]
r = 4
maxiter = 100

# Uncomment for mean of error
# er1_moy = 0
# er2_moy = 0
# ek1_moy = 0
# ek2_moy = 0

# S = I initialisation
S_init = np.zeros((r,r))
for s in range(r):
    S_init[s,s] = 1
 
# Uncomment for mean of error
# for t in range(100):

"----- Random initialisation -----"
    
# W initialisation
W_init_rand = np.random.rand(n,r)

# First solution computed for scaling purposes
Sol_init_rand = W_init_rand@S_init@W_init_rand.transpose()
for d in range(n):
    Sol_init_rand[d,d] = 0

# Scaling    
# alpha_rand = Compute_alpha(X, Sol_init_rand)
# W_init_rand_scaled = W_init_rand * (alpha_rand**0.5)

# Keep every line max value
W_rand_max_val = np.zeros_like(W_init_rand)
max_indices = np.argmax(W_init_rand, axis=1)
for i in range(n):
    W_rand_max_val[i, max_indices[i]] = W_init_rand[i, max_indices[i]]

# Run both algo
Wr1, Sr1, er1 = ODtri_symNMF(X, W_rand_max_val, S_init, maxiter)
Wr2, Sr2, er2 = ODtri_symNMF_2(X, W_rand_max_val, S_init, maxiter)

# Uncomment for mean of error
# er1_moy += er1[-1]
# er2_moy += er2[-1]

"----- Kmeans initialisation -----"

# W initialisation
kmeans = KMeans(r)
kmeans.fit(X)
W_init_kmeans = kmeans.cluster_centers_.transpose()

# First solution computed for scaling purposes
Sol_init_kmeans = W_init_kmeans@S_init@W_init_kmeans.transpose()
for d in range(n):
    Sol_init_kmeans[d,d] = 0 

# Scaling
# alpha_kmeans = Compute_alpha(X, Sol_init_kmeans)
# W_init_kmeans_scaled = W_init_kmeans * (alpha_kmeans**0.5)

# Keep every line max value
W_kmeans_max_val = np.zeros_like(W_init_kmeans)
max_indices = np.argmax(W_init_kmeans, axis=1)
for i in range(n):
    W_kmeans_max_val[i, max_indices[i]] = W_init_kmeans[i, max_indices[i]]

# Run both algo
Wk1, Sk1, ek1 = ODtri_symNMF(X, W_kmeans_max_val, S_init, maxiter)
Wk2, Sk2, ek2 = ODtri_symNMF_2(X, W_kmeans_max_val, S_init, maxiter)

# Uncomment for mean of error     
# ek1_moy += ek1[-1]
# ek2_moy += ek2[-1]
 
# Uncomment for mean of error    
# er1_moy = er1_moy / 100
# er2_moy = er2_moy / 100
# ek1_moy = ek1_moy / 100
# ek2_moy = ek2_moy / 100

# For further display purposes
errors = [ek1[-1], er1[-1], ek2[-1], er2[-1]]
S_matrices = [Sk1, Sr1, Sk2, Sr2]
num = [1, 1, 2, 2]
init = ['kmeans', 'aléatoire', 'kmeans', 'aléatoire']

"----- Cluster computation -----"

W_matrices = [Wk1, Wr1, Wk2, Wr2]
c2 = 0
for w in W_matrices:
    
    max_values = np.max(w, axis=1)

    clusters = [[] for _ in range(r)]
    unclassified = []

    for row_num, max_value in enumerate(max_values):
        if max_value == 0 :
            unclassified.append(row_num+1)
        else:
            row = w[row_num, :]
            max_col_indices = np.where(row == max_value)[0]  
            for max_col_index in max_col_indices:
                clusters[max_col_index].append(row_num+1)
                
    print('Algo {}, initialisation {}'.format(num[c2],init[c2]))
    for clust_index, clust_list in enumerate(clusters):
        print(f"Cluster {clust_index+1}: {clust_list}")
    print('Non classés : {}'.format(unclassified))

    c2 += 1

"----- Matrices latex form print -----"

print('############## S ##############')
c = 0
for s in S_matrices :
    s = np.round(s, 2)
    latex_code = "\\begin{bmatrix}\n"
    for row in s:
        latex_code += " & ".join(map(str, row)) + " \\\\\n"
    latex_code += "\\end{bmatrix}"
    
    print(num[c],init[c],errors[c],'\n',latex_code,'\n')
    c += 1

print('############## W ##############')
c = 0
for w in W_matrices :
    w = np.round(w, 2)
    latex_code = "\\begin{bmatrix}\n"
    for row in w:
        latex_code += " & ".join(map(str, row)) + " \\\\\n"
    latex_code += "\\end{bmatrix}"
    
    print(num[c],init[c],'\n',latex_code,'\n')
    c += 1
