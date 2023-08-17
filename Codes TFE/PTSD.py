import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import KMeans

from ODtri_symNMF import ODtri_symNMF
from ODtri_symNMF_2 import ODtri_symNMF_2
from Compute_alpha import Compute_alpha

"""
This script is used to test the algo on psychiatric data
See "4.2 Données psychiatriques"

Note : To compute the mean of the errors, 
uncomment declaration of related variables,
uncomment the for loop
and indent lignes 40 to 97
    
"""

X = np.loadtxt('X_psy.csv', delimiter=',')

n = np.shape(X)[0]
r = 3
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
alpha_rand = Compute_alpha(X, Sol_init_rand)
W_init_rand_scaled = W_init_rand * (alpha_rand**0.5)

# Keep every line max value
W_rand_max_val = np.zeros_like(W_init_rand_scaled)
max_indices = np.argmax(W_init_rand_scaled, axis=1)
for i in range(n):
    W_rand_max_val[i, max_indices[i]] = W_init_rand_scaled[i, max_indices[i]]

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
alpha_kmeans = Compute_alpha(X, Sol_init_kmeans)
W_init_kmeans_scaled = W_init_kmeans * (alpha_kmeans**0.5)

# Keep every line max value
W_kmeans_max_val = np.zeros_like(W_init_kmeans_scaled)
max_indices = np.argmax(W_init_kmeans_scaled, axis=1)
for i in range(n):
    W_kmeans_max_val[i, max_indices[i]] = W_init_kmeans_scaled[i, max_indices[i]]

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

"----- Graphs -----"

W_matrices = [Wk1, Wr1, Wk2, Wr2]
c2 = 0
# For each W given by 4 versions of the algo-init combination
for w in W_matrices:
    
    WWT = []
    
    # Comutes each rank 1 WW' matrices
    for i in range(r):
        temp = np.outer(w[:,i],w[:,i])
        for d in range(n):
            temp[d,d] = 0
        WWT.append(temp)
    
    # Creation of 3 graphs
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    graph_names = ['W(:,1)*W(:,1)^T', 'W(:,2)*W(:,2)^T', 'W(:,3)*W(:,3)^T']
    
    # Computes weights scale
    all_edge_weights = [edge[2]["weight"] for matrix in WWT for edge in nx.from_numpy_matrix(matrix).edges(data=True)]
    max_weight = max(all_edge_weights)
    
    # Create each graph
    for i, matrix in enumerate(WWT):
        G = nx.from_numpy_matrix(matrix)
        pos = nx.shell_layout(G)
        edge_weights = [edge[2]["weight"] for edge in G.edges(data=True)]
        normalized_widths = [5 * (weight / max_weight) for weight in edge_weights]
        node_labels = {node: node + 1 for node in G.nodes()}
        nx.draw(G, pos, ax=axes[i], with_labels=True, labels=node_labels, node_size=500, font_size=10, edge_color=edge_weights, width=normalized_widths, edge_cmap=plt.cm.Blues)
        axes[i].set_title(graph_names[i])
    
    # Plot
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=max_weight))
    sm.set_array([])
    plt.colorbar(sm, ax=axes[i])
    plt.suptitle('Algo {}, initialisation {}'.format(num[c2],init[c2]), fontsize = 15)
    plt.tight_layout()
    plt.show()
    
    "----- Clusters computation -----"
    
    clusters = [[] for _ in range(r)]

    for row_num in range(n):
        row = w[row_num, :]
        max_col_index = np.argmax(row)
        clusters[max_col_index].append(row_num+1)
    
    print('Algo {}, initialisation {}'.format(num[c2],init[c2]))
    for clust_index, clust_list in enumerate(clusters):
        print(f"Cluster {clust_index+1}: {clust_list}".format())
    
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