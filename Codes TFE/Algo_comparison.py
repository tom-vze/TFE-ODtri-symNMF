import numpy as np
import matplotlib.pyplot as plt

from ODtri_symNMF import ODtri_symNMF
from ODtri_symNMF_2 import ODtri_symNMF_2
from Compute_alpha import Compute_alpha

"""
This script is used to compare ODtri_symNMF and ODtri_symNMF_2
See "4.1.2 Comparaison des versions de l’algorithme"
    
"""

maxiter = 50
sizes = np.arange(6,40,3)
ranks = [2, 3, 4, 5]

for r in ranks :
    
    # Stock every data needed for futher analysis
    alpha_zero = []

    eb1 = []
    eb2 = []
    
    er1 = []
    er2 = []

    W_b1_dict = {}
    W_b2_dict = {}

    S_b1_dict = {}
    S_b2_dict = {}

    W_r1_dict = {}
    W_r2_dict = {}

    S_r1_dict = {}
    S_r2_dict = {}
    
    # S = I initialisation
    S_init = np.zeros((r,r))
    for s in range(r):
        S_init[s,s] = 1
    
    i = 0
    for n in sizes :
        print('\n ------', n, '------ \n')
        
        # Random W initialisation
        W_init = np.random.rand(n,r)
        
        # First solution computed for scaling purposes
        Sol_init = W_init@S_init@W_init.transpose()
        for d in range(n):
            Sol_init[d,d] = 0
        
        "----- Binary -----"
        
        # Creation of synthetic binary data
        X_b = np.random.randint(2, size = (n,n))
        X_b = np.triu(X_b) + np.tril(X_b.transpose(),-1)
        
        # Scaling
        alpha_b = Compute_alpha(X_b, Sol_init)
        W_init_b_scaled = W_init * (alpha_b**0.5)
        
        # Run both algo
        W_b1, S_b1, e_b1 = ODtri_symNMF(X_b, W_init_b_scaled, S_init, maxiter)
        W_b2, S_b2, e_b2 = ODtri_symNMF_2(X_b, W_init_b_scaled, S_init, maxiter)
        
        # Stock important outputs
        if alpha_b == 0:
            alpha_zero.append(i)
        
        eb1.append(e_b1[-1])
        eb2.append(e_b2[-1])
        
        W_b1_dict[n] = W_b1
        W_b2_dict[n] = W_b2
    
        S_b1_dict[n] = S_b1
        S_b2_dict[n] = S_b2
    
        
        "----- Real -----"
        
        # Creation of synthetic real data
        X_r = np.random.rand(n,n)
        X_r = np.triu(X_r) + np.tril(X_r.transpose(),-1)
        
        # Scaling
        alpha_r = Compute_alpha(X_r, Sol_init)
        W_init_r_scaled = W_init * (alpha_r**0.5)
        
        # Run both algo
        W_r1, S_r1, e_r1 = ODtri_symNMF(X_r, W_init_r_scaled, S_init, maxiter)
        W_r2, S_r2, e_r2 = ODtri_symNMF_2(X_r, W_init_r_scaled, S_init, maxiter)
        
        # Stock important outputs
        er1.append(e_r1[-1])
        er2.append(e_r2[-1])
        
        W_r1_dict[n] = W_r1
        W_r2_dict[n] = W_r2
    
        S_r1_dict[n] = S_r1
        S_r2_dict[n] = S_r2
    
        i += 1
    
    "----- Comparison graphs -----"
    
    # Creation of 2 graphs
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # First 
    axes[0].plot(sizes, eb1, color = 'blue', label='algo 1')
    axes[0].plot(sizes, eb2, color = 'green', label='algo 2')
    axes[0].set_title('Données synthétiques binaires')
    axes[0].set_xlabel('dimension de la matrice (n)')
    axes[0].set_ylabel("erreur relative ||X-WSW'||_1,OD / ||X||_1,OD")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True)
    axes[0].legend()
    
    # Second
    axes[1].plot(sizes, er1, color = 'blue', label='algo 1')
    axes[1].plot(sizes, er2, color = 'green', label='algo 2')
    axes[1].set_title('Données synthétiques réelles positives')
    axes[1].set_xlabel('dimension de la matrice (n)')
    axes[1].set_ylabel("erreur relative ||X-WSW'||_1,OD / ||X||_1,OD")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True)
    axes[1].legend()
    
    # Plot
    plt.suptitle('Comparaison des algorithmes, factorisation rang {}'.format(r), fontsize=16)
    plt.tight_layout()
    plt.show()
    
    
    "----- Alpha = 0 graph -----"
    
    eb1 = np.array(eb1)
    eb2 = np.array(eb2)
    
    plt.figure(figsize=(8, 6))
    plt.plot(sizes, eb1, color = 'blue', label='algo 1')
    plt.plot(sizes, eb2, color = 'green', label='algo 2')
    plt.scatter(sizes[alpha_zero], eb1[alpha_zero], facecolors='none', edgecolors='r', label='alpha = 0')
    plt.scatter(sizes[alpha_zero], eb2[alpha_zero], facecolors='none', edgecolors='r')
    plt.title('Comparaison des algorithmes, factorisation rang {}'.format(r), y=1.1)
    plt.suptitle('Mise en évidence de alpha', y=0.95)
    plt.xlabel('dimension de la matrice (n)')
    plt.ylabel("erreur relative ||X-WSW'||_1,OD / ||X||_1,OD")
    plt.grid(True)
    plt.legend()
    plt.show()
