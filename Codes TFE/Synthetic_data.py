import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from ODtri_symNMF import ODtri_symNMF
from ODtri_symNMF_2 import ODtri_symNMF_2
from Compute_alpha import Compute_alpha

"""
This script is used to compare random and kmeans initialisations
See "4.1.3 Comparaison des initialisations"
Also to show the relative error evolution
See "4.1.1 Convergence"
Uncomment lines 187 to 221 for convergence
    
"""


maxiter = 50
sizes = np.arange(6,40,3)
ranks = [2, 3, 4, 5]

alpha_bs_dict = {}
alpha_bsk_dict = {}

for r in ranks:
    
    # Stock every data needed for futher analysis
    alpha_bs = []
    alpha_bsk = []

    ebs1 = []
    ebsk1 = []
    
    ebs2 = []
    ebsk2 = []
    
    ers1 = []
    ersk1 = []

    ers2 = []
    ersk2 = []

    W_bs1_dict = {}
    W_bsk1_dict = {}
    
    W_bs2_dict = {}
    W_bsk2_dict = {}
    
    W_rs1_dict = {}
    W_rsk1_dict = {}
    
    W_rs2_dict = {}
    W_rsk2_dict = {}

    S_bs1_dict = {}
    S_bsk1_dict = {}
    
    S_bs2_dict = {}
    S_bsk2_dict = {}

    S_rs1_dict = {}
    S_rsk1_dict = {}
    
    S_rs2_dict = {}
    S_rsk2_dict = {}
    
    # S = I initialisation
    S_init = np.zeros((r,r))
    for s in range(r):
        S_init[s,s] = 1

    for n in sizes:
        print('\n ------', n, '------ \n')
        
        # Random W initialisation
        W_init_rand = np.random.rand(n,r)
        
        # First solution computed for scaling purposes
        Sol_init_rand = W_init_rand@S_init@W_init_rand.transpose()
        for d in range(n):
            Sol_init_rand[d,d] = 0
        
        "----- Binary -----"
        
        # Creation of synthetic binary data
        X_b = np.random.randint(2, size = (n,n))
        X_b = np.triu(X_b) + np.tril(X_b.transpose(),-1)
        
        # kmeans W initialisation
        kmeans_b = KMeans(n_clusters=r)
        kmeans_b.fit(X_b)
        W_init_b_kmeans = kmeans_b.cluster_centers_.transpose()
        Sol_init_b_kmeans = W_init_b_kmeans@S_init@W_init_b_kmeans.transpose()
        for d in range(n):
            Sol_init_b_kmeans[d,d] = 0       
        
        # Scaling on random init
        alpha_b_rand = Compute_alpha(X_b, Sol_init_rand)
        alpha_bs.append(alpha_b_rand)
        W_init_b_rand_scaled = W_init_rand * (alpha_b_rand**0.5)
        
        # Run both algo with random scaled init
        W_bs1, S_bs1, e_bs1 = ODtri_symNMF(X_b, W_init_b_rand_scaled, S_init, maxiter)
        W_bs2, S_bs2, e_bs2 = ODtri_symNMF_2(X_b, W_init_b_rand_scaled, S_init, maxiter)
        
        # Stock important outputs
        ebs1.append(e_bs1[-1])
        ebs2.append(e_bs2[-1])
        
        W_bs1_dict[n] = W_bs1
        W_bs2_dict[n] = W_bs2
    
        S_bs1_dict[n] = S_bs1
        S_bs2_dict[n] = S_bs2
        
        # Scaling on kmeans init
        alpha_b_kmeans = Compute_alpha(X_b, Sol_init_b_kmeans)
        alpha_bsk.append(alpha_b_kmeans)
        W_init_b_kmeans_scaled = W_init_b_kmeans * (alpha_b_kmeans**0.5)
        
        # Run both algo with kmeans scaled init
        W_bsk1, S_bsk1, e_bsk1 = ODtri_symNMF(X_b, W_init_b_kmeans_scaled, S_init, maxiter)
        W_bsk2, S_bsk2, e_bsk2 = ODtri_symNMF_2(X_b, W_init_b_kmeans_scaled, S_init, maxiter)
        
        # Stock important outputs
        ebsk1.append(e_bsk1[-1])
        ebsk2.append(e_bsk2[-1])
        
        W_bsk1_dict[n] = W_bsk1
        W_bsk2_dict[n] = W_bsk2
    
        S_bsk1_dict[n] = S_bsk1
        S_bsk2_dict[n] = S_bsk2


        "----- Real -----"
        
        # Creation of synthetic real data
        X_r = np.random.rand(n,n)
        X_r = np.triu(X_r) + np.tril(X_r.transpose(),-1)

        # kmeans W initialisation
        kmeans_r = KMeans(n_clusters=r)
        kmeans_r.fit(X_r)
        W_init_r_kmeans = kmeans_r.cluster_centers_.transpose()
        Sol_init_r_kmeans = W_init_r_kmeans@S_init@W_init_r_kmeans.transpose()
        for d in range(n):
            Sol_init_r_kmeans[d,d] = 0
        
        # Scaling on random init
        alpha_r_rand = Compute_alpha(X_r, Sol_init_rand)
        W_init_r_rand_scaled = W_init_rand * (alpha_r_rand**0.5)
        
        # Run both algo with random scaled init
        W_rs1, S_rs1, e_rs1 = ODtri_symNMF(X_r, W_init_r_rand_scaled, S_init, maxiter)
        W_rs2, S_rs2, e_rs2 = ODtri_symNMF_2(X_r, W_init_r_rand_scaled, S_init, maxiter)
        
        # Stock important outputs
        ers1.append(e_rs1[-1])
        ers2.append(e_rs2[-1])
        
        W_rs1_dict[n] = W_rs1
        W_rs2_dict[n] = W_rs2
    
        S_rs1_dict[n] = S_rs1
        S_rs2_dict[n] = S_rs2
        
        # Scaling on kmeans init
        alpha_r_kmeans = Compute_alpha(X_r, Sol_init_r_kmeans)
        W_init_r_kmeans_scaled = W_init_r_kmeans * (alpha_r_kmeans**0.5)
        
        # Run both algo with kmeans scaled init
        W_rsk1, S_rsk1, e_rsk1 = ODtri_symNMF(X_r, W_init_r_kmeans_scaled, S_init, maxiter)
        W_rsk2, S_rsk2, e_rsk2 = ODtri_symNMF_2(X_r, W_init_r_kmeans_scaled, S_init, maxiter)
        
        # Stock important outputs
        ersk1.append(e_rsk1[-1])
        ersk2.append(e_rsk2[-1])
        
        W_rsk1_dict[n] = W_rsk1
        W_rsk2_dict[n] = W_rsk2
    
        S_rsk1_dict[n] = S_rsk1
        S_rsk2_dict[n] = S_rsk2
        
        # "----- Convergence comparison graphs -----"
        
        # # Creation of 2 graphs
        # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # x = np.arange(1,maxiter+1,1)
        
        # # First
        # axes[0].plot(x, e_bs1, color = 'salmon', label='binaire, aléatoire')
        # axes[0].plot(x, e_bsk1, color = 'steelblue', label='binaire, kmeans')
        # axes[0].plot(x, e_rs1, color = 'lightgreen', label='réelles, aléatoire')
        # axes[0].plot(x, e_rsk1, color = 'orchid', label='réelles, kmeans')
        # axes[0].set_title('Algo 1')
        # axes[0].set_xlabel('itérations')
        # axes[0].set_ylabel("erreur relative ||X-WSW'||_1,OD / ||X||_1,OD")
        # axes[0].grid(True)
        # axes[0].legend()
        
        # # Second
        # axes[1].plot(x, e_bs2, color = 'salmon', label='binaire, aléatoire')
        # axes[1].plot(x, e_bsk2, color = 'steelblue', label='binaire, kmeans')
        # axes[1].plot(x, e_rs2, color = 'lightgreen', label='réelles, aléatoire')
        # axes[1].plot(x, e_rsk2, color = 'orchid', label='réelles, kmeans')
        # axes[1].set_title('Algo 2')
        # axes[1].set_xlabel('itérations')
        # axes[1].set_ylabel("erreur relative ||X-WSW'||_1,OD / ||X||_1,OD")
        # axes[1].grid(True)
        # #axes[1,0].legend()
        
        # # Plot
        # for ax in axes:
        #     ax.set_ylim(0, 1)
        # plt.suptitle('Convergence, matrice ({}x{}), factorisation rang {}'.format(n,n,r), fontsize=16)
        # plt.tight_layout()
        # plt.show()
    
    # Stock informations
    alpha_bs_dict[r] = alpha_bs
    alpha_bsk_dict[r] = alpha_bsk
    
    "----- Initialisation comparis graphs -----"
    
    # Creation of 4 graphs
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # First
    axes[0,0].plot(sizes, ebs1, color = 'purple', label='aléatoire')
    axes[0,0].plot(sizes, ebsk1, color = 'orange', label='kmeans')
    axes[0,0].set_title('Données synthétiques binaires, Algo 1')
    axes[0,0].set_xlabel('dimension de la matrice (n)')
    axes[0,0].set_ylabel("erreur relative ||X-WSW'||_1,OD / ||X||_1,OD")
    axes[0,0].set_ylim(0, 1)
    axes[0,0].grid(True)
    axes[0,0].legend()
    
    # Second
    axes[0,1].plot(sizes, ebs2, color = 'purple', label='aléatoire')
    axes[0,1].plot(sizes, ebsk2, color = 'orange', label='kmeans')
    axes[0,1].set_title('Données synthétiques binaires, Algo 2')
    axes[0,1].set_xlabel('dimension de la matrice (n)')
    axes[0,1].set_ylabel("erreur relative ||X-WSW'||_1,OD / ||X||_1,OD")
    axes[0,1].set_ylim(0, 1)
    axes[0,1].grid(True)
    
    # Third
    axes[1,0].plot(sizes, ers1, color = 'purple', label='aléatoire')
    axes[1,0].plot(sizes, ersk1, color = 'orange', label='kmeans')
    axes[1,0].set_title('Données synthétiques réelles positives, Algo 1')
    axes[1,0].set_xlabel('dimension de la matrice (n)')
    axes[1,0].set_ylabel("erreur relative ||X-WSW'||_1,OD / ||X||_1,OD")
    axes[1,0].set_ylim(0, 1)
    axes[1,0].grid(True)
    
    # Fourth
    axes[1,1].plot(sizes, ers2, color = 'purple', label='aléatoire')
    axes[1,1].plot(sizes, ersk2, color = 'orange', label='kmeans')
    axes[1,1].set_title('Données synthétiques réelles positives, Algo 2')
    axes[1,1].set_xlabel('dimension de la matrice (n)')
    axes[1,1].set_ylabel("erreur relative ||X-WSW'||_1,OD / ||X||_1,OD")
    axes[1,1].set_ylim(0, 1)
    axes[1,1].grid(True)
    
    # Plot
    plt.suptitle('Comparaison des initialisations, factorisation rang {}'.format(r), fontsize=16)  
    plt.subplots_adjust(hspace=0.5)
    plt.show()