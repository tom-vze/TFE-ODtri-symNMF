from weighted_median import weighted_median

"""
Goal is to scale WSW'

Computes an optimal solution of
    min   |X-alpha*(WSW')|
   alpha
Where W and WSW' are flatten as vectors.

Inputs : 
    X               : (n x n) Reference matrix
    W_approx        : (n x n) matrix to scale -> WSW'

Output :
    alpha           : (1 x 1) optimal solution -> scaling factor
    
"""

def Compute_alpha(X, X_approx):
    
    # Flatten 2d matrices in 1d vectors
    X_flat = X.flatten()
    X_approx_flat = X_approx.flatten()
    
    # Convert to list to fit weighted_median needs
    X_flat_list = X_flat.tolist()
    X_approx_flat_list = X_approx_flat.tolist()
    
    # Computes alpha
    alpha = weighted_median(X_flat_list, X_approx_flat_list)
    
    return alpha

