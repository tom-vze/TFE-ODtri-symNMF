"""
Computes an optimal solution of
    min |a-bx|
    x

Inputs : 
    a, b    : (1 x d) vectors with the same dimension

Output :
    x       : (1 x 1) optimal solution
    
This function comes from the paper
"Dimensionality Reduction, Classification, and Spectral Mixture Analysis
using Nonnegative Underapproximation", N. Gillis and R.J. Plemmons,
Optical Engineering 50, 027001, February 2011.
Available on http://sites.google.com/site/nicolasgillis/code
"""

def weighted_median(a,b):
    
    #Reduce the problem for nonzero entries of b
    index = []
    for i in range(len(b)):
        if abs(b[i]) < 1e-16:
            index.append(i)
    for j in sorted(index, reverse=True):
        del a[j]
        del b[j]

    # If b is empty then h is full of zeros and we are trying to minimize a
    # constant function => we return x=1 to keep this cluster alive.
    if len(b) == 0:
        x = 1
        return x
    
    #We compute the kinks of the function and sort them
    a = [ai/bi for ai,bi in zip(a,b)]

    # We normalize the x-axis    
    b = [x/sum(b) for _,x in sorted(zip(a,b))]

    a = sorted(a)

    # Main loop to find the kink where the overall slope starts climbing again
    i = 0
    cumsum = 0
    while cumsum < 0.5:
        cumsum += b[i]
        x = a[i]
        i += 1

    return x
