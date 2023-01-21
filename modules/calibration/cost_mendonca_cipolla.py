import numpy as np

def cost_mendonca_cipolla(F, X, Method):
    '''
    Parameters:
        F -> Fundamental matrix b/w pair of 2 images.
        X -> Approximate intrinsic values.
        Method -> Type of cost function to use (1 or 2).

    Returns:
        E -> Computer errors for different pair of images.
    '''
    # transform intrinsics to matrix format.
    K = np.array([[X[0], X[1], X[2]],
                  [0,    X[3], X[4]],
                  [0,    0,    1]])

    E = []

    N = F.shape[2]

    Den = N * (N - 1) / 2.

    for i in range(0, N-1):
        for j in range(i+1, N):

            EM = np.matmul(K.T, np.matmul(F[:, :, i, j], K))
            _, D, _ = np.linalg.svd(EM)
            
            r = D[0]
            s = D[1]

            if Method == "1":
                E1 = (1 / Den) * ((r - s) / s)
            elif Method == "2":
                E1 = (1 / Den) * ((r - s) / (r + s))
            
            E.append(E1)

    return E
