import numpy as np

def get_skew_mat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def cost_kruppa_classical(F, X):
    '''
    Parameters:
        F -> Fundamental matrix b/w pair of 2 images.
        X -> Approximate intrinsic values.
    Returns:
        E -> Computer errors for different pair of images.
    '''
    # transform intrinsics to matrix format.
    K = np.array([[X[0], X[1], X[2]],
                  [0,    X[3], X[4]],
                  [0,    0,    1]])

    W_inv = np.matmul(K, K.T)

    E = []

    N = F.shape[2]

    for i in range(N-1):
        for j in range(i+1,N):
            
            A = np.matmul(F[:, :, i, j], np.matmul(W_inv, F[:, :, i, j].T))
            A /= np.linalg.norm(A, ord="fro")

            _, _, V = np.linalg.svd(F[:, :, i, j].T)

            V = V[:,-1]

            Epi = get_skew_mat(V)

            B = np.matmul(Epi, np.matmul(W_inv, Epi.T))

            B /= np.linalg.norm(B, ord="fro")

            E1 = A - B

            E.extend(E1[0,:3])
            E.extend(E1[1,1:3])

    return E
