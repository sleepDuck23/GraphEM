import numpy as np

def CreateAdjacencyAR1(N,rho):
    A = np.zeros((N,N))

    for i in range(N):
        for j in range(N):
            A[i, j] = rho ** abs((i-1)-(j-1))

    return A

def GenerateSynthetic_order_p(K,A,H,p,x0,sigma_P, sigma_Q, sigma_R):
    Ny, Nx = H.shape

    x = np.zeros((Nx,K))
    y = np.zeros((Ny,K))

    for pp in range(p):
        x[:, pp] = (x0.flatten() + sigma_P * np.random.randn(Nx)).flatten()

    for k in range(p, K):
        deterministic_state = np.zeros(Nx)
        for pp in range(p):
            deterministic_state += A @ x[:, k - pp - 1]
        x[:, k] = deterministic_state + sigma_Q * np.random.randn(Nx)
        y[:, k] = H @ x[:, k] + sigma_R * np.random.randn(Ny)

    return y, x