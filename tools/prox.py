import numpy as np
from scipy.linalg import schur, solve
import control
import slycot

from tools.matrix import Extend_matA

def Scaling_to_stabilize(A, bnd):
    p = A.shape[2]
    A_ex = Extend_matA(A)  # Extended matrix A
    a = bnd / np.max(np.abs(np.linalg.eigvals(A_ex)))
    A_stable = np.copy(A)  # Initialize A_stable as a copy of A
    for i in range(p):
      A_stable[:, :, i] = (a**(i+1)) * A[:, :, i]  

    A_ex_stable = Extend_matA(A_stable)  # Extended matrix A_stable
    return A_ex_stable, A_stable

def remove_bias(x_in, K, y, seuil):
    index_unb = np.where(np.abs(x_in) >= seuil)[0]
    K_unb = K[:, index_unb]
    x_out = np.copy(x_in)
    x_out[index_unb] = np.linalg.pinv(K_unb.T @ K_unb) @ K_unb.T @ y
    return x_out

def prox_stable(D, eta):
    U, s, V = np.linalg.svd(D)
    S = np.diag(np.minimum(s, eta))
    Dprox = U @ S @ V.T
    return Dprox

def prox_ML_D1(C, Phi, sigma_Q, gamma, D1, K):
    temp = (gamma * K) / (sigma_Q**2)
    D1prox = (temp @ C + D1) @ np.linalg.pinv(Phi @ temp + np.eye(Phi.shape[0]))
    return D1prox

def prox_L1plus(gamma, gammareg, D):
    temp = gamma * gammareg
    Dprox = np.sign(D) * np.maximum(0, np.abs(D) - temp)
    Dprox = np.maximum(Dprox, 0)
    return Dprox

def prox_L1(gamma, gammareg, D):
    temp = gamma * gammareg
    Dprox = np.sign(D) * np.maximum(0, np.abs(D) - temp)
    return Dprox

def prox_21(W, lambda_reg):
    Nw = W.shape[1]
    Norm_rows = np.sqrt(np.sum(W**2, axis=1))
    Norm_rows_large = np.tile(Norm_rows[:, np.newaxis], (1, Nw))
    X = W * np.maximum(1 - lambda_reg / Norm_rows_large, 0)
    return X

def sylv(A, B, C):
    m, n = C.shape

    ZA, TA = schur(A, output='complex')
    if B is None:
        ZB = ZA
        TB = TA.conj().T
        solve_direction = 'backward'
    elif np.allclose(A, B.conj().T):
        ZB = np.conjugate(ZA)
        TB = TA.T
        solve_direction = 'backward'
    else:
        ZB, TB = schur(B, output='complex')
        solve_direction = 'forward'

    F = ZA.conj().T @ C @ ZB

    if np.allclose(TA, np.diag(np.diag(TA))) and np.allclose(TB, np.diag(np.diag(TB))):
        L = -1 / (np.diag(TA)[:, None] + np.diag(TB)[None, :])
        Y = L * F
        X = ZA @ Y @ ZB.conj().T
        return X

    Y = np.zeros((m, n), dtype=np.complex_)
    p = np.diag(TA)
    idx = np.arange(m)

    if solve_direction == 'backward':
        kk = range(n - 1, -1, -1)
    else:
        kk = range(n)

    for k in kk:
        rhs = F[:, k] - Y @ TB[:, k]
        TA_shifted = TA.copy()
        np.fill_diagonal(TA_shifted, p + TB[k, k])
        Y[:, k] = solve(TA_shifted, -rhs)

    X = ZA @ Y @ ZB.conj().T
    return X


def lyap(A, B, C=None, E=None):
    A_ctrl = np.array(A, dtype=complex)  # Convert to complex for control library
    B_ctrl = np.array(B, dtype=complex) if B is not None else None
    C_ctrl = np.array(C, dtype=complex) if C is not None else None
    E_ctrl = np.array(E, dtype=complex) if E is not None else None

    if C_ctrl is None and E_ctrl is None:
        # Lyapunov equation: A*X + X*A' + B = 0
        X = control.lyap(A_ctrl, B_ctrl)
        if np.isrealobj(A) and np.isrealobj(B):
            X = np.real(X)

    elif C_ctrl is not None and E_ctrl is None:
        # Sylvester equation: A*X + X*B + C = 0
        X = control.slyap(A_ctrl, B_ctrl, -C_ctrl) # Note the -C for the form used by slyap
        if np.isrealobj(A) and np.isrealobj(B) and np.isrealobj(C):
            X = np.real(X)

    elif E_ctrl is not None and C_ctrl is None:
        # Generalized Lyapunov equation: A*X*E' + E*X*A' + B = 0
        try:
            X = control.lyap(A_ctrl, B_ctrl, E=E_ctrl)
            if np.isrealobj(A) and np.isrealobj(B) and np.isrealobj(E):
                X = np.real(X)
        except Exception as e:
            print(f"Error solving generalized Lyapunov equation: {e}")
            X = None
    else:
        raise ValueError("Incorrect combination of input arguments.")

    return X