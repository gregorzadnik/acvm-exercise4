import numpy as np
import math
from ex4_utils import kalman_step
import matplotlib.pyplot as plt
from enum import Enum
import sympy as sp


class models(Enum):
    RW = 'RW'
    NCV = 'NCV'
    NCA = 'NCA'


def prepare_matrices(model: models, q, r):
    F, L, H, R = None, None, None, None
    if model == models.RW:
        F = [[0, 0],
              [0, 0]]
        
        L = [[1, 0],
             [0, 1]]

        H = np.array(
            [[1, 0],
             [0, 1]],
            dtype="float")
        
        R = r * np.array([[1, 0],
                          [0, 1]],
                         dtype="float")
        
    elif model == models.NCV:
        F = [[0, 0, 1, 0],
             [0, 0, 0, 1],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]
        
        L = [[0, 0],
                    [0, 0],
                    [1, 0],
                    [0, 1]]
        
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]],
                     dtype="float")

        R = r * np.array([[1, 0],
                          [0, 1]],
                         dtype="float")
        
    elif model == models.NCA:
        F = [[0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]]
        
        L = [[0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [1, 0],
                    [0, 1]]
        
        H = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0]],
                     dtype="float")

        R = r * np.array(
            [[1, 0],
             [0, 1]],
            dtype="float")

    T = sp.symbols('T')
    F = sp.Matrix(F)
    L = sp.Matrix(L)
    Fi = np.array(sp.exp(F*T).subs(T, 1), dtype= float)
    Q = sp.integrate((Fi * L) * q * (Fi * L).T, (T, 0, T)).subs(T, 1)
    Q = np.array(Q, dtype=float)

    return Fi, H, Q, R 


def plot_trajectories(gt, model: models, params, ax):
    # Plot ground truth
    x, y = gt
    q, r = params

    ax.plot(gt[0], gt[1], '-o', c='red', linewidth=1, fillstyle = 'none')
    
    Fi, H, Q, R = prepare_matrices(model, q, r)
    sx = np.zeros((x.size, 1), dtype=np.float32).flatten()
    sy = np.zeros((y.size, 1), dtype=np.float32).flatten()
    sx[0]= x[0]
    sy[0] = y[0]

    state = np.zeros((Fi.shape[0], 1), dtype=np.float32).flatten()
    state[0] = x[0]
    state[1] = y[0]

    V = np.eye(Fi.shape[0],dtype=np.float32)

    for j in range(1, x.size):
        state, V, _, _ = kalman_step(Fi, H, Q, R,
        np.reshape(np.array([x[j], y[j]]), (-1, 1)),
        np.reshape(state, (-1, 1)),
        V)
        
        sx[j] = state[0]
        sy[j] = state[1]

    ax.plot(sx, sy, '-o', c='blue', linewidth=1, fillstyle = 'none')
    ax.title.set_text(f'{model.name}: q = {q}, r = {r}')
    ax.set_xticks([])
    ax.set_yticks([])


if __name__ == "__main__":
    # Prepare necessary axes 
    
    # Copy function for random trajectory generation
    N = 40
    v_ = np.linspace (5 *math.pi, 0 , N )
    x_ = np.cos (v_) * v_
    y_ = np.sin (v_) * v_
    
    v_quad = np.linspace(5*math.pi, 0, 35)
    r = 1 * np.sin(2 * v_quad)
    x_quad = r * np.cos(v_quad)
    y_quad = r * np.sin(v_quad)
    
    v_liss = np.linspace(5*math.pi, 0, 10)
    x_liss = 1* np.sin(1.5 * v_liss + np.pi/2)
    y_liss = 1* np.sin(2 * v_liss)
    
    gt_ =[(x_, y_), (x_quad, y_quad), (x_liss, y_liss)]
    
    # Plot curves with different kalman filters and lines
    plt.ion()
    for gt in gt_:
        fig1, axs = plt.subplots(3, 5, figsize=(15, 9))
        parameters_set = [(100, 1), (5.0, 1), (1, 1), (1, 5), (1, 100)]
        for i, params in enumerate(parameters_set):
            plot_trajectories(gt, models.RW, params, axs[0,i])
            
        for i, params in enumerate(parameters_set):
            plot_trajectories(gt, models.NCV, params, axs[1, i])

        for i, params in enumerate(parameters_set):
            plot_trajectories(gt, models.NCA,  params, axs[2, i])
        
        plt.draw()
        plt.waitforbuttonpress()
        plt.close()