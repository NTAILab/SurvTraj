from .surv_mixup import SurvivalMixup
from sklearn.ensemble import RandomForestClassifier
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import concordance_index_censored
from sksurv.ensemble import RandomSurvivalForest

import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime
from .utility import get_str_array, sksurv_loader


from sklearn.manifold import TSNE
from scipy.stats import normaltest
from . import DEVICE
from torch import tensor, get_default_dtype as tens_type, no_grad


def draw_latent_space(model, *args):
    dim = model.vae.latent_dim
    if dim < 2:
        return None
    fig, ax = plt.subplots(1, 1)
    X = np.concatenate(args, axis=0)
    with no_grad():
        _, _, Z, mu, *_ = model(tensor(X, dtype=tens_type(), device=DEVICE))
        Z = Z.cpu().numpy()
    print('Latent space std:', np.std(Z, 0))
    _, p_val = normaltest(Z)
    print('p values of the normal test:', p_val)
    worst_p_args = np.argsort(p_val)[:2]
    worst_z = Z[:, worst_p_args]
    if dim > 2:
        Z = TSNE().fit_transform(Z)
    if dim == 2:
        fig.suptitle('Latent space 2 dim')
    else:
        fig.suptitle(f'Latent space {dim} dim, t-SNE')
    start_idx, end_idx = 0, 0
    for i in range(len(args)):
        end_idx += args[i].shape[0]
        cur_cls = Z[start_idx:end_idx]
        ax.scatter(*cur_cls.T)
        start_idx += args[i].shape[0]
    fig, ax = plt.subplots(1, 1)
    ax.scatter(*worst_z.T)
    fig.suptitle(f'Worst p vals projection (axis {worst_p_args})')
    return fig, ax

def x_experiment_linear():
    cls_centers = 0.1 * np.asarray(
        [
            [-2, 1],
            [-3, 4],
            [2, 2],
            [5, 5]
        ]
    ).astype(np.double)
    times = np.asarray(
        [
            5,
            10,
            5,
            10
        ]
    ).astype(np.double)
    n_per_cls = 50
    
    cls_points = np.stack(
        [np.random.normal(c, 0.025, (n_per_cls, cls_centers.shape[1])) for c in cls_centers], 0)
    x_train = []
    t_train = []
    clr = ['r', 'b']
    fig, ax = plt.subplots(1, 1)
    for i in range(0, cls_centers.shape[0], 2):
        mix_coef = np.random.uniform(0, 1, (n_per_cls, 1))
        X = cls_points[i] * mix_coef + cls_points[i + 1] * (1 - mix_coef)
        X += np.random.normal(0, 0.005, X.shape)
        x_train.append(X)
        mix_coef = mix_coef.ravel()
        y = times[i] * mix_coef + times[i + 1] * (1 - mix_coef)
        y += np.random.normal(0, 0.1, y.shape)
        t_train.append(y)
        ax.scatter(*X.T, c=clr[i // 2], s=0.8)
        ax.scatter(*cls_points[i].T, c=clr[i // 2], s=0.8)
        ax.scatter(*cls_points[i + 1].T, c=clr[i // 2], s=0.8)
        # ax3d.scatter(*X.T, y, c=clr[i // 2])
    x_train = np.concatenate(x_train, 0)
    x_train = np.concatenate((x_train, cls_points.reshape(-1, cls_centers.shape[1])), 0)
    d_train = np.ones(x_train.shape[0])
    t_train = np.concatenate(t_train, 0)
    t_train = np.concatenate((t_train, np.repeat(times, n_per_cls)), 0)
    y = get_str_array(t_train, d_train)
    
    model = SurvivalMixup(**mixup_kw)
    date_str = strftime('%m_%d %H_%M_%S', gmtime())
    model.fit(x_train, y, 'TensorBoard/linear/' + date_str)
    exp_points_2d = np.concatenate(
        (cls_centers, (cls_centers[None, -1] + cls_centers[None, -2]) / 2, (cls_centers[None, -3] + cls_centers[None, -4]) / 2),
        axis=0
    )
    x_ec, T, D = model.predict(exp_points_2d)
    print(T)
    ax.scatter(*cls_centers.T, c='k', marker='*')
    ax.scatter(*x_ec.T, c='m', marker='^')
    
    exp_points = np.concatenate([
            (cls_centers[None, -1] + cls_centers[None, -2]) / 2,
            (cls_centers[None, -3] + cls_centers[None, -4]) / 2,
            np.mean(cls_centers, axis=0, keepdims=True),
            cls_centers[None, 0],
            cls_centers[None, 3]
        ], axis=0,
    )
    
    x_e_all, T_all, D_all  = model.predict(x_train)
    
    fig = plt.figure()
    fig.suptitle('Reconstruction')
    ax3d = fig.add_subplot(111, projection='3d')
    for i in range(0, cls_centers.shape[0], 2):
        cur_points = np.concatenate((
                x_train[(i // 2) * n_per_cls : (i // 2 + 1) * n_per_cls],
                x_train[(i + 2) * n_per_cls : (i + 4) * n_per_cls]
            ), axis=0
        )
        cur_y = np.concatenate((
                t_train[(i // 2) * n_per_cls : (i // 2 + 1) * n_per_cls],
                t_train[(i + 2) * n_per_cls : (i + 4) * n_per_cls],
            ), axis=0
        )
        ax3d.scatter(*cur_points.T, cur_y, c=clr[i // 2])
    ax3d.scatter(*x_e_all.T, T_all, c='k')
    
    
    x_explain, T_life = model.predict_trajectory(exp_points, 100, t_train.min(), t_train.max())
    
    fig = plt.figure()
    fig.suptitle('Trajectories')
    ax3d = fig.add_subplot(111, projection='3d')
    for i in range(0, cls_centers.shape[0], 2):
        cur_points = np.concatenate((
                x_train[(i // 2) * n_per_cls : (i // 2 + 1) * n_per_cls],
                x_train[(i + 2) * n_per_cls : (i + 4) * n_per_cls]
            ), axis=0
        )
        cur_y = np.concatenate((
                t_train[(i // 2) * n_per_cls : (i // 2 + 1) * n_per_cls],
                t_train[(i + 2) * n_per_cls : (i + 4) * n_per_cls],
            ), axis=0
        )
        ax3d.scatter(*cur_points.T, cur_y, c=clr[i // 2])
    
    for i in range(exp_points.shape[0]):
        ax3d.scatter(*x_explain[i].T, T_life[i], c='k')
    
    fig = plt.figure()
    fig.suptitle('Sampling')
    ax3d = fig.add_subplot(111, projection='3d')
    
    
    x_smp, T_smp, D_smp = model.sample_data(500)
    
    for i in range(0, cls_centers.shape[0], 2):
        cur_points = np.concatenate((
                x_train[(i // 2) * n_per_cls : (i // 2 + 1) * n_per_cls],
                x_train[(i + 2) * n_per_cls : (i + 4) * n_per_cls]
            ), axis=0
        )
        cur_y = np.concatenate((
                t_train[(i // 2) * n_per_cls : (i // 2 + 1) * n_per_cls],
                t_train[(i + 2) * n_per_cls : (i + 4) * n_per_cls],
            ), axis=0
        )
        ax3d.scatter(*cur_points.T, cur_y, c=clr[i // 2])
    
    ax3d.scatter(*x_smp.T, T_smp, c='k')
    
    x_cls = (
        np.concatenate((
                x_train[(i // 2) * n_per_cls : (i // 2 + 1) * n_per_cls],
                x_train[(i + 2) * n_per_cls : (i + 4) * n_per_cls]
            ), axis=0
        )
        for i in range(0, cls_centers.shape[0], 2)
    )
    draw_latent_space(model, *x_cls)
    
    plt.show()
    
def x_experiment_spiral():
    spiral = lambda tau: 0.1 * np.stack((tau * np.sin(tau), tau * np.cos(tau)), axis=-1)
    n_per_cls = 100
    noise_lvl = 0.02
    pi = np.pi
    tau_bounds = [
        (pi / 3, 2.5 * pi / 2),
        (1.75 * pi, 4.5 * pi / 2)
    ]
    
    responses = [
        lambda tau: tau + np.random.normal(0, 0.1, tau.shape),
        lambda tau: tau + np.random.normal(0, 0.1, tau.shape),
    ]
    
    x_train = []
    t_train = []
    clr = ['r', 'b']
    fig, ax = plt.subplots(1, 1)
    for i in range(len(tau_bounds)):
        tau = np.random.uniform(*tau_bounds[i], (n_per_cls))
        X = spiral(tau)
        X += np.random.normal(0, noise_lvl, X.shape)
        x_train.append(X)
        t = responses[i](tau)
        t_train.append(t)
        ax.scatter(*X.T, c=clr[i], s=0.8)
    x_train = np.concatenate(x_train, 0)
    d_train = np.ones(x_train.shape[0])
    t_train = np.concatenate(t_train, 0)
    y = get_str_array(t_train, d_train)
    
    cls_centers = spiral(np.mean(tau_bounds, axis=-1))
    
    
    model = SurvivalMixup(**mixup_kw)
    date_str = strftime('%m_%d %H_%M_%S', gmtime())
    model.fit(x_train, y, 'TensorBoard/spiral/' + date_str)
    x_ec, T, D = model.predict(cls_centers)
    print(T)
    ax.scatter(*cls_centers.T, c='k', marker='*')
    ax.scatter(*x_ec.T, c='m', marker='^')
    
    exp_points = cls_centers
    
    x_e_all, T_all, D_all  = model.predict(x_train)
    
    fig = plt.figure()
    fig.suptitle('Reconstruction')
    ax3d = fig.add_subplot(111, projection='3d')
    for i in range(len(tau_bounds)):
        cur_points = x_train[i * n_per_cls : (i + 1) * n_per_cls]
        cur_t = t_train[i * n_per_cls : (i + 1) * n_per_cls]
        ax3d.scatter(*cur_points.T, cur_t, c=clr[i])
    ax3d.scatter(*x_e_all.T, T_all, c='k')
    
    
    x_explain, T_life = model.predict_trajectory(exp_points, 100, t_train.min(), t_train.max())
    
    fig = plt.figure()
    fig.suptitle('Trajectories')
    ax3d = fig.add_subplot(111, projection='3d')
    for i in range(len(tau_bounds)):
        cur_points = x_train[i * n_per_cls : (i + 1) * n_per_cls]
        cur_t = t_train[i * n_per_cls : (i + 1) * n_per_cls]
        ax3d.scatter(*cur_points.T, cur_t, c=clr[i])
    
    clr = ['m', 'k'] * (exp_points.shape[0] // 2)
    if exp_points.shape[0] % 2 != 0:
        clr.append('m')
    for i in range(exp_points.shape[0]):
        ax3d.scatter(*x_explain[i].T, T_life[i], c=clr[i])
    
    fig = plt.figure()
    fig.suptitle('Sampling')
    ax3d = fig.add_subplot(111, projection='3d')

    x_smp, T_smp, D_smp = model.sample_data(500)
    
    for i in range(len(tau_bounds)):
        cur_points = x_train[i * n_per_cls : (i + 1) * n_per_cls]
        cur_t = t_train[i * n_per_cls : (i + 1) * n_per_cls]
        ax3d.scatter(*cur_points.T, cur_t, c=clr[i])
    
    ax3d.scatter(*x_smp.T, T_smp, c='k')
    
    plt.show()
    
def x_experiment_moons():
    moon = lambda y, s, mu_x, mu_y: s * (y - mu_y) ** 2 + mu_x
    n_per_cls = 100
    noise_lvl = 0.01
    y_bounds = [
        (0, 1),
        (0.2, 0.8)
    ]
    
    cls_params = [
        (-6, 0, 0.5),
        (15, 3, 0.5)
    ]
    
    responses = [
        lambda X: (X[:, 0] + 3) * 6  + np.random.normal(0, 0.5, X.shape[0]),
        lambda X: (-X[:, 0] + 5) * 5 + np.random.normal(0, 0.5, X.shape[0]),
    ]
    
    x_train = []
    t_train = []
    clr = ['r', 'b']
    # fig, ax = plt.subplots(1, 1)
    # fig = plt.figure()
    # ax3d = fig.add_subplot(projection='3d')
    for i in range(len(cls_params)):
        y = np.random.uniform(*y_bounds[i], (n_per_cls))
        x = moon(y, *cls_params[i])
        X = np.stack((x, y), axis=-1)
        X += np.random.normal(0, noise_lvl, X.shape)
        x_train.append(X)
        t = responses[i](X)
        t_train.append(t)
        # ax.scatter(*X.T, c=clr[i], s=0.8)
        # ax3d.scatter(*X.T, t, c=clr[i])
    x_train = np.concatenate(x_train, 0)
    mean = np.mean(x_train, 0, keepdims=True)
    std = np.std(x_train, 0, keepdims=True)
    x_train = (x_train - mean) / std
    d_train = np.ones(x_train.shape[0])
    t_train = np.concatenate(t_train, 0)
    y = get_str_array(t_train, d_train)
    
    cls_centers = (np.asarray(
        [
            [0, 0.5],
            [3, 0.5]
        ]
    ) - mean) / std
    
    model = SurvivalMixup(**mixup_kw)
    date_str = strftime('%m_%d %H_%M_%S', gmtime())
    model.fit(x_train, y, 'TensorBoard/moons/' + date_str)
    t = np.linspace(np.min(t_train), np.max(t_train), 100)
    exp_points = cls_centers
    
    
    x_e_all, T_all, D_all = model.predict(x_train)
    
    fig = plt.figure()
    fig.suptitle('Reconstruction')
    ax3d = fig.add_subplot(111, projection='3d')
    for i in range(len(cls_params)):
        cur_points = x_train[i * n_per_cls : (i + 1) * n_per_cls]
        cur_t = t_train[i * n_per_cls : (i + 1) * n_per_cls]
        ax3d.scatter(*cur_points.T, cur_t, c=clr[i])
    ax3d.scatter(*x_e_all.T, T_all, c='k')
    
    
    x_explain, T_life = model.predict_trajectory(exp_points, 100, t_train.min(), t_train.max())
    
    fig = plt.figure()
    fig.suptitle('Trajectories in centers')
    ax3d = fig.add_subplot(111, projection='3d')
    for i in range(len(cls_params)):
        cur_points = x_train[i * n_per_cls : (i + 1) * n_per_cls]
        cur_t = t_train[i * n_per_cls : (i + 1) * n_per_cls]
        ax3d.scatter(*cur_points.T, cur_t, c=clr[i])
    
    e_clr = ['m', 'k'] * (exp_points.shape[0] // 2)
    if exp_points.shape[0] % 2 != 0:
        e_clr.append('m')
    for i in range(exp_points.shape[0]):
        ax3d.scatter(*x_explain[i].T, T_life[i], c=e_clr[i])
        
    exp_points = (np.asarray([
        [moon(0.8, *cls_params[0]), 0.8],
        [moon(0.3, *cls_params[1]), 0.3]
    ]) - mean) / std
    x_explain, T_life = model.predict_trajectory(exp_points, 100, t_train.min(), t_train.max())
    
    fig = plt.figure()
    fig.suptitle('Trajectories in good points')
    ax3d = fig.add_subplot(111, projection='3d')
    for i in range(len(cls_params)):
        cur_points = x_train[i * n_per_cls : (i + 1) * n_per_cls]
        cur_t = t_train[i * n_per_cls : (i + 1) * n_per_cls]
        ax3d.scatter(*cur_points.T, cur_t, c=clr[i])
    
    e_clr = ['m', 'k'] * (exp_points.shape[0] // 2)
    if exp_points.shape[0] % 2 != 0:
        e_clr.append('m')
    for i in range(exp_points.shape[0]):
        ax3d.scatter(*x_explain[i].T, T_life[i], c=e_clr[i])
        
    fig = plt.figure()
    fig.suptitle('Sampling')
    ax3d = fig.add_subplot(111, projection='3d')
    for i in range(len(cls_params)):
        cur_points = x_train[i * n_per_cls : (i + 1) * n_per_cls]
        cur_t = t_train[i * n_per_cls : (i + 1) * n_per_cls]
        ax3d.scatter(*cur_points.T, cur_t, c=clr[i])
        
    x_smp, T_smp, D_smp = model.sample_data(500)
    
    ax3d.scatter(*x_smp.T, T_smp, c='k')
        
    plt.show()
    
    
def x_experiment_curves():
    curve = lambda y, s: np.sin(s * y)
    n_per_cls = 200
    noise_lvl = 0.01
    x_bounds = [
        (-2, 3),
        (0, 5)
    ]
    
    cls_params = [
        (2,),
        (3,)
    ]
    
    y_mu = [-1, 1]
    
    responses = [
        lambda X: (X[:, 0] + 3) * 2  + np.random.normal(0, 0.4, X.shape[0]),
        lambda X: (-X[:, 0] + 6) * 3 + np.random.normal(0, 0.4, X.shape[0]),
    ]
    
    y_func = lambda x, i: curve(x, *cls_params[i]) + y_mu[i]
    
    x_train = []
    t_train = []
    clr = ['r', 'b']
    # fig, ax = plt.subplots(1, 1)
    # fig = plt.figure()
    # ax3d = fig.add_subplot(projection='3d')
    for i in range(len(cls_params)):
        x = np.random.uniform(*x_bounds[i], (n_per_cls))
        y = y_func(x, i)
        X = np.stack((x, y), axis=-1)
        X += np.random.normal(0, noise_lvl, X.shape)
        x_train.append(X)
        t = responses[i](X)
        t_train.append(t)
        # ax.scatter(*X.T, c=clr[i], s=0.8)
        # ax3d.scatter(*X.T, t, c=clr[i])
    x_train = np.concatenate(x_train, 0)
    mean = np.mean(x_train, 0, keepdims=True)
    std = np.std(x_train, 0, keepdims=True)
    x_train = (x_train - mean) / std
    d_train = np.ones(x_train.shape[0])
    t_train = np.concatenate(t_train, 0)
    y = get_str_array(t_train, d_train)
    
    cls_centers = np.asarray(
        [
            [np.mean(x_bounds[0]), y_func(np.mean(x_bounds[0]), 0)],
            [np.mean(x_bounds[1]), y_func(np.mean(x_bounds[1]), 1)]
        ]
    )
    t_exp = [responses[0](cls_centers[None, 0]), responses[1](cls_centers[None, 1])]
    cls_centers = (cls_centers - mean) / std
    
    model = SurvivalMixup(**mixup_kw)
    date_str = strftime('%m_%d %H_%M_%S', gmtime())
    model.fit(x_train, y, 'TensorBoard/curves/' + date_str)
    t = np.linspace(np.min(t_train), np.max(t_train), 100)
    exp_points = cls_centers
    
    
    x_e_all, T_all, D_all = model.predict(x_train)
    
    fig = plt.figure()
    fig.suptitle('Reconstruction')
    ax3d = fig.add_subplot(111, projection='3d')
    for i in range(len(cls_params)):
        cur_points = x_train[i * n_per_cls : (i + 1) * n_per_cls]
        cur_t = t_train[i * n_per_cls : (i + 1) * n_per_cls]
        ax3d.scatter(*cur_points.T, cur_t, c=clr[i])
    ax3d.scatter(*x_e_all.T, T_all, c='k')
    
    
    x_explain, T_life = model.predict_trajectory(exp_points, 100, t_train.min(), t_train.max())
    
    fig = plt.figure()
    fig.suptitle('Trajectories')
    ax3d = fig.add_subplot(111, projection='3d')
    for i in range(len(cls_params)):
        cur_points = x_train[i * n_per_cls : (i + 1) * n_per_cls]
        cur_t = t_train[i * n_per_cls : (i + 1) * n_per_cls]
        ax3d.scatter(*cur_points.T, cur_t, c=clr[i])
    
    e_clr = ['m', 'k'] * (exp_points.shape[0] // 2)
    if exp_points.shape[0] % 2 != 0:
        e_clr.append('m')
    for i in range(exp_points.shape[0]):
        ax3d.scatter(*x_explain[i].T, T_life[i], c=e_clr[i])
        
    ax3d.scatter(*cls_centers.T, t_exp, c='green', marker='*', s=200.0)
    
    fig = plt.figure()
    fig.suptitle('Sampling')
    ax3d = fig.add_subplot(111, projection='3d')
    for i in range(len(cls_params)):
        cur_points = x_train[i * n_per_cls : (i + 1) * n_per_cls]
        cur_t = t_train[i * n_per_cls : (i + 1) * n_per_cls]
        ax3d.scatter(*cur_points.T, cur_t, c=clr[i])
    
    x_smp, T_smp, D_smp = model.sample_data(500)
    ax3d.scatter(*x_smp.T, T_smp, c='k')
    
    plt.show()
    
    
def x_experiment_overlap():
    spiral = lambda tau: np.stack((np.sin(tau), np.cos(tau)), axis=-1)
    n_per_cls = 100
    noise_lvl = 0.02
    pi = np.pi
    tau_bounds = [
        (pi / 3, 3 * pi / 2),
        (pi / 2, 4 * pi / 2)
    ]
    
    responses = [
        lambda tau: 5 + np.random.normal(0, 0.1, tau.shape),
        lambda tau: 10 + np.random.normal(0, 0.1, tau.shape),
    ]
    
    x_train = []
    t_train = []
    clr = ['r', 'b']
    fig, ax = plt.subplots(1, 1)
    for i in range(len(tau_bounds)):
        tau = np.random.uniform(*tau_bounds[i], (n_per_cls))
        X = spiral(tau)
        X += np.random.normal(0, noise_lvl, X.shape)
        x_train.append(X)
        t = responses[i](tau)
        t_train.append(t)
        ax.scatter(*X.T, c=clr[i], s=0.8)
    x_train = np.concatenate(x_train, 0)
    d_train = np.ones(x_train.shape[0])
    t_train = np.concatenate(t_train, 0)
    y = get_str_array(t_train, d_train)
    
    cls_centers = spiral(np.mean(tau_bounds, axis=-1))
    
    
    model = SurvivalMixup(**mixup_kw)
    date_str = strftime('%m_%d %H_%M_%S', gmtime())
    model.fit(x_train, y, 'TensorBoard/overlap/' + date_str)
    x_ec, T, D = model.predict(cls_centers)
    print(T)
    ax.scatter(*cls_centers.T, c='k', marker='*')
    ax.scatter(*x_ec.T, c='m', marker='^')
    
    exp_points = cls_centers
    
    x_e_all, T_all, D_all = model.predict(x_train)
    
    fig = plt.figure()
    fig.suptitle('Reconstruction')
    ax3d = fig.add_subplot(111, projection='3d')
    for i in range(len(tau_bounds)):
        cur_points = x_train[i * n_per_cls : (i + 1) * n_per_cls]
        cur_t = t_train[i * n_per_cls : (i + 1) * n_per_cls]
        ax3d.scatter(*cur_points.T, cur_t, c=clr[i])
    ax3d.scatter(*x_e_all.T, T_all, c='k')
    
    
    x_explain, T_life = model.predict_trajectory(exp_points, 100, t_train.min(), t_train.max())
    
    fig = plt.figure()
    fig.suptitle('Trajectories')
    ax3d = fig.add_subplot(111, projection='3d')
    for i in range(len(tau_bounds)):
        cur_points = x_train[i * n_per_cls : (i + 1) * n_per_cls]
        cur_t = t_train[i * n_per_cls : (i + 1) * n_per_cls]
        ax3d.scatter(*cur_points.T, cur_t, c=clr[i])
    
    e_clr = ['m', 'k'] * (exp_points.shape[0] // 2)
    if exp_points.shape[0] % 2 != 0:
        e_clr.append('m')
    for i in range(exp_points.shape[0]):
        ax3d.scatter(*x_explain[i].T, T_life[i], c=e_clr[i])
        
    fig = plt.figure()
    fig.suptitle('Trajectories')
    ax3d = fig.add_subplot(111, projection='3d')
    for i in range(len(tau_bounds)):
        cur_points = x_train[i * n_per_cls : (i + 1) * n_per_cls]
        cur_t = t_train[i * n_per_cls : (i + 1) * n_per_cls]
        ax3d.scatter(*cur_points.T, cur_t, c=clr[i])
    
    x_smp, T_smp, D_smp = model.sample_data(500)
    ax3d.scatter(*x_smp.T, T_smp, c='k')
    
    plt.show()
    
def censored_exp():
    cens_num = 200
    uncens_num = 100
    inner_circle = np.random.uniform(-1, 1, (cens_num, 2))
    norm = np.linalg.norm(inner_circle, axis=-1)
    outer_points = norm > 1
    inner_circle[outer_points] = inner_circle[outer_points] / norm[outer_points, None]
    y_inner = 2 - np.sum(inner_circle ** 2, axis=-1) + np.random.normal(0, 0.01, cens_num)
    fig, ax = plt.subplots(1, 1)
    ax.scatter(*inner_circle.T, c='r', s=0.8)
    
    dist = 0.75
    width = 0.25
    angles = np.random.uniform(0, 2 * np.pi, uncens_num)
    noise = np.random.uniform(-width, width, uncens_num)
    outer_circle = (1 + dist) * np.stack(
        (np.cos(angles), np.sin(angles)), axis=-1
    )
    outer_circle += noise[:, None] * np.stack(
        (np.cos(angles), np.sin(angles)), axis=-1
    )
    y_outer = 1 + angles / (2 * np.pi) + np.random.normal(0, 0.01, uncens_num)
    ax.scatter(*outer_circle.T, c='b', s=0.8)
    
    idx_shf = np.arange(cens_num + uncens_num)
    rng = np.random.default_rng()
    rng.shuffle(idx_shf)
    
    X = np.concatenate((inner_circle, outer_circle), axis=0)[idx_shf]
    Y = np.concatenate((y_inner, y_outer))[idx_shf]
    D = np.concatenate((np.zeros(cens_num), np.ones(uncens_num)))[idx_shf]
    
    fig = plt.figure()
    fig.suptitle('Dataset')
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.scatter(*X.T, Y, c=['r' if d == 0 else 'b' for d in D])
    
    x_train = X
    y_train = get_str_array(Y, D)
    model = SurvivalMixup(cens_cls_model=RandomForestClassifier(n_estimators=40), **mixup_kw)
    date_str = strftime('%m_%d %H_%M_%S', gmtime())
    model.fit(x_train, y_train, 'TensorBoard/censor/' + date_str)
    
    x_recon, y_recon, d_recon = model.predict(x_train)
    fig = plt.figure()
    fig.suptitle('Reconstruction')
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.scatter(*x_recon.T, y_recon, c=['r' if d == 0 else 'b' for d in d_recon])
    fig, ax = plt.subplots(1, 1)
    ax.scatter(*x_recon.T, c=['r' if d == 0 else 'b' for d in d_recon], s=0.8)
    
    
    plt.show()
    
    
def real_ds_test(x, y, name='real ds'):
    # seed = 123
    def draw_tsne(x_list, name_list=None):
        X = np.concatenate(x_list, 0)
        z = TSNE().fit_transform(X)
        fig, ax = plt.subplots(1, 1)
        fig.suptitle(name)
        start_idx, end_idx = 0, 0
        for i in range(len(x_list)):
            end_idx += x_list[i].shape[0]
            cur_cls = z[start_idx:end_idx]
            label = name_list[i] if name_list is not None else ''
            ax.scatter(*cur_cls.T, s=0.87, label=label)
            start_idx += x_list[i].shape[0]
        return fig, ax
    
    def draw_kaplan(T, D, name='', axis=None):
        if axis is None:
            fig, ax = plt.subplots(1, 1)
            # fig.suptitle(name)
        else:
            fig = None
            ax = axis
        x, y, conf_int = kaplan_meier_estimator(D.astype(bool), T, conf_type="log-log")
        ax.step(x, y, where="post", label=name)
        ax.fill_between(x, conf_int[0], conf_int[1], alpha=0.25, step="post")
        ax.set_ylim(0, 1)
        ax.grid(True)
        return fig, ax
    
    km_fig, km_axis = draw_kaplan(y['time'], y['censor'], 'Original data')
    
    model = SurvivalMixup(cens_cls_model=RandomForestClassifier(), **mixup_kw)
    date_str = strftime('%m_%d %H_%M_%S', gmtime())
    model.fit(x, y, f'TensorBoard/{name}/{date_str}')
    model.samples_num = 64
    model.batch_load = 256
    x_rec, y_rec, d_rec = model.predict(x)
    x_samples, t_samples, d_samples = model.sample_data(x.shape[0])

    _, tsne_ax = draw_tsne((x, x_rec), ('Original data', 'Reconstruction'))
    tsne_ax.legend()
    _, tsne_ax = draw_tsne((x, x_samples), ('Original data', 'Sampling'))
    tsne_ax.legend()
    draw_kaplan(y_rec, d_rec, 'Reconstruction', km_axis)
    draw_kaplan(t_samples, d_samples, 'Sampling', km_axis)
    km_fig.suptitle(name)
    km_axis.legend()
    
    c_ind_model, *_ = concordance_index_censored(y['censor'], y['time'], -y_rec)
    
    rf = RandomSurvivalForest().fit(x, y)
    c_ind_rf = rf.score(x, y)
    y_sim = get_str_array(y_rec, d_rec)
    c_ind_simul = rf.score(x_rec, y_sim)
    rf = RandomSurvivalForest().fit(x_rec, y_sim)
    c_ind_full_sim = rf.score(x_rec, y_sim)
    x_enl = np.concatenate((x, x_rec), axis=0)
    t_enl = np.concatenate((y_rec, y['time']))
    d_enl = np.concatenate((d_rec, y['censor']))
    y_enl = get_str_array(t_enl, d_enl)
    rf = RandomSurvivalForest().fit(x_enl, y_enl)
    c_ind_enl = rf.score(x, y)
    
    print('Model c_ind:', c_ind_model)
    print('RF c_ind (orig ds):', c_ind_rf)
    print('RF c_ind (orig ds -> simulated data):', c_ind_simul)
    print('RF c_ind (simulated data -> simulated data)', c_ind_full_sim)
    print('RF c_ind (orig ds + simulated data -> orig ds)', c_ind_enl)
    plt.show()


def veterans_exp():
    vae_kw['latent_dim'] = 8
    mixup_kw['batch_num'] = 16
    mixup_kw['epochs'] = 100
    loader = sksurv_loader()
    real_ds_test(*loader.load_veterans_lung_cancer, 'Veterans')
    
def whas500_exp():
    vae_kw['latent_dim'] = 18
    mixup_kw['batch_num'] = 16
    mixup_kw['epochs'] = 100
    mixup_kw['benk_vae_loss_rat'] = 0.55
    loader = sksurv_loader()
    real_ds_test(*loader.load_whas500, 'WHAS500')
    
def gbsg2_exp():
    vae_kw['latent_dim'] = 18
    mixup_kw['batch_num'] = 20
    mixup_kw['epochs'] = 100
    mixup_kw['benk_vae_loss_rat'] = 0.66
    mixup_kw['gumbel_tau'] = 1
    mixup_kw['c_ind_temp'] = 2
    loader = sksurv_loader()
    real_ds_test(*loader.load_gbsg2, 'GBSG2')
    
def aids_exp():
    vae_kw['latent_dim'] = 16
    mixup_kw['batch_num'] = 10
    mixup_kw['epochs'] = 100
    mixup_kw['benk_vae_loss_rat'] = 0.7
    loader = sksurv_loader()
    real_ds_test(*loader.load_aids, 'AIDS')
    
if __name__=='__main__':
    vae_kw = {
        'latent_dim': 8,
        'regular_coef': 60,
        'sigma_z': 1
    }
    mixup_kw = {
        'vae_kw': vae_kw,
        'samples_num': 64,
        'batch_num': 16,
        'epochs': 150,
        'lr_rate': 2e-3,
        'benk_vae_loss_rat': 0.2,
        'c_ind_temp': 1,
        'gumbel_tau': 0.50,
        'train_bg_part': 0.6,
        'batch_load': None,
    }
    x_experiment_linear()
    # x_experiment_spiral()
    # x_experiment_moons()
    # x_experiment_curves()
    # x_experiment_overlap()
    # censored_exp()
    
    # veterans_exp()
    # whas500_exp()
    # gbsg2_exp()
    # aids_exp()