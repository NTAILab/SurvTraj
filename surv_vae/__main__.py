from .surv_mixup import SurvivalMixup
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import concordance_index_censored
from sksurv.ensemble import RandomSurvivalForest

import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime
from .utility import get_str_array, sksurv_loader, get_traject_plot


from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from scipy.stats import normaltest
from torch import tensor, get_default_dtype as tens_type, no_grad


def draw_latent_space(model, *args):
    dim = model.vae.latent_dim
    if dim < 2:
        return None
    fig, ax = plt.subplots(1, 1)
    X = np.concatenate(args, axis=0)
    with no_grad():
        _, _, _, Z = model(tensor(X, dtype=tens_type(), device=model.device))
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
    x_clusters = []
    t_clusters = []
    clr = ['r', 'b']
    for i in range(0, cls_centers.shape[0], 2):
        mix_coef = np.random.uniform(0, 1, (n_per_cls, 1))
        X = cls_points[i] * mix_coef + cls_points[i + 1] * (1 - mix_coef)
        X += np.random.normal(0, 0.005, X.shape)
        x_train.append(X)
        mix_coef = mix_coef.ravel()
        y = times[i] * mix_coef + times[i + 1] * (1 - mix_coef)
        y += np.random.normal(0, 0.1, y.shape)
        t_train.append(y)
        cluster_points = np.concatenate((X, cls_points[i], cls_points[i + 1]), axis=0)
        x_clusters.append(cluster_points)
        cluster_y = np.concatenate((y, np.tile(times[i], n_per_cls), np.tile(times[i + 1], n_per_cls)))
        t_clusters.append(cluster_y)
    x_train = np.concatenate(x_train, 0)
    x_train = np.concatenate((x_train, cls_points.reshape(-1, cls_centers.shape[1])), 0)
    d_train = np.ones(x_train.shape[0])
    t_train = np.concatenate(t_train, 0)
    t_train = np.concatenate((t_train, np.repeat(times, n_per_cls)), 0)
    y = get_str_array(t_train, d_train)
    
    model = SurvivalMixup(**mixup_kw)
    date_str = strftime('%m_%d %H_%M_%S', gmtime())
    model.fit(x_train, y, log_dir='TensorBoard/linear/' + date_str)
    
    def draw_train_set_2d(name):
        fig, ax = plt.subplots(1, 1, dpi=100, figsize=(6, 6))
        fig.suptitle(name)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        for i in range(2):
            ax.scatter(*x_clusters[i].T, c=clr[i], s=0.8, label='Cluster ' + str(i + 1))
        ax.xaxis.set_zorder(-100)
        ax.yaxis.set_zorder(-100)
        ax.grid(linestyle='--', alpha=0.5)
        return fig, ax
    
    def draw_train_set_3d(name, z_name):
        fig = plt.figure(dpi=100, figsize=(6, 6))
        fig.suptitle(name)
        ax3d = fig.add_subplot(111, projection='3d')
        ax3d.set_xlabel('$x_1$')
        ax3d.set_ylabel('$x_2$')
        ax3d.set_zlabel(z_name)
        for i in range(2):
            ax3d.scatter(*x_clusters[i].T, t_clusters[i], c=clr[i], label='Cluster ' + str(i + 1))
        return fig, ax3d
    
    exp_points_2d = np.concatenate(
        (cls_centers, (cls_centers[None, -1] + cls_centers[None, -2]) / 2, (cls_centers[None, -3] + cls_centers[None, -4]) / 2),
        axis=0
    )
    x_ec, T, D = model.predict_recon(exp_points_2d)
    E_T = model.predict_exp_time(exp_points_2d)
    print(E_T)
    fig, ax = draw_train_set_2d('Reconstruction')
    ax.scatter(*exp_points_2d.T, c='k', marker='*', s=50, label='Test points')
    ax.scatter(*x_ec.T, c='m', marker='^', s=50, label='Sampled points')
    ax.legend()
    
    exp_points = np.concatenate([
            # (cls_centers[None, -1] + cls_centers[None, -2]) / 2,
            # (cls_centers[None, -3] + cls_centers[None, -4]) / 2,
            # np.mean(cls_centers, axis=0, keepdims=True),
            cls_centers[None, 0],
            cls_centers[None, 3]
        ], axis=0,
    )
    
    x_e_all, T_gen, D_all  = model.predict_recon(x_train)
    E_T = model.predict_exp_time(x_train)
    
    fig, ax3d = draw_train_set_3d('Reconstruction', '$T_{gen}$')
    ax3d.scatter(*x_e_all.T, T_gen, c='k', label='Reconstructions')
    ax3d.legend()
    
    fig, ax3d = draw_train_set_3d('Reconstruction', '$\\hat{T}$')
    ax3d.scatter(*x_e_all.T, E_T, c='k', label='Reconstructions')
    ax3d.legend()
    
    fig, ax = draw_train_set_2d('Reconstruction')
    ax.scatter(*x_e_all.T, c='k', s=1, label='Reconstructions')
    ax.legend()

    t_traj = np.linspace(t_train.min(), t_train.max(), 100)
    t_traj = np.tile(t_traj[None, :], (len(exp_points), 1))
    x_explain = model.predict_trajectory(exp_points, t_traj, True)
    
    fig, ax3d = draw_train_set_3d('Trajectories', 't')
    
    lab_list = ['A', 'B']
    clr_t = ['teal', 'tomato']
    for i in range(exp_points.shape[0]):
        ax3d.scatter(*x_explain[i].T, t_traj[i], c=clr_t[i], label='Trajectory ' + lab_list[i])
        
    ax3d.legend()
    
    markers = ['^', 'X']
    fig, ax = draw_train_set_2d('Trajectories')
    for i in range(2):
        ax.scatter(*x_explain[i].T, s=10, c=clr_t[i], label='Trajectory ' + lab_list[i])
        ax.scatter(*exp_points[i].T, s=100, c='k', edgecolors='w', zorder=100,
                   marker=markers[i], label='Point ' + lab_list[i])
    ax.legend()
    
    # fig, ax3d = draw_train_set_3d('Sampling', '$T_{gen}$')
    
    # x_smp, T_smp, D_smp = model.sample_data(300)
    
    # ax3d.scatter(*x_smp.T, T_smp, c='k', label='Sampled points')
    # ax3d.legend()
    # fig, ax = draw_train_set_2d('Sampling')
    # ax.scatter(*x_smp.T, s=10, c='k', label='Sampled points')
    # ax.legend()
    
    draw_latent_space(model, *x_clusters)
    
    get_traject_plot(x_explain[0], t_traj[0])
    
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
        lambda X: (-X[:, 0] + 6) * 4 + np.random.normal(0, 0.5, X.shape[0]),
    ]
    
    x_train_list = []
    t_train_list = []
    clr = ['r', 'b']
    for i in range(len(cls_params)):
        y = np.random.uniform(*y_bounds[i], (n_per_cls))
        x = moon(y, *cls_params[i])
        X = np.stack((x, y), axis=-1)
        X += np.random.normal(0, noise_lvl, X.shape)
        x_train_list.append(X)
        t = responses[i](X)
        t_train_list.append(t)
        
    x_train = np.concatenate(x_train_list, 0)
    mean = np.mean(x_train, 0, keepdims=True)
    std = np.std(x_train, 0, keepdims=True)
    x_train = (x_train - mean) / std
    d_train = np.ones(x_train.shape[0])
    t_train = np.concatenate(t_train_list, 0)
    y = get_str_array(t_train, d_train)
    
    x_clusters = []
    t_clusters = []
    for i in range(len(cls_params)):
        x_clusters.append(x_train[i * n_per_cls : (i + 1) * n_per_cls])
        t_clusters.append(t_train[i * n_per_cls : (i + 1) * n_per_cls])
    
    # cls_centers = (np.asarray(
    #     [
    #         [0, 0.5],
    #         [3, 0.5]
    #     ]
    # ) - mean) / std
    
    model = SurvivalMixup(**mixup_kw)
    date_str = strftime('%m_%d %H_%M_%S', gmtime())
    model.fit(x_train, y, log_dir='TensorBoard/moons/' + date_str)
    t = np.linspace(np.min(t_train), np.max(t_train), 100)
    # exp_points = cls_centers
    
    def draw_train_set_2d(name):
        fig, ax = plt.subplots(1, 1, dpi=100, figsize=(6, 6))
        fig.suptitle(name)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        for i in range(2):
            ax.scatter(*x_clusters[i].T, c=clr[i], s=0.8, label='Cluster ' + str(i + 1))
        ax.xaxis.set_zorder(-100)
        ax.yaxis.set_zorder(-100)
        ax.grid(linestyle='--', alpha=0.5)
        return fig, ax
    
    def draw_train_set_3d(name, z_name):
        fig = plt.figure(dpi=100, figsize=(6, 6))
        fig.suptitle(name)
        ax3d = fig.add_subplot(111, projection='3d')
        ax3d.set_xlabel('$x_1$')
        ax3d.set_ylabel('$x_2$')
        ax3d.set_zlabel(z_name)
        for i in range(2):
            ax3d.scatter(*x_clusters[i].T, t_clusters[i], c=clr[i], label='Cluster ' + str(i + 1))
        return fig, ax3d
    
    x_e_all, T_gen, D_all  = model.predict_recon(x_train)
    E_T = model.predict_exp_time(x_train)
    
    fig, ax3d = draw_train_set_3d('Reconstruction', '$T_{gen}$')
    ax3d.scatter(*x_e_all.T, T_gen, c='k', label='Reconstructions')
    ax3d.legend()
    
    fig, ax3d = draw_train_set_3d('Reconstruction', '$\\hat{T}$')
    ax3d.scatter(*x_e_all.T, E_T, c='k', label='Reconstructions')
    ax3d.legend()
    
    exp_points = (np.asarray([
        [moon(0.8, *cls_params[0]), 0.8],
        [moon(0.3, *cls_params[1]), 0.3]
    ]) - mean) / std

    t_traj = np.linspace(t_train.min(), t_train.max(), 100)
    t_traj = np.tile(t_traj[None, :], (len(exp_points), 1))
    x_explain = model.predict_trajectory(exp_points, t_traj, True)
    
    fig, ax3d = draw_train_set_3d('Trajectories', 't')
    
    lab_list = ['A', 'B']
    clr_t = ['teal', 'tomato']
    for i in range(exp_points.shape[0]):
        ax3d.scatter(*x_explain[i].T, t_traj[i], c=clr_t[i], label='Trajectory ' + lab_list[i])
        
    ax3d.legend()
        
    markers = ['^', 'X']
    fig, ax = draw_train_set_2d('Trajectories')
    for i in range(2):
        ax.scatter(*x_explain[i].T, s=10, c=clr_t[i], label='Trajectory ' + lab_list[i])
        ax.scatter(*exp_points[i].T, s=100, c='k', edgecolors='w', zorder=100, marker=markers[i], label='Point ' + lab_list[i])
    ax.legend()
    
    # fig, ax3d = draw_train_set_3d('Sampling', '$T_{gen}$')
    
    # x_smp, T_smp, D_smp = model.sample_data(300)
    
    # ax3d.scatter(*x_smp.T, T_smp, c='k', label='Sampled points')
    # ax3d.legend()
    # fig, ax = draw_train_set_2d('Sampling')
    # ax.scatter(*x_smp.T, s=10, c='k', label='Sampled points')
    # ax.legend()
    
    draw_latent_space(model, *x_clusters)
        
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
    
    x_train_list = []
    t_train_list = []
    clr = ['r', 'b']
    x_clusters = []
    t_clusters = []
    for i in range(len(tau_bounds)):
        tau = np.random.uniform(*tau_bounds[i], (n_per_cls))
        X = spiral(tau)
        X += np.random.normal(0, noise_lvl, X.shape)
        x_train_list.append(X)
        t = responses[i](tau)
        t_train_list.append(t)
        x_clusters.append(X)
        t_clusters.append(t)
    x_train = np.concatenate(x_train_list, 0)
    d_train = np.ones(x_train.shape[0])
    t_train = np.concatenate(t_train_list, 0)
    y = get_str_array(t_train, d_train)
    
    cls_centers = spiral(np.mean(tau_bounds, axis=-1))
    
    def draw_train_set_3d(name, z_name):
        fig = plt.figure(dpi=100, figsize=(6, 6))
        fig.suptitle(name)
        ax3d = fig.add_subplot(111, projection='3d')
        ax3d.set_xlabel('$x_1$')
        ax3d.set_ylabel('$x_2$')
        ax3d.set_zlabel(z_name)
        for i in range(2):
            ax3d.scatter(*x_clusters[i].T, t_clusters[i], c=clr[i], label='Cluster ' + str(i + 1))
        return fig, ax3d
    
    fig, ax = draw_train_set_3d('Data', 't')
    ax.legend()
    
    
    model = SurvivalMixup(**mixup_kw)
    date_str = strftime('%m_%d %H_%M_%S', gmtime())
    model.fit(x_train, y, log_dir='TensorBoard/overlap/' + date_str)
    
    x_e_all, T_gen, D_all = model.predict_recon(x_train)
    E_T = model.predict_exp_time(x_train)
    
    fig, ax3d = draw_train_set_3d('Reconstruction', '$T_{gen}$')
    ax3d.scatter(*x_e_all.T, T_gen, c='k', label='Reconstruction')
    ax3d.legend()
    fig, ax3d = draw_train_set_3d('Reconstruction', '$\\hat{T}$')
    ax3d.scatter(*x_e_all.T, E_T, c='k', label='Reconstruction')
    ax3d.legend()
    
    # x_smp, T_smp, D_smp = model.sample_data(300)
    # fig, ax3d = draw_train_set_3d('Sampling', '$T_{gen}$')
    # ax3d.scatter(*x_smp.T, T_smp, c='k', label='Sampled points')
    # ax3d.legend()
    
    draw_latent_space(model, *x_clusters)
    
    plt.show()
    
def censored_exp():
    cens_num = 200
    uncens_num = 100
    inner_circle = np.random.uniform(-1, 1, (cens_num, 2))
    norm = np.linalg.norm(inner_circle, axis=-1)
    outer_points = norm > 1
    inner_circle[outer_points] = inner_circle[outer_points] / norm[outer_points, None]
    y_inner = 2 - np.sum(inner_circle ** 2, axis=-1) + np.random.normal(0, 0.01, cens_num)
    
    dist = 1.5
    width = 0.25
    angles = np.random.uniform(0, 1.75 * np.pi, uncens_num)
    noise = np.random.uniform(-width, width, uncens_num)
    outer_circle = (1 + dist) * np.stack(
        (np.cos(angles), np.sin(angles)), axis=-1
    )
    outer_circle += noise[:, None] * np.stack(
        (np.cos(angles), np.sin(angles)), axis=-1
    )
    y_outer = 1 + angles / (2 * np.pi) + np.random.normal(0, 0.01, uncens_num)
    
    idx_shf = np.arange(cens_num + uncens_num)
    rng = np.random.default_rng()
    rng.shuffle(idx_shf)
    
    X = np.concatenate((inner_circle, outer_circle), axis=0)[idx_shf]
    Y = np.concatenate((y_inner, y_outer))[idx_shf]
    D = np.concatenate((np.zeros(cens_num), np.ones(uncens_num)))[idx_shf]
    
    def draw_set_2d(name, censored_x, uncensored_x):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        fig.suptitle(name)
        ax.scatter(*censored_x.T, c='r', s=0.8, label='Censored points')
        ax.scatter(*uncensored_x.T, c='b', s=0.8, label='Uncensored points')
        ax.xaxis.set_zorder(-100)
        ax.yaxis.set_zorder(-100)
        ax.grid(linestyle='--', alpha=0.5)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.legend()
        return fig, ax
    
    
    def draw_set_3d(name, z_name, censored_x, censored_t, uncensored_x, uncensored_t):
        fig = plt.figure(figsize=(6, 6))
        fig.suptitle(name)
        ax3d = fig.add_subplot(111, projection='3d')
        ax3d.scatter(*censored_x.T, censored_t, c='r', label='Censored points')
        ax3d.scatter(*uncensored_x.T, uncensored_t, c='b', label='Uncensored points')
        ax3d.set_xlabel('$x_1$')
        ax3d.set_ylabel('$x_2$')
        ax3d.set_zlabel(z_name)
        ax3d.legend()
        return fig, ax3d
    
    draw_set_2d('Dataset', inner_circle, outer_circle)
    draw_set_3d('Dataset', 't', inner_circle, y_inner, outer_circle, y_outer)
    
    x_train = X
    y_train = get_str_array(Y, D)
    model = SurvivalMixup(cens_clf_model=RandomForestClassifier(n_estimators=100), **mixup_kw)
    date_str = strftime('%m_%d %H_%M_%S', gmtime())
    model.fit(x_train, y_train, log_dir='TensorBoard/censor/' + date_str)
    
    x_recon, t_gen, d_recon = model.predict_recon(x_train)
    e_t = model.predict_exp_time(x_train)
    draw_set_2d('Recontruction', x_recon[d_recon == 0], x_recon[d_recon == 1])
    draw_set_3d('Recontruction', '$T_{gen}$', x_recon[d_recon == 0], t_gen[d_recon == 0], x_recon[d_recon == 1], t_gen[d_recon == 1])
    draw_set_3d('Recontruction', '$\\hat{T}$', x_recon[d_recon == 0], e_t[d_recon == 0], x_recon[d_recon == 1], e_t[d_recon == 1])
    
    # X, T, D = model.sample_data(400)
    # draw_set_2d('Sampling', X[D == 0], X[D == 1])
    # draw_set_3d('Sampling', '$T_{gen}$', X[D == 0], T[D == 0], X[D == 1], T[D == 1])
    
    plt.show()
    
    
def real_ds_test(x, y, name='real ds', cens_clf=None):
    def draw_tsne(x_list, name_list=None, clr_list=None):
        X = np.concatenate(x_list, 0)
        z = TSNE().fit_transform(X)
        fig, ax = plt.subplots(1, 1)
        fig.suptitle(name)
        start_idx, end_idx = 0, 0
        for i in range(len(x_list)):
            end_idx += x_list[i].shape[0]
            cur_cls = z[start_idx:end_idx]
            label = name_list[i] if name_list is not None else ''
            clr = None if clr_list is None else clr_list[i]
            ax.scatter(*cur_cls.T, s=3, label=label, c=clr)
            start_idx += x_list[i].shape[0]
        ax.xaxis.set_zorder(-100)
        ax.yaxis.set_zorder(-100)
        ax.grid(linestyle='--', alpha=0.5)
        return fig, ax
    
    def draw_kaplan(T, D, name='', axis=None, clr=None):
        if axis is None:
            fig, ax = plt.subplots(1, 1)
            # fig.suptitle(name)
        else:
            fig = None
            ax = axis
        x, y, conf_int = kaplan_meier_estimator(D.astype(bool), T, conf_type="log-log")
        ax.step(x, y, where="post", label=name, c=clr)
        ax.fill_between(x, conf_int[0], conf_int[1], alpha=0.25, step="post", color=clr)
        ax.set_ylim(0, 1)
        ax.xaxis.set_zorder(-100)
        ax.yaxis.set_zorder(-100)
        ax.grid(linestyle='--', alpha=0.5)
        return fig, ax
    
    km_fig, km_axis = draw_kaplan(y['time'], y['censor'], 'Original data', clr='orange')
    
    
    model = SurvivalMixup(cens_clf_model=RandomForestClassifier() if cens_clf is None else cens_clf, **mixup_kw)
    date_str = strftime('%m_%d %H_%M_%S', gmtime())
    model.fit(x, y, log_dir=f'TensorBoard/{name}/{date_str}')
    # model.samples_num = 64
    model.batch_load = 256
    x_rec, t_gen, d_rec = model.predict_recon(x)
    # x_samples, t_samples, d_samples = model.sample_data(x.shape[0])

    tsne_clr = ['b', 'r']
    _, tsne_ax = draw_tsne((x, x_rec), ('Original data', 'Reconstruction'), tsne_clr)
    tsne_ax.legend()
    # _, tsne_ax = draw_tsne((x, x_samples), ('Original data', 'Sampling'), tsne_clr)
    # tsne_ax.legend()
    draw_kaplan(t_gen, d_rec, 'Reconstruction', km_axis, clr='teal')
    # draw_kaplan(t_samples, d_samples, 'Sampling', km_axis, clr='tomato')
    km_fig.suptitle(name)
    km_axis.legend()
    km_fig, km_axis = draw_kaplan(y['time'], y['censor'], 'Original data', clr='orange')
    draw_kaplan(t_gen, d_rec, 'Reconstruction', km_axis, clr='teal')
    km_fig.suptitle(name)
    km_axis.legend()
    km_fig, km_axis = draw_kaplan(y['time'], y['censor'], 'Original data', clr='orange')
    # draw_kaplan(t_samples, d_samples, 'Sampling', km_axis, clr='tomato')
    km_fig.suptitle(name)
    km_axis.legend()
    
    c_ind_model = model.score(x, y)
    
    rf = RandomSurvivalForest().fit(x, y)
    c_ind_rf = rf.score(x, y)
    y_sim = get_str_array(t_gen, d_rec)
    c_ind_simul = rf.score(x_rec, y_sim)
    rf = RandomSurvivalForest().fit(x_rec, y_sim)
    c_ind_full_sim = rf.score(x_rec, y_sim)
    x_enl = np.concatenate((x, x_rec), axis=0)
    t_enl = np.concatenate((t_gen, y['time']))
    d_enl = np.concatenate((d_rec, y['censor']))
    y_enl = get_str_array(t_enl, d_enl)
    rf = RandomSurvivalForest().fit(x_enl, y_enl)
    c_ind_enl = rf.score(x, y)
    
    print('Cens clf roc-auc:', roc_auc_score(y['censor'], d_rec))
    print('Model c_ind:', c_ind_model)
    print('RF c_ind (orig ds):', c_ind_rf)
    print('RF c_ind (orig ds -> simulated data):', c_ind_simul)
    print('RF c_ind (simulated data -> simulated data)', c_ind_full_sim)
    print('RF c_ind (orig ds + simulated data -> orig ds)', c_ind_enl)
    plt.show()


def veterans_exp():
    vae_kw['latent_dim'] = 15
    mixup_kw['batch_num'] = 16
    vae_kw['regular_coef'] = 40
    mixup_kw['epochs'] = 200
    loader = sksurv_loader()
    real_ds_test(*loader.load_veterans_lung_cancer, 'Veterans')
    
def whas500_exp():
    vae_kw['latent_dim'] = 16
    mixup_kw['batch_num'] = 16
    vae_kw['regular_coef'] = 25
    mixup_kw['epochs'] = 250
    mixup_kw['benk_vae_loss_rat'] = 0.9
    loader = sksurv_loader()
    real_ds_test(*loader.load_whas500, 'WHAS500', CatBoostClassifier(iterations=1000, depth=6,
                                                                     loss_function='CrossEntropy',verbose=0))
    
def gbsg2_exp():
    vae_kw['latent_dim'] = 12
    mixup_kw['batch_num'] = 20
    mixup_kw['epochs'] = 300
    vae_kw['regular_coef'] = 40
    mixup_kw['benk_vae_loss_rat'] = 0.9
    mixup_kw['gumbel_tau'] = 1
    mixup_kw['c_ind_temp'] = 1
    loader = sksurv_loader()
    real_ds_test(*loader.load_gbsg2, 'GBSG2', CatBoostClassifier(iterations=1000, depth=6,
                                                                     loss_function='CrossEntropy',verbose=0))
    
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
        'samples_num': 48,
        'batch_num': 16,
        'epochs': 150,
        'lr_rate': 1e-3,
        'c_ind_weight': 0.5,
        'vae_weight': 1.0,
        'traj_weight': 0.5,
        'likelihood_weight': 0.00005,
        'c_ind_temp': 1,
        'gumbel_tau': 1.0,
        'train_bg_part': 0.6,
        'batch_load': None,
        'device': 'cuda:0',
    }
    # x_experiment_linear()
    # x_experiment_spiral()
    # x_experiment_moons()
    # x_experiment_curves()
    # x_experiment_overlap()
    # censored_exp()
    
    veterans_exp()
    # whas500_exp()
    # gbsg2_exp()
    # aids_exp()