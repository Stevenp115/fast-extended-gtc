"""
GTCRegime, for the purposes of Research Project 3.

This contains the utilities for full regime estimation, with dependency on MixtureFactor.
"""

import cupy as cp
from sklearn.cluster import KMeans
from scipy.stats import norm, invgamma, nct, multivariate_normal
from scipy.optimize import fsolve
from scipy.special import digamma
import copy
from pprint import pprint
from mixturemodel import MixtureFactor
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from numpy.lib.stride_tricks import sliding_window_view
from types import SimpleNamespace
import warnings
import torch
from torch.distributions.inverse_gamma import InverseGamma


warnings.filterwarnings('ignore')

cp.set_printoptions(
    precision=3,
    suppress=True,
    floatmode="fixed"
)


def mse(x, y):
    return ((x-y)**2).mean()


def rmse(x, y):
    return cp.sqrt(mse(x, y))


def mse_ratio(true, est):
    return mse(true, est) / true.var()


def get_defaults():
    defaults = {
        'var_sample_size': 100_000,
        'sim_size': 100_000,
        'tol': 0.10,
        'sim_tol': 0.10,
        'nreq': 10,
        'subsample_size': 25,
        'subslice_size': 2,
        'alpha_cap': 0.999,
        'n_mixture_iters': 100,
        'n_mixture_seeds': 1
    }
    return defaults


class GTCEstimator:
    def __init__(self, **kwargs):
        """
        r (int): Number of regimes.
        Note, each paramter mu, sd, df, alpha should be of shape r x 1.
        """
        args = SimpleNamespace(**kwargs)

        self.r = args.r
        self.g = args.g

        if (
            hasattr(args, 'mu') and
            hasattr(args, 'sd') and
            hasattr(args, 'df') and
            hasattr(args, 'pi') and
            hasattr(args, 'alpha')
        ):
            self.manual = False
            self.n = args.n
            self.p = args.p
            self.pi_true = args.pi
            self.mu_true = args.mu
            self.sd_true = args.sd
            self.df_true = args.df
            self.alpha_true = args.alpha

            if (self.r > 1) and (self.g > 1):
                raise NotImplementedError(
                    "This package supports only (r=1, g>=1) or (r>1, g=1).")

            if args.pi is None:
                self.pi_true = cp.random.dirichlet(
                    alpha=[1 for i in range(self.r)], size=self.g)
                print(self.pi_true)

            self.labels_true = cp.random.dirichlet(
                alpha=self.pi_true, size=self.n).argmax(axis=1)

            picker = cp.zeros((self.n, self.r))

            # This is n x r, with picker[i, j] = 1 if obs i is in regime j.
            picker[(cp.arange(0, self.n, 1), self.labels_true)] = 1
            self.picker_true = picker

            # Generate components
            self.Z_true, self.U_true = self.generate_Z()
            self.W_true = self.generate_W()
            self.X = self.Z_true * self.W_true[:, None]

            self.W_hat_lst = []

        else:
            self.manual = True
            self.X = args.X
            self.n, self.p = self.X.shape

        # Auxiliary hyperparameters / helper variables.
        self.var_sample_size = args.var_sample_size
        self.sim_size = args.sim_size
        self.tol = args.tol
        self.sim_tol = args.sim_tol
        self.nreq = args.nreq
        self.subsample_size = args.subsample_size
        self.subslice_size = args.subslice_size
        self.ones = cp.ones(self.p)
        self.ones_mat = cp.ones((self.p, self.p))
        self.eye = cp.eye(self.p)
        self.square_alpha_est = args.square_alpha_est

        self.alpha_cap = args.alpha_cap
        self.n_mixture_iters = args.n_mixture_iters

        self.ari_lst = []
        self.Z_hat_lst = []
        self.W_hat_lst = []
        self.U_hat_lst = []

        self.same_label_count = 1

        return

    def generate_Z(self, n=None, p=None, picker=None, mu=None, sd=None, alpha=None):

        if picker is None:
            picker = self.picker_true
        if mu is None:
            mu = self.mu_true
        if sd is None:
            sd = self.sd_true
        if alpha is None:
            alpha = self.alpha_true
        if n is None:
            n = self.n
        if p is None:
            p = self.p

        U = cp.random.normal(size=n)
        shell = mu + alpha[None, :] * sd[None, :] * U[:, None]  # n x r

        common = cp.einsum("ij,ij->i", shell, picker)

        idio_shell = (
            sd[None, None, :]
            * ((1 - alpha[None, None, :] ** 2) ** 0.5)
            * cp.random.normal(0, 1, size=(n, p))[:, :, None]
        )

        idio = cp.einsum("ijk,ik->ij", idio_shell, picker)

        Z = common[:, None] + idio

        return Z, U

    def generate_W(self, picker=None, n=None, r=None, df=None):
        """
        df (cupy array, r x 1): The degrees of freedom parameter for the chi-squared component.
        """

        if df is None:
            df = self.df_true
        if r is None:
            r = self.r
        if n is None:
            n = self.n
        if picker is None:
            picker = self.picker_true

        V = cp.random.uniform(size=n)

        W_shell = cp.vstack(
            [
                cp.sqrt(
                    cp.array(df[j].get() * invgamma.ppf(V.get(), a=df[j].get()/2, scale=1/2)))
                for j in range(r)
            ]
        )

        W = cp.einsum("ji,ij->i", W_shell, picker)

        return W

    def set_params(self, mu=None, sd=None, alpha=None, df=None, pi=None, is_true=False):
        """
        Function to set the estimates manually.
        """
        if not is_true:
            if mu is not None:
                self.mu_hat = mu
            if sd is not None:
                self.sd_hat = sd
            if alpha is not None:
                self.alpha_hat = alpha
            if df is not None:
                self.df_hat = df
            if pi is not None:
                self.pi_hat = pi
        else:
            raise NotImplementedError
        return

    def get_initial_params(self, X_data, fdf=None):
        """
        Uses an nct.fit() scheme to estimate initial parameters crudely. Note that due to the
        time cost associated with the likelihood estimation, we recommend that the sample input 
        be small in both directions (subslice_size and subsample_size) usually.

        Returns (df_hat, mu_hat, sd_hat, gamma_hat, alpha_hat).
        """
        n, p = X_data.shape
        subslice_index = cp.arange(0, p)
        cp.random.shuffle(subslice_index)
        subslice_index = subslice_index[:self.subslice_size]

        subsample_index = cp.arange(0, n)
        cp.random.shuffle(subsample_index)
        subsample_index = subsample_index[:self.subsample_size]

        sample = X_data[:, subslice_index][subsample_index, :]
        if fdf is not None:
            # (df, nc, loc=0, scale)
            a, b, c, d = nct.fit(sample.get(), floc=0, fdf=fdf, method='MLE')
        else:
            # (df, nc, loc=0, scale)
            a, b, c, d = nct.fit(sample.get(), floc=0, method='MLE')

        df_hat = a
        mu_hat = b * d
        sd_hat = d

        # alpha_hat
        T = cp.random.uniform(size=self.var_sample_size)
        cov_hat = cp.cov(X_data.T)
        # Get average of covariance.
        gamma_hat = (cov_hat.sum() - cp.diag(cov_hat).sum()) / (p * (p - 1))
        """
        Cov(X, X') = E[(X - m)(X' - m)]
        = E[(WZ - m)(WZ' - m)]
        = E[W^2 ZZ' - 2WZm + m^2]
        = E[W^2] E[ZZ'] - 2m^2E[W] + m^2
        So E[ZZ'] = (Cov(X,X') - m^2 + 2m^2 E[W]) / E[W^2]
        So Cov(Z, Z') = E[(Z-m)(Z'-m)] = E[ZZ'] - m^2
        So \alpha = \sqrt{(E[ZZ'] - m^2)/s^2}
        """

        if n < 5:
            alpha_hat = cp.array(0.71)
        else:
            alpha_hat = ((cp.corrcoef(X_data.T).sum() - p) /
                         (p * (p - 1))) ** 0.50
            # This is often large, so we can omit the square root and use raw correlation.
            if self.square_alpha_est:  # Good default.
                alpha_hat = alpha_hat ** 2

        del T
        # print(f"Init: {(df_hat, mu_hat, sd_hat)}")
        return (df_hat, mu_hat, sd_hat, gamma_hat.get(), alpha_hat.get())

    def get_naive_proxy(self):
        """        
        This is simply the L^2 proxy with alpha = 0.
        Potentially useful in casees where alpha ~ 0, however the full L^2 proxy
        is already very fast. 
        """
        shell = cp.sqrt(((self.X ** 2).mean(axis=1)[:, None] / (
            (self.sd_hat ** 2 + self.mu_hat ** 2)[None, :])))  # n x r
        proxy = cp.einsum("ij,ij->i", shell, self.picker_hat)
        # print(proxy.shape)

        return proxy

    def print_rmse(self, initial=False):
        """
        Prints the current RMSE of the proxies for W and U (if available).
        """

        w_rmse = rmse(self.W_true, self.W_hat)
        print("W_RMSE: ", w_rmse)
        if not initial:
            u_rmse = rmse(self.U_true, self.U_hat)
            print("U_RMSE: ", u_rmse)
            return w_rmse, u_rmse

        return w_rmse, None

    def get_smart_proxy(self, sep=None, initial=False, suff=True):
        """
        The idea is to first generate a large sized vector of W' (N x 1), Z' , and hence X' (of size N x p), where N very large, say N >> n.
        From X', we then find the sample square-mean statistic S' (N x 1).

        Note: W' and S' are both N x 1. Also, the true S is observed.

        This uses a CUDA-accelerated scheme with PyTorch.

        Now, find which simulants W' produced S' which were closest to S (say, within eps).
        """
        W_hat = cp.zeros(self.n)
        W_hat_naive = self.get_naive_proxy()
        for reg in cp.unique(sep):
            df = self.df_hat[reg]
            mu = self.mu_hat[reg]
            sd = self.sd_hat[reg]
            alpha = self.alpha_hat[reg]
            corr = self.corr_hat[reg]

            if initial:
                # Simulate a large number of W
                a_ = torch.tensor([df.get()/2])
                scale_ = torch.tensor([1/2])
                ig = InverseGamma(a_, scale_)

                W_sim = cp.sqrt(
                    cp.array(ig.rsample((self.sim_size,)) * df.get())
                ).flatten()

                U_sim = cp.random.standard_normal(size=self.sim_size)

                P_sim = (
                    (W_sim ** 2) * (
                        (mu + sd * alpha * U_sim) ** 2
                        + (sd ** 2) * (1 - alpha**2)
                    )
                )  # This is a proxy for S. Note instead of simulating Z', multiplying, and taking the mean, we just take the mean of Z' straight away.

                tol = self.sim_tol
                msq = (self.X[self.labels_hat == reg] ** 2).mean(axis=1)

                # (n, N), mask[i, j] = |S(W_j) - S_i| < tol.

                mask = cp.abs(P_sim - msq[:, None]) < tol

                counts = mask.T.sum(axis=0)

                W_hat_smart = cp.einsum(
                    "i,ij->j", W_sim, mask.T) / (mask.T.sum(axis=0) + 1e-10)

                # Note, mask.T.sum(axis=0)[i] = 0 <==> numerator = 0, so the 1e-10 will not cause any blowups.

                insuff = (counts <= self.nreq)  # (n x 1)
                W_hat_smart[insuff] = W_hat_naive[self.labels_hat == reg][insuff]
                print("Insufficient: ", insuff.sum())

            else:
                row_means = (self.X[self.labels_hat == reg] ** 2).mean(axis=1)
                equicor = corr * cp.ones((self.p, self.p)) + \
                    (1 - corr) * cp.eye(self.p)
                inv = cp.linalg.inv(equicor)
                # cond_var = 1 - ((alpha ** 2) * self.ones @ inv @ self.ones)
                cond_var = (1 - alpha ** 2) / (1 + (self.p - 1) * (alpha ** 2))
                first_mom = self.U_hat[self.labels_hat == reg]
                sec_mom = cond_var + first_mom ** 2
                # print(f"cond_var: {cond_var}, new: {(1 - alpha ** 2) / (1 + (self.p - 1) * (alpha ** 2))}")
                if suff:
                    kernel = (
                        (mu ** 2)
                        + 2 * mu * sd * alpha * first_mom
                        + (sd ** 2) * (alpha ** 2) * sec_mom
                        + (sd ** 2) * (1 - alpha**2)
                    )
                else:
                    kernel = ((mu + sd * alpha * self.U_hat[self.labels_hat == reg]) ** 2
                              + (sd ** 2) * (1 - alpha**2))

                W_hat_smart = cp.sqrt(row_means / kernel)

            W_hat[sep == reg] = W_hat_smart.copy()

        return W_hat

    def get_df_hat(self):
        """
        Gets estimate for \nu via maximum likelihood, using the previous estimate as the initial point.
        Generally quite fast despite using the Digamma function.
        """
        # Transform W_hat to --> 1 / (W_hat ^ 2), which is approx. V/v, V ~ \chi^2_v.
        # MLE involves Digamma function, but is very quick.
        # Note: MME is a good ballpark estimate, and can be used as x0 if needed.

        df_hat = cp.zeros(shape=self.r)

        for reg in cp.unique(self.labels_hat):
            # print(self.W_hat[self.labels_hat == reg])
            invert = 1 / (self.W_hat[self.labels_hat == reg] ** 2)
            sumlog = (cp.log(invert)).sum()
            sumid = invert.sum()

            # y = self.df_hat

            def ll(x): return (
                self.n * (1/x[0]
                          - 0.5 * cp.log(2)
                          - 0.5 * digamma(0.5 * x[0])
                          + 0.5 * (x[0] + x[0] * cp.log(x[0]) - 2) / x[0])
                + 0.5 * sumlog - 0.5 * sumid
            ).get()

            df_hat[reg] = fsolve(ll, x0=self.df_hat[reg].get())[0]

        self.df_hat = df_hat
        return self.df_hat

    def get_factor_est(self):
        """
        Uses the formula for E[U|Z] discussed in the paper. Currently calculates for each regime 
        individually (it would be nice to vectorise this.)
        """

        U_hat = cp.zeros(self.n)
        for reg in cp.unique(self.labels_hat):
            equicor = self.corr_hat[reg] * cp.ones((self.p, self.p)) + (
                1 - self.corr_hat[reg]) * cp.eye(self.p)
            inv = cp.linalg.inv(equicor)
            U_sub = (self.alpha_hat[reg] / self.sd_hat[reg]) * cp.ones(self.p) @ inv @ (
                self.Z_hat[self.labels_hat == reg] - self.mu_hat[reg]).T
            U_hat[self.labels_hat == reg] = U_sub.copy()

        return U_hat

    def get_kmeans_approx(self):  # Assume number of regimes are KNOWN.
        """
        Gets a simple KMeans label estimate for X.
        """
        self.kmeans_obj = KMeans(n_clusters=self.r)
        labels = cp.array(self.kmeans_obj.fit_predict(self.X.get()))
        return labels

    def get_gmm_approx(self):
        """
        Gets a simple GMM label estimate for X.
        """
        self.gmm_obj = GaussianMixture(n_components=self.r, n_init=10)
        gmm_labels = cp.array(self.gmm_obj.fit_predict(self.X.get()))
        return gmm_labels

    def get_picker(self, labels):
        """
        Returns an n x r matrix P with P[i, k] = 1(\eta_i = k), where \eta = labels.
        """
        picker = cp.zeros((self.n, self.r))
        # This is n x r, with picker[i, j] = 1 if obs i is in regime j.
        picker[(cp.arange(0, self.n, 1), labels)] = 1
        return picker

    def generate_param_guess(
        self,
        mu=None, sd=None, alpha=None, pi=None,
        scale=1, alpha_scale=1, pi_chunk_size=60,
        n_seeds=1,
        data=None,
        center_method=False
    ):

        if (data is None) and not center_method:
            raise ValueError(
                "If 'center_method' is False, then 'data' cannot be None.")

        """
        scale (float): We sample \mu_guess ~ N(\mu, scale*\sd^2).
        """
        if n_seeds == 1:
            return (
                mu.reshape(n_seeds, self.r),
                sd.reshape(n_seeds, self.r),
                alpha.reshape(n_seeds, self.r),
                pi.reshape(n_seeds, self.r)
            )

        if center_method:
            mu_guess = cp.random.multivariate_normal(
                mu,
                scale * cp.diag(sd**2),
                size=n_seeds
            )

            sd_guess = cp.sqrt(
                (sd ** 2)[:, None]
                * cp.random.chisquare(self.n-1, (self.r, n_seeds))
                / (self.n-1)
            ).T  # n_seeds x self.r

        else:  # Use data driven approach.
            # First we generate bounds for our parameters.
            mu_lower = data.mean(axis=1).min()
            mu_upper = data.mean(axis=1).max()

            window = 5
            slide = sliding_window_view(data, window, axis=0)
            slider_std = slide.std(axis=-1).mean(axis=1)

            sd_lower = slider_std.min()
            sd_upper = slider_std.max()

            mu_guess = cp.random.uniform(
                mu_lower, mu_upper, size=(n_seeds, self.r))
            sd_guess = cp.random.uniform(
                sd_lower, sd_upper, size=(n_seeds, self.r))

        tform = cp.arctanh(alpha ** 2)
        sampled = cp.random.multivariate_normal(
            tform,
            alpha_scale * cp.eye(self.r) / cp.sqrt(self.n - 3),
            size=n_seeds
        )

        rho_guess = cp.tanh(sampled)
        rho_guess[rho_guess < 0] = 0
        alpha_guess = cp.sqrt(rho_guess)

        pi_core = cp.random.dirichlet(
            pi.get(),
            size=n_seeds*pi_chunk_size
        ).reshape(n_seeds, pi_chunk_size, self.r)

        pi_guess = pi_core.mean(axis=1)

        return mu_guess, sd_guess, alpha_guess, pi_guess

    def redistribute(self, pi, min_thresh=0.03):
        bad = (pi < min_thresh).copy()
        count = bad.sum(axis=1)
        excess = (min_thresh * count - (pi * bad).sum(axis=1))
        excess_ammort = excess / (self.r - count)
        pi[pi < min_thresh] = min_thresh
        pi = pi - ((1-bad) * excess_ammort[:, None])

        return

    def generate_point_estimates(self):
        labels = self.labels_hat
        X = self.X
        unique = cp.unique(labels)
        mu_X_hat = cp.zeros(shape=unique.shape)
        sd_X_hat = cp.zeros(shape=unique.shape)
        alpha_X_hat = cp.zeros(shape=unique.shape)
        for i in cp.unique(unique):
            slice_ = X[labels == i]
            p = slice_.shape[1]
            mu_X_hat[i] = slice_.mean()
            sd_X_hat[i] = slice_.std(axis=0).mean()
            if slice_.shape[0] < 5:
                alpha_X_hat[i] = cp.array(0.71)
            else:
                alpha_X_hat[i] = cp.sqrt(
                    (cp.corrcoef(slice_.T).sum() - p) / (p * (p - 1)))

        return mu_X_hat, sd_X_hat, alpha_X_hat

    def update_lists(self, initial=False):
        if not initial:
            self.Z_hat_lst.append(self.Z_hat)
            self.W_hat_lst.append(self.W_hat)
            self.U_hat_lst.append(self.U_hat)
        if not self.manual:
            self.ari_lst.append(
                adjusted_rand_score(
                    self.labels_true.get(),
                    self.labels_hat.get()
                )
            )

        return

    def print_params(self, only_est=True):

        print("Estimates:")
        print(cp.vstack((self.mu_hat, self.sd_hat,
              self.alpha_hat, self.df_hat, self.pi_hat)).T)
        if only_est:
            return

        print("True:")
        print(cp.vstack((self.mu_true, self.sd_true,
              self.alpha_true, self.df_true, self.pi_true)).T)
        return

    def get_ll(self, mu, sd, alpha, df, pi, labels):
        return

    def run_rmfa(self, n_seeds=None):
        mu_guess, sd_guess, alpha_guess, pi_guess = self.generate_param_guess(
            self.mu_hat,
            self.sd_hat,
            self.alpha_hat,
            self.pi_hat,
            alpha_scale=3,
            pi_chunk_size=20,
            n_seeds=n_seeds,
            center_method=True
        )  # Generate several initial parameter values for MixtureFactor.
        self.redistribute(pi_guess)
        alpha_guess[alpha_guess > self.alpha_cap] = self.alpha_cap

        out_params = MixtureFactor(self.Z_hat, self.r).fit(
            mu_init=mu_guess,
            sd_init=sd_guess,
            alpha_init=alpha_guess,
            pi_init=pi_guess,
            n_iters=self.n_mixture_iters,
            tol=self.tol
        )

        Q_lst = [x[-1][-1][-1] for x in out_params]
        best_index = int(cp.nanargmax(cp.array(Q_lst)))

        self.mu_hat = out_params[best_index][0][0]
        self.sd_hat = self.n / (self.n - 1) * out_params[best_index][0][1]
        self.alpha_hat = out_params[best_index][0][2]
        self.corr_hat = self.alpha_hat ** 2
        self.pi_hat = out_params[best_index][0][3]

        self.labels_hat = out_params[best_index][0][4].argmax(axis=1)
        self.picker_hat = self.get_picker(self.labels_hat)
        return

    def get_norm_params(self):
        """
        This is the direct parameter estimate method, which is ideal for r=1 or for r>1 once regimes are identified.
        """

        unique_labels = cp.unique(self.labels_hat)
        self.corr_hat = cp.zeros(shape=self.r)
        self.mu_hat = cp.zeros(shape=self.r)
        self.sd_hat = cp.zeros(shape=self.r)

        for lab in unique_labels:
            corr = cp.corrcoef(self.Z_hat[self.labels_hat == lab].T)
            p = self.p
            self.corr_hat[lab] = (corr.sum() - p)[None] / (p * (p - 1))
            self.mu_hat[lab] = self.Z_hat[self.labels_hat == lab].mean()[None]
            self.sd_hat[lab] = cp.sqrt(
                self.Z_hat[self.labels_hat == lab].var(axis=0, ddof=1).mean()
            )[None]

        self.corr_hat[self.corr_hat < 0] = 0
        self.alpha_hat = cp.sqrt(self.corr_hat)

        return

    def method_mapper(self, method):
        if method not in ['direct', 'rmfa', 'default']:
            raise ValueError(
                "'method' must be one of ['direct', 'rmfa', 'default'].")
        if method in ['direct', 'rmfa']:
            return method
        if self.r == 1:
            return 'direct'
        return 'rmfa'

    def norm_step(self, method='default', n_seeds=None):

        self.labels_old = copy.deepcopy(self.labels_hat)
        adj_old = (self.labels_old == self.labels_old[:, None])

        final_method = self.method_mapper(method)
        if final_method == 'direct':
            self.get_norm_params()
        else:
            self.run_rmfa(n_seeds)

        adj_new = (self.labels_hat == self.labels_hat[:, None])
        if cp.all(adj_old == adj_new):
            self.same_label_count += 1
        else:
            self.same_label_count = 1

        return

    def generate_point_estimates(self):
        labels = self.labels_hat
        X = self.X
        unique = cp.unique(labels)
        mu_X_hat = cp.zeros(shape=unique.shape)
        sd_X_hat = cp.zeros(shape=unique.shape)
        alpha_X_hat = cp.zeros(shape=unique.shape)
        for i in cp.unique(unique):
            slice_ = X[labels == i]
            p = slice_.shape[1]
            mu_X_hat[i] = slice_.mean()
            sd_X_hat[i] = slice_.std(axis=0).mean()
            if slice_.shape[0] < 5:
                alpha_X_hat[i] = cp.array(0.71)
            else:
                alpha_X_hat[i] = cp.sqrt(
                    (cp.corrcoef(slice_.T).sum() - p) / (p * (p - 1)))

        return mu_X_hat, sd_X_hat, alpha_X_hat

    def run_nct_pass(self, sep=None, fdf=None):
        gridarr = cp.zeros(shape=(self.r, 5))
        for j in cp.unique(sep):
            gridarr[j] = self.get_initial_params(
                X_data=self.X[sep == j],
                fdf=fdf
            )

        self.df_hat = gridarr[:, 0]
        self.df_hat[self.df_hat > 100] = 100
        self.mu_hat = gridarr[:, 1]
        self.sd_hat = gridarr[:, 2]
        self.gamma_hat = gridarr[:, 3]
        self.alpha_hat = gridarr[:, 4]
        self.corr_hat = self.alpha_hat ** 2
        # print(gridarr)

        self.picker_hat = self.get_picker(self.labels_hat)
        return

    def run_initial_step(self, fdf=None, verbose=False):
        self.labels_hat = self.get_kmeans_approx()

        self.pi_hat = (self.labels_hat == cp.arange(
            0, self.r, 1)[:, None]).mean(axis=1)

        if verbose:
            print("========= Initial =========")
            self.print_params()

        return

    def run_second_pass(self, n_init_seeds=None):

        mu_X_hat, sd_X_hat, alpha_X_hat = self.generate_point_estimates()
        mu_X_guess, sd_X_guess, alpha_X_guess, pi_X_guess = self.generate_param_guess(
            data=self.X,
            alpha=alpha_X_hat,
            pi=self.pi_hat,
            alpha_scale=10,
            pi_chunk_size=2,
            n_seeds=n_init_seeds,
            center_method=False
        )
        self.redistribute(pi_X_guess)
        alpha_X_guess[alpha_X_guess > self.alpha_cap] = self.alpha_cap

        self.mu_X_guess = mu_X_guess
        self.sd_X_guess = sd_X_guess
        self.alpha_X_guess = alpha_X_guess
        self.pi_X_guess = pi_X_guess

        M = 20

        slices = [list(range(n_init_seeds))[i: i+M]
                  for i in range(0, n_init_seeds, M)]
        out_paramsX = []
        Q_lst = []
        # print(slices)
        for u, sl in enumerate(slices):
            print(
                f"Running initialisation batch {u+1}/{len(slices)} with batch size {min(M, n_init_seeds)}.")
            out_paramsX.extend(
                MixtureFactor(self.X, self.r).fit(
                    mu_init=mu_X_guess[sl],
                    sd_init=sd_X_guess[sl],
                    alpha_init=alpha_X_guess[sl],
                    pi_init=pi_X_guess[sl],
                    n_iters=self.n_mixture_iters,
                    tol=self.tol
                )
            )
            # print(f"Best Q so far: {max(Q_lst)}")

        Q_lst = [x[-1][-1][-1] for x in out_paramsX]
        self.out_params_slice = out_paramsX[0]

        best_index_X = int(cp.nanargmax(cp.array(Q_lst)))

        self.labels_hat = out_paramsX[best_index_X][0][4].argmax(axis=1)
        self.picker_hat = self.get_picker(self.labels_hat)

        return

    def run_iteration(
        self,
        initial=False,
        always_use_initial_proxy=False,
        n_init_seeds=10,
        n_seeds=10,
        method='default',
        fdf=None,
        verbose=False,
        max_same=5,
        just_initial=False
    ):

        # Run the two-pass initialisation.
        if initial:
            # First pass initialises (labels, pi) nonparametrically (KMeans)

            self.labels_hat = self.get_kmeans_approx()
            self.picker_hat = self.get_picker(self.labels_hat)
            self.pi_hat = (self.labels_hat == cp.arange(
                0, self.r, 1)[:, None]).mean(axis=1)

            self.update_lists(initial=True)

            # Second pass updates the (labels, pi) with RMFA.
            if self.r > 1:
                self.run_second_pass(n_init_seeds=n_init_seeds)
                self.update_lists(initial=True)

            # Finally, initialise (mu, sd, nu, alpha) via scipy.nct.
            self.run_nct_pass(sep=self.labels_hat, fdf=fdf)

        print(self.alpha_hat)

        # Get proxies
        self.W_hat = self.get_smart_proxy(
            initial=initial, sep=self.labels_hat, suff=True)

        self.Z_hat = self.X / (self.W_hat[:, None])

        if just_initial:
            return

        self.get_df_hat()
        self.df_hat[self.df_hat > 100] = 100

        # Update regime labels and normal paramters.
        if self.same_label_count < max_same:
            self.norm_step(n_seeds=n_seeds, method=method)
        else:
            # NEED TO DO THIS PER REGIME!
            self.norm_step(n_seeds=n_seeds, method='direct')

        # Update estimate of common factor(s).
        self.U_hat = self.get_factor_est()

        # Update performance tracking logs.
        self.update_lists()

        return

    def fit(self, iters=10, n_init_seeds=10, n_mixture_seeds=10, method='default', verbose=False, max_same=5, just_initial=False):

        for i in range(iters):
            if verbose:
                print(
                    f"==================== Iteration: {i+1} ====================")
            self.run_iteration(
                initial=(i == 0),
                n_init_seeds=n_init_seeds,
                n_seeds=n_mixture_seeds,
                method=method,
                verbose=verbose,
                max_same=max_same,
                just_initial=just_initial
            )

        return
