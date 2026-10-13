import cupy as cp
from sklearn.cluster import KMeans
from scipy.stats import invgamma, nct
from scipy.optimize import fsolve
from scipy.special import digamma
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
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
        self.g = args.g

        if (
            hasattr(args, 'mu') and
            hasattr(args, 'sd') and
            hasattr(args, 'df') and
            hasattr(args, 'alpha')
        ):
            self.pi_true = args.pi
            self.mu_true = args.mu
            self.sd_true = args.sd
            self.df_true = args.df
            self.alpha_true = args.alpha
            self.n = args.n
            self.p = args.p

            if args.pi is None:
                self.pi_true = cp.random.dirichlet(
                    alpha=[1 for i in range(self.g)], size=self.g)
                self.redistribute(self.pi_true)

            self.groups_true = cp.random.dirichlet(
                alpha=self.pi_true, size=self.p).argmax(axis=1)

            picker = cp.zeros((self.p, self.g))

            # This is p x g, with picker[i, j] = 1 if feature j is in group l.
            picker[(cp.arange(0, self.p, 1), self.groups_true)] = 1
            self.picker_true = picker

            # Generate components
            self.Z_true, self.U_true = self.generate_Z()
            self.W_true = self.generate_W()  # n x g

            # W @     P^T  =      W~
            # (n x g) x (g, p) ---> (n, p)
            # W~[i, j] = sum_l W[i,l]P[j,l] = W[i, g(j)]
            self.X = self.Z_true * (self.W_true @ self.picker_true.T)

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
        self.labels_list = []
        self.exception = None

        self.same_label_count = 1

        return

    def redistribute(self, pi, min_thresh=0.03):
        bad = (pi < min_thresh).copy()
        count = bad.sum(axis=1)
        excess = (min_thresh * count - (pi * bad).sum(axis=1))
        excess_ammort = excess / (self.r - count)
        pi[pi < min_thresh] = min_thresh
        pi = pi - ((1-bad) * excess_ammort[:, None])

        return

    def generate_Z(self, n=None, p=None, g=None, picker=None, mu=None, sd=None, alpha=None):

        if picker is None:
            picker = self.picker_true  # p x g
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
        if g is None:
            g = self.g

        U = cp.random.normal(size=(n, g))
        common = (mu + alpha[None, :] * sd[None, :] * U) @ picker.T  # n x p
        idio = (
            ((sd[None, :] * ((1 - alpha[None, :] ** 2) ** 0.5)) @ picker.T)
            * cp.random.normal(0, 1, size=(n, p))
        )

        Z = common + idio

        return Z, U

    def generate_W(self, picker=None, n=None, g=None, df=None):
        """
        df (cupy array, r x 1): The degrees of freedom parameter for the chi-squared component.
        """

        if df is None:
            df = self.df_true
        if g is None:
            g = self.g
        if n is None:
            n = self.n
        if picker is None:
            picker = self.picker_true

        V = cp.random.uniform(size=n)

        W = cp.vstack(
            [
                cp.sqrt(
                    cp.array(df[l].get() * invgamma.ppf(V.get(), a=df[l].get()/2, scale=1/2)))
                for l in range(g)
            ]
        ).T

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
            a, b, c, d = nct.fit(sample.get(), floc=0, method='MLE')

        df_hat = a
        mu_hat = b * d
        sd_hat = d

        T = cp.random.uniform(size=self.var_sample_size)
        cov_hat = cp.cov(X_data.T)
        # Get average of covariance.
        gamma_hat = (cov_hat.sum() - cp.diag(cov_hat).sum()) / (p * (p - 1))
        if n < 5:
            alpha_hat = cp.array(0.71)
        else:
            alpha_hat = ((cp.corrcoef(X_data.T).sum() - p) /
                         (p * (p - 1))) ** 0.50
            # This is often large, so we can omit the square root and use raw correlation.
            if self.square_alpha_est:  # Good default.
                alpha_hat = alpha_hat ** 2

        del T
        return (df_hat, mu_hat, sd_hat, gamma_hat.get(), alpha_hat.get())

    def get_naive_proxy(self):
        """        
        This is simply the L^2 proxy with alpha = 0.
        Potentially useful in cases where alpha ~ 0, however the full L^2 proxy
        is already very fast. 
        """
        proxy = cp.zeros((self.n, self.g))

        for l in range(self.g):
            proxy[:, l] = cp.sqrt((self.X[:, self.groups_hat == l] ** 2).mean(
                axis=1) / (self.sd_hat[l] ** 2 + self.mu_hat[l] ** 2))

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

    def get_smart_proxy(self, initial=False, suff=True, force_unit_mean=False):
        """
        The idea is to first generate a large sized vector of W' (N x 1), Z' , and hence X' (of size N x p), where N very large, say N >> n.
        From X', we then find the sample square-mean statistic S' (N x 1).

        Note: W' and S' are both N x 1. Also, the true S is observed.

        This uses a CUDA-accelerated scheme with PyTorch.

        Now, find which simulants W' produced S' which were closest to S (say, within eps).
        """
        W_hat = cp.zeros((self.n, self.g))
        W_hat_naive = self.get_naive_proxy()
        sep = self.groups_hat

        for l in cp.unique(sep):
            df = self.df_hat[l]
            mu = self.mu_hat[l]
            sd = self.sd_hat[l]
            alpha = self.alpha_hat[l]
            corr = self.corr_hat[l]
            p = self.counts[l]
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
                )  # This is a proxy for S.

                tol = self.sim_tol
                msq = (self.X[:, self.groups_hat == l] ** 2).mean(axis=1)

                # (n, N), mask[i, j] = |S(W_j) - S_i| < tol.

                mask = cp.abs(P_sim - msq[:, None]) < tol
                counts = mask.T.sum(axis=0)

                W_hat_smart = cp.einsum(
                    "i,ij->j", W_sim, mask.T) / (mask.T.sum(axis=0) + 1e-10)

                # Note, mask.T.sum(axis=0)[i] = 0 <==> numerator = 0, so the 1e-10 will not cause any blowups.

                insuff = (counts <= self.nreq)  # (n x 1)
                W_hat_smart[insuff] = W_hat_naive[:, l][insuff]

            else:
                row_means = (self.X[:, self.groups_hat == l] ** 2).mean(axis=1)
                cond_var = (1 - alpha ** 2) / (1 + (p - 1) * (alpha ** 2))
                first_mom = self.U_hat[:, l]
                sec_mom = cond_var + first_mom ** 2
                if suff:
                    kernel = (
                        (mu ** 2)
                        + 2 * mu * sd * alpha * first_mom
                        + (sd ** 2) * (alpha ** 2) * sec_mom
                        + (sd ** 2) * (1 - alpha**2)
                    )
                else:
                    kernel = ((mu + sd * alpha * self.U_hat[:, l]) ** 2
                              + (sd ** 2) * (1 - alpha**2))

                W_hat_smart = cp.sqrt(row_means / kernel)

            W_hat[:, l] = W_hat_smart.copy()

        if force_unit_mean:
            W_hat[W_hat < 1e-6] = 1e-6
            inv_sq_W_hat = 1 / (W_hat ** 2)
            W_hat = 1 / cp.sqrt(inv_sq_W_hat /
                                inv_sq_W_hat.mean(axis=0)[None, :])

        return W_hat

    def get_fast_inv(self, alpha, p):
        return (1 / (1 - alpha ** 2)) * (cp.eye(p) - cp.ones((p, p)) * (alpha ** 2) / (1 + (p - 1) * (alpha ** 2)))

    def get_df_hat(self):
        """
        Gets estimate for \nu via maximum likelihood, using the previous estimate as the initial point.
        Generally quite fast despite using the Digamma function.
        """
        # Transform W_hat to --> 1 / (W_hat ^ 2), which is approx. V/v, V ~ \chi^2_v.
        # MLE involves Digamma function, but is very quick.
        # Note: MME is a good ballpark estimate, and can be used as x0 if needed.

        df_hat = cp.zeros(shape=self.g)

        for gr in cp.unique(self.groups_hat):
            invert = 1 / (self.W_hat[:, gr] ** 2)
            sumlog = (cp.log(invert)).sum()
            sumid = invert.sum()

            def ll(x): return (
                self.n * (1/x[0]
                          - 0.5 * cp.log(2)
                          - 0.5 * digamma(0.5 * x[0])
                          + 0.5 * (x[0] + x[0] * cp.log(x[0]) - 2) / x[0])
                + 0.5 * sumlog - 0.5 * sumid
            ).get()

            df_hat[gr] = fsolve(ll, x0=self.df_hat[gr].get())[0]

        self.df_hat = df_hat
        return self.df_hat

    def get_factor_est(self):
        """
        Uses the formula for E[U|Z] discussed in the paper. Currently calculates for each regime 
        individually (it would be nice to vectorise this.)
        """

        U_hat = cp.zeros((self.n, self.g))
        for gr in cp.unique(self.groups_hat):
            pgr = (self.groups_hat == gr).sum().get()
            inv = self.get_fast_inv(self.alpha_hat[gr], pgr)
            U_sub = (self.alpha_hat[gr] / self.sd_hat[gr]) * cp.ones(pgr) @ inv @ (
                self.Z_hat[:, self.groups_hat == gr] - self.mu_hat[gr]).T
            U_hat[:, gr] = U_sub.copy()

        return U_hat

    def get_kmeans_approx(self):
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
        picker = cp.zeros((self.p, self.g))
        picker[(cp.arange(0, self.p, 1), labels)] = 1
        return picker

    def update_lists(self, initial=False):
        if not initial:
            self.Z_hat_lst.append(self.Z_hat)
            self.W_hat_lst.append(self.W_hat)
            self.U_hat_lst.append(self.U_hat)
        self.ari_lst.append(
            adjusted_rand_score(
                self.labels_true.get(),
                self.groups_hat.get()
            )
        )

        return

    def print_params(self, only_est=True):

        print("Estimates:")
        print(cp.vstack((self.mu_hat, self.sd_hat,
              self.alpha_hat, self.df_hat)).T)
        if only_est:
            return

        print("True:")
        print(cp.vstack((self.mu_true, self.sd_true,
              self.alpha_true, self.df_true)).T)
        return

    def get_ll(self, mu, sd, alpha, df, pi, labels):
        return

    def get_norm_params(self):
        """
        This is the direct parameter estimate method, which is ideal for r=1 or for r>1 once regimes are identified.
        """

        unique_groups = cp.unique(self.groups_hat)
        self.corr_hat = cp.zeros(shape=self.g)
        self.mu_hat = cp.zeros(shape=self.g)
        self.sd_hat = cp.zeros(shape=self.g)

        for gr in unique_groups:
            Z_slice = self.Z_hat[:, self.groups_hat == gr]
            corr = cp.corrcoef(Z_slice.T)
            p = (self.groups_hat == gr).sum()
            self.corr_hat[gr] = (corr.sum() - p)[None] / (p * (p - 1))
            self.mu_hat[gr] = Z_slice.mean()[None]
            self.sd_hat[gr] = cp.sqrt(
                Z_slice.var(axis=0, ddof=1).mean()
            )[None]

        self.corr_hat[self.corr_hat < 0] = 0
        self.alpha_hat = cp.sqrt(self.corr_hat)

        return

    def run_nct_pass(self, fdf=None):
        sep = self.groups_hat
        gridarr = cp.zeros(shape=(self.g, 5))
        for l in cp.unique(sep):
            gridarr[l] = self.get_initial_params(
                X_data=self.X[:, sep == l],
                fdf=fdf
            )

        self.df_hat = gridarr[:, 0]
        self.df_hat[self.df_hat > 100] = 100
        self.mu_hat = gridarr[:, 1]
        self.sd_hat = gridarr[:, 2]
        self.gamma_hat = gridarr[:, 3]
        self.alpha_hat = gridarr[:, 4]
        self.corr_hat = self.alpha_hat ** 2

        self.picker_hat = self.get_picker(self.groups_hat)
        return

    def run_blockbuster(self, data):
        """
        Assumes the features are columns.
        """
        g = self.g
        _, eVec = cp.linalg.eigh(cp.cov(data.T))
        slice_ = eVec[:, -g:]
        norms = cp.sqrt((slice_ ** 2).sum(axis=1))
        norm_slice = slice_ / norms[:, None]

        est_labels = KMeans(n_clusters=g).fit_predict(norm_slice.get())

        return cp.array(est_labels)

    def get_counts(self):
        return (self.groups_hat[:, None] == cp.arange(0, self.g)).sum(axis=0)

    def run_iteration(
        self,
        initial=False,
        always_use_initial_proxy=False,
        fdf=None,
        verbose=False,
        force_unit_mean=False,
        final=False
    ):

        if initial:
            self.groups_hat = self.run_blockbuster(self.X)
            self.labels_list.append(self.groups_hat)
            self.counts = self.get_counts()
            self.run_nct_pass(fdf=fdf)

        # Get proxies
        self.W_hat = self.get_smart_proxy(
            # Prox.
            initial=initial, suff=True, force_unit_mean=force_unit_mean)
        self.W_hat_lst.append(self.W_hat)
        self.Z_hat = self.X / (self.W_hat @ self.picker_hat.T)

        self.groups_hat = self.run_blockbuster(self.Z_hat)
        self.counts = self.get_counts()
        self.labels_list.append(self.groups_hat)

        if final:
            self.get_df_hat()
            self.df_hat[self.df_hat > 100] = 100

        self.get_norm_params()

        # Update estimate of common factor(s).
        self.U_hat = self.get_factor_est()
        if verbose:
            self.print_params()

        return

    def fit(self, iters=10, verbose=False, force_unit_mean=False):

        for i in range(iters):
            if verbose:
                print(
                    f"==================== Iteration: {i+1} ====================")
            self.run_iteration(
                initial=(i == 0),
                verbose=verbose,
                force_unit_mean=force_unit_mean,
                final=(i == (iters-1))
            )

        return
