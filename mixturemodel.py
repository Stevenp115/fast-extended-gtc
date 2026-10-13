import cupy as cp
import copy


class MixtureFactor:
    """
    Class for estimation of the Gaussian Equicorrelation Mixture Model. Note that this 
    module requires CuPy, and hence a CUDA-supported graphics card.

    Supports batch estimation (simultaneous estimation via a batch of initial values). 

    Be wary of memory limits. For example, testing on a 1000 x 500 x 4 dataset showed VRAM 
    saturation at a ~30 initial value batch on an RTX 4080 Super (16GB). Performance will drop drastically in this scenario.
    """

    def __init__(self, Z, n_regimes):
        """
        Z = observations (assumed normally distributed)
        Note: parameter shapes should be b x r, where b = batch size.
        """
        self.Z = Z  # n x p
        self.n_regimes = n_regimes
        self.n, self.p = Z.shape
        self.eyep = cp.eye(self.p)
        self.ones_matp = cp.ones((self.p, self.p))
        self.ones_vecp = cp.ones(self.p)

        self.xi = (-1 + cp.sqrt(-3, dtype=cp.complex128)) / 2
        self.xi_arr = cp.array([cp.array(1), self.xi, self.xi**2])

    def update_supp_variables(self, mu, sd, alpha, rho):
        """
        Gets the required conditional factor/marginal distribution parameters.
        """
        equicor = self.ones_matp[:, :, None, None] * \
            rho + self.eyep[:, :, None, None] * (1 - rho)
        equicor = cp.transpose(equicor, (2, 3, 0, 1))
        # print("Equicor shape: ", equicor.shape)

        covZ = (sd ** 2)[:, :, None, None] * equicor  # b x r x p x p
        condvarU = (1 - rho) / (1 + (self.p - 1) * rho)  # b x r
        condmeanU = (self.p * alpha / (sd * (1 + (self.p - 1) * rho))) * \
            (self.Z.mean(axis=1)[:, None, None] - mu)  # n x b x r

        return equicor, covZ, condvarU, condmeanU

    def update_log_pdf(self, mu, sd, rho):
        """
        Get the log pdf of Z, conditional on each parameter set.
        """
        logdets = (
            2*self.p*cp.log(sd)
            + (self.p-1)*cp.log(1-rho)
            + cp.log(1+(self.p-1)*rho)
        )  # ~ 0.25ms

        inverses = (
            1 / (sd ** 2)[:, :, None, None] * (
                (1 / (1 - rho))[:, :, None, None] * self.eyep
                - (rho / ((1 - rho) * (1 + (self.p-1)*rho))
                   )[:, :, None, None] * self.ones_matp
            )
        )  # ~ 0.05ms

        Z_centre = self.Z[:, :, None, None] - mu  # n x p x b x r

        lpdfs = (
            - 0.5 * self.p * cp.log(2 * cp.pi)
            - 0.5 * logdets[:, :, None]
            - 0.5 * cp.einsum("ijml,mlik,jkml->mlj", cp.transpose(Z_centre,
                              axes=(1, 0, 2, 3)), inverses, Z_centre)
        )  # ~ 1ms;  b x r x n

        return lpdfs

    def update_Q(self, l_, Q, Q_hist, condmeanU, T, lpdfs, pi):
        delta = cp.inf
        if l_ > 1:
            logpdf_U = (
                - 0.5 * cp.log(2 * cp.pi)
                - 0.5 * condmeanU * condmeanU
            )
            Q_new = (T * (cp.transpose(lpdfs, axes=(2, 0, 1)) +
                     logpdf_U + cp.log(pi))).sum(axis=(0, 2))

            if l_ > 2:
                delta = Q_new - Q

            Q = Q_new

            Q_hist.append(Q)

        return delta, Q

    def update_mu(self, T):
        mu = cp.einsum("ilj,i->lj", T, self.Z.mean(axis=1)) / T.sum(axis=0)
        return mu

    def update_sd(self, condmeanU, T, mu, alpha, rho):
        ssr_full = cp.einsum(
            "ijlk,ijlk->ilk", self.Z[:, :, None, None] - mu, self.Z[:, :, None, None] - mu)

        qc = - cp.einsum("ilj,ilj->lj", T, ssr_full) / (1 - rho)
        qb = alpha * cp.einsum("ilk,ijlk,j->lk", T * condmeanU,
                               self.Z[:, :, None, None] - mu, self.ones_vecp) / (1 - rho)
        qa = self.p * T.sum(axis=0)

        sd = (-qb + cp.sqrt(qb*qb - 4*qa*qc)) / (2 * qa)

        return ssr_full, sd

    def get_cubic_coef(self, ssr_full, condmeanU, condvarU, T, mu, sd):

        polyA = -self.p * cp.einsum("ilj,lj->lj", T, sd**2)
        condsec = condmeanU ** 2 + condvarU
        polyB = cp.einsum("ilj,ilj->lj", T, ssr_full +
                          self.p * (sd ** 2) * condsec)
        polyC = -sd * cp.einsum(
            "ilk,ijlk,j->lk",
            condmeanU * T,
            self.Z[:, :, None, None] - mu,
            self.ones_vecp
        )

        cubeA = -polyA
        cubeB = polyC
        cubeC = polyA + polyB
        cubeD = polyC

        return cubeA, cubeB, cubeC, cubeD

    def get_alpha_mask(self, alpha_test, pi, zero_thresh=1e-6, check_assert=True):
        """
        Gives the b x 3 x r mask to select the correct root to the cubic for each regime
        in each batch. 

        Handles degenerate clusters.
        """
        real_mask = cp.abs(cp.imag(alpha_test)) < zero_thresh
        alpha_real_grid = cp.real(alpha_test)
        bound_mask = ((0 <= alpha_real_grid) & (alpha_real_grid <= 1))
        null_group_mask = cp.abs(pi) < zero_thresh
        correct_mask = (real_mask & bound_mask) | null_group_mask[:, :, None]
        correct_mask = correct_mask & (correct_mask.cumsum(axis=2) == 1)
        correct_mask[~correct_mask.any(axis=2), 0] = True

        # print(correct_mask)
        assert ((correct_mask.sum(axis=2) == 1).all())

        return correct_mask

    def solve_cubic(self, cubeA, cubeB, cubeC, cubeD):

        Delta0 = (cubeB ** 2) - 3 * cubeA * cubeC
        Delta1 = 2 * (cubeB ** 3) - 9 * cubeA * cubeB * \
            cubeC + 27 * (cubeA ** 2) * cubeD
        cubeCon = self.xi_arr * \
            (((Delta1 + cp.sqrt(Delta1**2 - 4*(Delta0**3),
             dtype=cp.complex128)) / 2)**(1/3))[:, :, None]

        alpha_test = (-1 / (3 * cubeA[:, :, None])) * \
            (cubeB[:, :, None] + cubeCon + Delta0[:, :, None] / cubeCon)
        return alpha_test

    def update_alpha(self, ssr_full, condmeanU, condvarU, T, mu, sd, pi, zero_thresh=1e-6):

        cubeA, cubeB, cubeC, cubeD = self.get_cubic_coef(
            ssr_full, condmeanU, condvarU, T, mu, sd)
        condsec = condmeanU ** 2 + condvarU

        # Recall: T is of shape n x b x r, Z[:, :, None, None] is of shape n x p x b x r.
        # condmeanU is n x b x r, condvarU is b x r.
        # ssr_full is n x b x r

        alpha_test = self.solve_cubic(cubeA, cubeB, cubeC, cubeD)

        # Get the mask
        correct_mask = self.get_alpha_mask(
            alpha_test, pi, zero_thresh, check_assert=True)

        alpha = cp.real(alpha_test[correct_mask])
        alpha[(alpha > 1) | (alpha < 0)] = 0

        return alpha.reshape(*mu.shape)

    def update_pi(self, T):
        pi = T.mean(axis=0)
        return pi

    def update_parameters(self, condmeanU, condvarU, T, mu, sd, alpha, rho, pi, zero_thresh=1e-6):
        mu = self.update_mu(T)
        ssr_full, sd = self.update_sd(condmeanU, T, mu, alpha, rho)
        alpha = self.update_alpha(
            ssr_full, condmeanU, condvarU, T, mu, sd, pi, zero_thresh)
        rho = alpha ** 2
        pi = self.update_pi(T)

        return mu, sd, alpha, rho, pi

    def cuda_softmax(self, x, axis=None):
        """
        This implementation is taken exactly from Scipy's source:
        https://github.com/scipy/scipy/blob/v1.18.0/scipy/special/_logsumexp.py#L251-L347

        Only difference is using cupy, so we drop the namespace checks.
        """
        x_max = cp.max(x, axis=axis, keepdims=True)
        exp_x_shifted = cp.exp(x - x_max)
        return exp_x_shifted / cp.sum(exp_x_shifted, axis=axis, keepdims=True)

    def format_histories(self, mu_hist, sd_hist, alpha_hist, pi_hist, labels_hist, Q_hist):
        mu_hist_c = cp.transpose(cp.array(mu_hist), axes=(1, 0, 2))
        sd_hist_c = cp.transpose(cp.array(sd_hist), axes=(1, 0, 2))
        alpha_hist_c = cp.transpose(cp.array(alpha_hist), axes=(1, 0, 2))
        pi_hist_c = cp.transpose(cp.array(pi_hist), axes=(1, 0, 2))
        labels_hist_c = cp.transpose(cp.array(labels_hist), axes=(1, 0, 2))
        Q_hist_c = cp.array(Q_hist).T
        return mu_hist_c, sd_hist_c, alpha_hist_c, pi_hist_c, labels_hist_c, Q_hist_c

    def fit(self, mu_init, sd_init, alpha_init, pi_init, n_iters=100, tol=0.1, plot_mode=False, outer_iter=None, verbose=False):
        mu = copy.deepcopy(mu_init)
        sd = copy.deepcopy(sd_init)
        alpha = copy.deepcopy(alpha_init)
        pi = copy.deepcopy(pi_init)

        rho = alpha ** 2
        T = None
        Q = None
        # Histories
        mu_hist, sd_hist, alpha_hist, pi_hist, labels_hist, Q_hist = [], [], [], [], [], []

        for l_ in range(n_iters):
            # E-step
            equicor, covZ, condvarU, condmeanU = self.update_supp_variables(
                mu, sd, alpha, rho)
            lpdfs = self.update_log_pdf(mu, sd, rho)
            # Q is evaluated with current parameters, but old value for T.
            delta, Q = self.update_Q(l_, Q, Q_hist, condmeanU, T, lpdfs, pi)
            if l_ > 2:
                if (delta < tol).all():
                    print("Change in Q is below tolerance. Terminating.")
                    mu_hist, sd_hist, alpha_hist, pi_hist, labels_hist, Q_hist = self.format_histories(
                        mu_hist, sd_hist, alpha_hist, pi_hist, labels_hist, Q_hist)
                    out = list(
                        zip(
                            zip(mu, sd, alpha, pi, cp.transpose(
                                T, axes=(1, 0, 2))),
                            zip(mu_hist, sd_hist, alpha_hist,
                                pi_hist, labels_hist, Q_hist)
                        )
                    )
                    return out

            T = cp.transpose(self.cuda_softmax(
                # n x b x r
                cp.log(pi)[:, :, None] + lpdfs, axis=1), axes=(2, 0, 1))
            # M-step
            mu, sd, alpha, rho, pi = self.update_parameters(
                condmeanU, condvarU, T, mu, sd, alpha, rho, pi, zero_thresh=1e-6)

            mu_hist.append(copy.deepcopy(mu))
            sd_hist.append(copy.deepcopy(sd))
            alpha_hist.append(copy.deepcopy(alpha))
            pi_hist.append(copy.deepcopy(pi))
            labels_hist.append(T.argmax(axis=2).T)

        mu_hist, sd_hist, alpha_hist, pi_hist, labels_hist, Q_hist = self.format_histories(
            mu_hist, sd_hist, alpha_hist, pi_hist, labels_hist, Q_hist)

        out = list(
            zip(
                zip(mu, sd, alpha, pi, cp.transpose(T, axes=(1, 0, 2))),
                zip(mu_hist, sd_hist, alpha_hist, pi_hist, labels_hist, Q_hist)
            )
        )
        # E.g,. out[0] = (params_seed_1, histories_seed_1).
        # out[0][-1] = (mu_hist_seed_1, ..., Q_hist_seed_1)
        # out[0][-1][-1] = Q_hist_seed_1
        # out[0][-1][-1][-1] = final Q for seed 1.

        return out
