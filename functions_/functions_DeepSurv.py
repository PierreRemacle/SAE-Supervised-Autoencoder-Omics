from math import inf
from lifelines import KaplanMeierFitter
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch
import torch.nn as nn
from torch import Tensor
from torchtuples import Model, optim, tuplefy
import torchtuples as tt

# from pycox.datasets import metabric

from datetime import datetime, timedelta
import os
import warnings

import warnings
import numba

import ot
def wasserstein_distance(u, v): return ot.wasserstein_1d(u, v, p=2)


def idx_at_times(index_surv, times, steps='pre', assert_sorted=True):
    """Gives index of `index_surv` corresponding to `time`, i.e.
    `index_surv[idx_at_times(index_surv, times)]` give the values of `index_surv`
    closet to `times`.

    Arguments:
        index_surv {np.array} -- Durations of survival estimates
        times {np.array} -- Values one want to match to `index_surv`

    Keyword Arguments:
        steps {str} -- Round 'pre' (closest value higher) or 'post'
          (closest value lower) (default: {'pre'})
        assert_sorted {bool} -- Assert that index_surv is monotone (default: {True})

    Returns:
        np.array -- Index of `index_surv` that is closest to `times`
    """
    if assert_sorted:
        assert pd.Series(
            index_surv).is_monotonic_increasing, "Need 'index_surv' to be monotonic increasing"
    if steps == 'pre':
        idx = np.searchsorted(index_surv, times)
    elif steps == 'post':
        idx = np.searchsorted(index_surv, times, side='right') - 1
    return idx.clip(0, len(index_surv)-1)


@numba.njit
def _group_loop(n, surv_idx, durations, events, di, ni):
    idx = 0
    for i in range(n):
        idx += durations[i] != surv_idx[idx]
        di[idx] += events[i]
        ni[idx] += 1
    return di, ni


def kaplan_meier(durations, events, start_duration=0):
    """A very simple Kaplan-Meier fitter. For a more complete implementation
    see `lifelines`.

    Arguments:
        durations {np.array} -- durations array
        events {np.arrray} -- events array 0/1

    Keyword Arguments:
        start_duration {int} -- Time start as `start_duration`. (default: {0})

    Returns:
        pd.Series -- Kaplan-Meier estimates.
    """
    n = len(durations)
    assert n == len(events)
    if start_duration > durations.min():
        warnings.warn(f"start_duration {start_duration} is larger than minimum duration {durations.min()}. "
                      "If intentional, consider changing start_duration when calling kaplan_meier.")
    order = np.argsort(durations)
    durations = durations[order]
    events = events[order]
    surv_idx = np.unique(durations)
    ni = np.zeros(len(surv_idx), dtype='int')
    di = np.zeros_like(ni)
    di, ni = _group_loop(n, surv_idx, durations, events, di, ni)
    ni = n - ni.cumsum()
    ni[1:] = ni[:-1]
    ni[0] = n
    survive = 1 - di / ni
    zero_survive = survive == 0
    if zero_survive.any():
        i = np.argmax(zero_survive)
        surv = np.zeros_like(survive)
        surv[:i] = np.exp(np.log(survive[:i]).cumsum())
        # surv[i:] = surv[i-1]
        surv[i:] = 0.
    else:
        surv = np.exp(np.log(1 - di / ni).cumsum())
    if start_duration < surv_idx.min():
        tmp = np.ones(len(surv) + 1, dtype=surv.dtype)
        tmp[1:] = surv
        surv = tmp
        tmp = np.zeros(len(surv_idx) + 1, dtype=surv_idx.dtype)
        tmp[1:] = surv_idx
        surv_idx = tmp
    surv = pd.Series(surv, surv_idx)
    return surv


class EvalSurv:
    """Class for evaluating predictions.

    Arguments:
        surv {pd.DataFrame} -- Survival predictions.
        durations {np.array} -- Durations of test set.
        events {np.array} -- Events of test set.

    Keyword Arguments:
        censor_surv {str, pd.DataFrame, EvalSurv} -- Censoring distribution.
            If provided data frame (survival function for censoring) or EvalSurv object,
            this will be used.
            If 'km', we will fit a Kaplan-Meier to the dataset.
            (default: {None})
        censor_durations {np.array}: -- Administrative censoring times. (default: {None})
        steps {str} -- For durations between values of `surv.index` choose the higher index 'pre'
            or lower index 'post'. For a visualization see `help(EvalSurv.steps)`. (default: {'post'})
    """

    def __init__(self, surv, durations, events, censor_surv=None, censor_durations=None, steps='post'):
        assert (type(durations) == type(events) ==
                np.ndarray), 'Need `durations` and `events` to be arrays'
        self.surv = surv
        self.durations = durations
        self.events = events
        self.censor_surv = censor_surv
        self.censor_durations = censor_durations
        self.steps = steps
        assert pd.Series(self.index_surv).is_monotonic_increasing

    @property
    def censor_surv(self):
        """Estimated survival for censorings.
        Also an EvalSurv object.
        """
        return self._censor_surv

    @censor_surv.setter
    def censor_surv(self, censor_surv):
        if isinstance(censor_surv, EvalSurv):
            self._censor_surv = censor_surv
        elif type(censor_surv) is str:
            if censor_surv == 'km':
                self.add_km_censor()
            else:
                txt = f"censor_surv cannot be {censor_surv}. Use e.g. 'km'"
                raise ValueError(txt)

        elif censor_surv is not None:
            self.add_censor_est(censor_surv)
        else:
            self._censor_surv = None

    @property
    def index_surv(self):
        return self.surv.index.values

    @property
    def steps(self):
        """How to handle predictions that are between two indexes in `index_surv`.

        For a visualization, run the following:
            ev = EvalSurv(pd.DataFrame(np.linspace(1, 0, 7)),
                          np.empty(7), np.ones(7), steps='pre')
            ax = ev[0].plot_surv()
            ev.steps = 'post'
            ev[0].plot_surv(ax=ax, style='--')
            ax.legend(['pre', 'post'])
        """
        return self._steps

    @steps.setter
    def steps(self, steps):
        vals = ['post', 'pre']
        if steps not in vals:
            raise ValueError(f"`steps` needs to be {vals}, got {steps}")
        self._steps = steps

    def add_censor_est(self, censor_surv, steps='post'):
        """Add censoring estimates so one can use inverse censoring weighting.
        `censor_surv` are the survival estimates trained on (durations, 1-events),

        Arguments:
            censor_surv {pd.DataFrame} -- Censor survival curves.

    Keyword Arguments:
        round {str} -- For durations between values of `surv.index` choose the higher index 'pre'
            or lower index 'post'. If `None` use `self.steps` (default: {None})
        """
        if not isinstance(censor_surv, EvalSurv):
            censor_surv = self._constructor(censor_surv, self.durations, 1-self.events, None,
                                            steps=steps)
        self.censor_surv = censor_surv
        return self

    def add_km_censor(self, steps='post'):
        """Add censoring estimates obtained by Kaplan-Meier on the test set
        (durations, 1-events).
        """
        km = kaplan_meier(self.durations, 1-self.events)
        surv = pd.DataFrame(np.repeat(km.values.reshape(-1, 1), len(self.durations), axis=1),
                            index=km.index)
        return self.add_censor_est(surv, steps)

    @property
    def censor_durations(self):
        """Administrative censoring times."""
        return self._censor_durations

    @censor_durations.setter
    def censor_durations(self, val):
        if val is not None:
            assert (self.durations[self.events == 0] == val[self.events == 0]).all(), \
                'Censored observations need same `durations` and `censor_durations`'
            assert (self.durations[self.events == 1] <= val[self.events == 1]).all(), \
                '`durations` cannot be larger than `censor_durations`'
            if (self.durations == val).all():
                warnings.warn("`censor_durations` are equal to `durations`." +
                              " `censor_durations` are likely wrong!")
            self._censor_durations = val
        else:
            self._censor_durations = val

    @property
    def _constructor(self):
        return EvalSurv

    def __getitem__(self, index):
        if not (hasattr(index, '__iter__') or type(index) is slice):
            index = [index]
        surv = self.surv.iloc[:, index]
        durations = self.durations[index]
        events = self.events[index]
        new = self._constructor(surv, durations, events,
                                None, steps=self.steps)
        if self.censor_surv is not None:
            new.censor_surv = self.censor_surv[index]
        return new

    def plot_surv(self, **kwargs):
        """Plot survival estimates.
        kwargs are passed to `self.surv.plot`.
        """
        if len(self.durations) > 50:
            raise RuntimeError(
                "We don't allow to plot more than 50 lines. Use e.g. `ev[1:5].plot()`")
        if 'drawstyle' in kwargs:
            raise RuntimeError(
                f"`drawstyle` is set by `self.steps`. Remove from **kwargs")
        return self.surv.plot(drawstyle=f"steps-{self.steps}", **kwargs)

    def idx_at_times(self, times):
        """Get the index (iloc) of the `surv.index` closest to `times`.
        I.e. surv.loc[tims] (almost)= surv.iloc[idx_at_times(times)].

        Useful for finding predictions at given durations.
        """
        return idx_at_times(self.index_surv, times, self.steps)

    def _duration_idx(self):
        return self.idx_at_times(self.durations)

    def surv_at_times(self, times):
        idx = self.idx_at_times(times)
        return self.surv.iloc[idx]

    # def prob_alive(self, time_grid):
    #     return self.surv_at_times(time_grid).values

    def concordance_td(self, method='adj_antolini'):
        """Time dependent concorance index from
        Antolini, L.; Boracchi, P.; and Biganzoli, E. 2005. A time-dependent discrimination
        index for survival data. Statistics in Medicine 24:3927–3944.

        If 'method' is 'antolini', the concordance from Antolini et al. is computed.

        If 'method' is 'adj_antolini' (default) we have made a small modifications
        for ties in predictions and event times.
        We have followed step 3. in Sec 5.1. in Random Survival Forests paper, except for the last
        point with "T_i = T_j, but not both are deaths", as that doesn't make much sense.
        See 'metrics._is_concordant'.

        Keyword Arguments:
            method {str} -- Type of c-index 'antolini' or 'adj_antolini' (default {'adj_antolini'}).

        Returns:
            float -- Time dependent concordance index.
        """
        return concordance_td(self.durations, self.events, self.surv.values,
                              self._duration_idx(), method)

    def brier_score(self, time_grid, max_weight=np.inf):
        """Brier score weighted by the inverse censoring distribution.
        See Section 3.1.2 or [1] for details of the wighting scheme.

        Arguments:
            time_grid {np.array} -- Durations where the brier score should be calculated.

        Keyword Arguments:
            max_weight {float} -- Max weight value (max number of individuals an individual
                can represent (default {np.inf}).

        References:
            [1] Håvard Kvamme and Ørnulf Borgan. The Brier Score under Administrative Censoring: Problems
                and Solutions. arXiv preprint arXiv:1912.08581, 2019.
                https://arxiv.org/pdf/1912.08581.pdf
        """
        if self.censor_surv is None:
            raise ValueError("""Need to add censor_surv to compute Brier score. Use 'add_censor_est'
            or 'add_km_censor' for Kaplan-Meier""")
        print("#debug brier_score")
        print(self.censor_surv.surv.values.min())
        self.censor_surv.surv.values

        print(np.isnan(self.surv.values).sum())  # Count NaNs
        print((self.surv.values < 0).sum())      # Count negatives
        print((self.surv.values > 1).sum())      # Count > 1
        bs = brier_score(time_grid, self.durations, self.events, self.surv.values,
                         self.censor_surv.surv.values + 1e-6, self.index_surv,
                         self.censor_surv.index_surv, max_weight, True, self.steps,
                         self.censor_surv.steps)
        print("#debug brier_score end")
        return pd.Series(bs, index=time_grid).rename('brier_score')

    def nbll(self, time_grid, max_weight=np.inf):
        """Negative binomial log-likelihood weighted by the inverse censoring distribution.
        See Section 3.1.2 or [1] for details of the wighting scheme.

        Arguments:
            time_grid {np.array} -- Durations where the brier score should be calculated.

        Keyword Arguments:
            max_weight {float} -- Max weight value (max number of individuals an individual
                can represent (default {np.inf}).

        References:
            [1] Håvard Kvamme and Ørnulf Borgan. The Brier Score under Administrative Censoring: Problems
                and Solutions. arXiv preprint arXiv:1912.08581, 2019.
                https://arxiv.org/pdf/1912.08581.pdf
        """
        if self.censor_surv is None:
            raise ValueError("""Need to add censor_surv to compute the score. Use 'add_censor_est'
            or 'add_km_censor' for Kaplan-Meier""")
        bll = binomial_log_likelihood(time_grid, self.durations, self.events, self.surv.values,
                                      self.censor_surv.surv.values, self.index_surv,
                                      self.censor_surv.index_surv, max_weight, True, self.steps,
                                      self.censor_surv.steps)
        return pd.Series(-bll, index=time_grid).rename('nbll')

    def integrated_brier_score(self, time_grid, max_weight=np.inf):
        """Integrated Brier score weighted by the inverse censoring distribution.
        Essentially an integral over values obtained from `brier_score(time_grid, max_weight)`.

        Arguments:
            time_grid {np.array} -- Durations where the brier score should be calculated.

        Keyword Arguments:
            max_weight {float} -- Max weight value (max number of individuals an individual
                can represent (default {np.inf}).
        """
        if self.censor_surv is None:
            raise ValueError(
                "Need to add censor_surv to compute briser score. Use 'add_censor_est'")
        return integrated_brier_score(time_grid, self.durations, self.events, self.surv.values,
                                      self.censor_surv.surv.values, self.index_surv,
                                      self.censor_surv.index_surv, max_weight, self.steps,
                                      self.censor_surv.steps)

    def integrated_nbll(self, time_grid, max_weight=np.inf):
        """Integrated negative binomial log-likelihood weighted by the inverse censoring distribution.
        Essentially an integral over values obtained from `nbll(time_grid, max_weight)`.

        Arguments:
            time_grid {np.array} -- Durations where the brier score should be calculated.

        Keyword Arguments:
            max_weight {float} -- Max weight value (max number of individuals an individual
                can represent (default {np.inf}).
        """
        if self.censor_surv is None:
            raise ValueError(
                "Need to add censor_surv to compute the score. Use 'add_censor_est'")
        ibll = integrated_binomial_log_likelihood(time_grid, self.durations, self.events, self.surv.values,
                                                  self.censor_surv.surv.values, self.index_surv,
                                                  self.censor_surv.index_surv, max_weight, self.steps,
                                                  self.censor_surv.steps)
        return -ibll

    def brier_score_admin(self, time_grid):
        """The Administrative Brier score proposed by [1].
        Removes individuals as they are administratively censored, event if they have experienced an
        event.

        Arguments:
            time_grid {np.array} -- Durations where the brier score should be calculated.

        References:
            [1] Håvard Kvamme and Ørnulf Borgan. The Brier Score under Administrative Censoring: Problems
                and Solutions. arXiv preprint arXiv:1912.08581, 2019.
                https://arxiv.org/pdf/1912.08581.pdf
        """
        if self.censor_durations is None:
            raise ValueError(
                "Need to provide `censor_durations` (censoring durations) to use this method")
        bs = admin.brier_score(time_grid, self.durations, self.censor_durations, self.events,
                               self.surv.values, self.index_surv, True, self.steps)
        return pd.Series(bs, index=time_grid).rename('brier_score')

    def integrated_brier_score_admin(self, time_grid):
        """The Integrated administrative Brier score proposed by [1].
        Removes individuals as they are administratively censored, event if they have experienced an
        event.

        Arguments:
            time_grid {np.array} -- Durations where the brier score should be calculated.

        References:
            [1] Håvard Kvamme and Ørnulf Borgan. The Brier Score under Administrative Censoring: Problems
                and Solutions. arXiv preprint arXiv:1912.08581, 2019.
                https://arxiv.org/pdf/1912.08581.pdf
        """
        if self.censor_durations is None:
            raise ValueError(
                "Need to provide `censor_durations` (censoring durations) to use this method")
        ibs = admin.integrated_brier_score(time_grid, self.durations, self.censor_durations, self.events,
                                           self.surv.values, self.index_surv, self.steps)
        return ibs

    def nbll_admin(self, time_grid):
        """The negative administrative binomial log-likelihood proposed by [1].
        Removes individuals as they are administratively censored, event if they have experienced an
        event.

        Arguments:
            time_grid {np.array} -- Durations where the brier score should be calculated.

        References:
            [1] Håvard Kvamme and Ørnulf Borgan. The Brier Score under Administrative Censoring: Problems
                and Solutions. arXiv preprint arXiv:1912.08581, 2019.
                https://arxiv.org/pdf/1912.08581.pdf
        """
        if self.censor_durations is None:
            raise ValueError(
                "Need to provide `censor_durations` (censoring durations) to use this method")
        bll = admin.binomial_log_likelihood(time_grid, self.durations, self.censor_durations, self.events,
                                            self.surv.values, self.index_surv, True, self.steps)
        return pd.Series(-bll, index=time_grid).rename('nbll')

    def integrated_nbll_admin(self, time_grid):
        """The Integrated negative administrative binomial log-likelihood score proposed by [1].
        Removes individuals as they are administratively censored, event if they have experienced an
        event.

        Arguments:
            time_grid {np.array} -- Durations where the brier score should be calculated.

        References:
            [1] Håvard Kvamme and Ørnulf Borgan. The Brier Score under Administrative Censoring: Problems
                and Solutions. arXiv preprint arXiv:1912.08581, 2019.
                https://arxiv.org/pdf/1912.08581.pdf
        """
        if self.censor_durations is None:
            raise ValueError(
                "Need to provide `censor_durations` (censoring durations) to use this method")
        ibll = admin.integrated_binomial_log_likelihood(time_grid, self.durations, self.censor_durations,
                                                        self.events, self.surv.values, self.index_surv,
                                                        self.steps)
        return -ibll


@numba.njit(parallel=True)
def _inv_cens_scores(func, time_grid, durations, events, surv, censor_surv, idx_ts_surv, idx_ts_censor,
                     idx_tt_censor, scores, weights, n_times, n_indiv, max_weight):
    def _inv_cens_score_single(func, ts, durations, events, surv, censor_surv, idx_ts_surv_i,
                               idx_ts_censor_i, idx_tt_censor, scores, weights, n_indiv, max_weight):
        min_g = 1./max_weight
        for i in range(n_indiv):
            tt = durations[i]
            d = events[i]
            s = surv[idx_ts_surv_i, i]
            g_ts = censor_surv[idx_ts_censor_i, i]
            g_tt = censor_surv[idx_tt_censor[i], i]
            g_ts = max(g_ts, min_g)
            g_tt = max(g_tt, min_g)
            score, w = func(ts, tt, s, g_ts, g_tt, d)
            # w = min(w, max_weight)
            scores[i] = score * w
            weights[i] = w

    for i in numba.prange(n_times):
        ts = time_grid[i]
        idx_ts_surv_i = idx_ts_surv[i]
        idx_ts_censor_i = idx_ts_censor[i]
        scores_i = scores[i]
        weights_i = weights[i]
        _inv_cens_score_single(func, ts, durations, events, surv, censor_surv, idx_ts_surv_i,
                               idx_ts_censor_i, idx_tt_censor, scores_i, weights_i, n_indiv, max_weight)


def _inverse_censoring_weighted_metric(func):
    if not func.__class__.__module__.startswith('numba'):
        raise ValueError("Need to provide numba compiled function")
    def metric(time_grid, durations, events, surv, censor_surv, index_surv, index_censor, max_weight=np.inf,
               reduce=True, steps_surv='post', steps_censor='post'):
        if not hasattr(time_grid, '__iter__'):
            time_grid = np.array([time_grid])
        assert (type(time_grid) is type(durations) is type(events) is type(surv) is type(censor_surv) is
                type(index_surv) is type(index_censor) is np.ndarray), 'Need all input to be np.ndarrays'
        n_times = len(time_grid)
        n_indiv = len(durations)
        scores = np.zeros((n_times, n_indiv))
        weights = np.zeros((n_times, n_indiv))
        idx_ts_surv = idx_at_times(
            index_surv, time_grid, steps_surv, assert_sorted=True)
        idx_ts_censor = idx_at_times(
            index_censor, time_grid, steps_censor, assert_sorted=True)
        idx_tt_censor = idx_at_times(
            index_censor, durations, 'pre', assert_sorted=True)
        if steps_censor == 'post':
            idx_tt_censor = (idx_tt_censor - 1).clip(0)
            #  This ensures that we get G(tt-)
        _inv_cens_scores(func, time_grid, durations, events, surv, censor_surv, idx_ts_surv, idx_ts_censor,
                         idx_tt_censor, scores, weights, n_times, n_indiv, max_weight)
        if reduce is True:
            return np.sum(scores, axis=1) / (np.sum(weights, axis=1)+1e-60)
        return scores, weights
    return metric


@numba.njit()
def _brier_score(ts, tt, s, g_ts, g_tt, d):
    if (tt <= ts) and d == 1:
        if g_tt == 0:
            print("gtt is zero")
            print(0/0)
        return np.power(s, 2), 1./g_tt
    if tt > ts:
        if g_ts == 0:
            print("gts is zero")
            print(0/0)
        return np.power(1 - s, 2), 1./g_ts
    return 0., 0.


@numba.njit()
def _binomial_log_likelihood(ts, tt, s, g_ts, g_tt, d, eps=1e-7):
    s = eps if s < eps else s
    s = (1-eps) if s > (1 - eps) else s
    if (tt <= ts) and d == 1:
        return np.log(1 - s), 1./g_tt
    if tt > ts:
        return np.log(s), 1./g_ts
    return 0., 0.


brier_score = _inverse_censoring_weighted_metric(_brier_score)
binomial_log_likelihood = _inverse_censoring_weighted_metric(
    _binomial_log_likelihood)


def _integrated_inverce_censoring_weighed_metric(func):
    def metric(time_grid, durations, events, surv, censor_surv, index_surv, index_censor,
               max_weight=np.inf, steps_surv='post', steps_censor='post'):
        scores = func(time_grid, durations, events, surv, censor_surv, index_surv, index_censor,
                      max_weight, True, steps_surv, steps_censor)
        integral = scipy.integrate.simpson(y=scores, x=time_grid)
        return integral / (time_grid[-1] - time_grid[0])
    return metric


integrated_brier_score = _integrated_inverce_censoring_weighed_metric(
    brier_score)
integrated_binomial_log_likelihood = _integrated_inverce_censoring_weighed_metric(
    binomial_log_likelihood)


class CoxPHLoss(torch.nn.Module):
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.

    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """

    def forward(self, log_h: Tensor, durations: Tensor, events: Tensor, lambda_wass: float = 0.1) -> Tensor:
        if torch.sum(events) == 0:
            return torch.tensor(0.0, requires_grad=True).to(log_h.device)
        text = "log_h" + str(log_h) + "duration" + \
            str(durations) + "events" + str(events)
        # print("durations")
        # print(durations.shape)
        loss = cox_ph_loss(log_h, durations, events, lambda_wass, 1e-7)
        return loss


def cox_ph_loss(log_h: Tensor, durations: Tensor, events: Tensor, lambda_wass: float = 0.1, eps: float = 1e-7) -> Tensor:
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.

    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    idx = durations.sort(descending=True)[1]
    events = events[idx]
    log_h = log_h[idx]
    return cox_ph_loss_sorted(log_h, events, durations,  eps=eps)


# def cox_ph_loss_sorted(log_h: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
#     """Requires the input to be sorted by descending duration time.
#     See DatasetDurationSorted.
#
#     We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
#     where h = exp(log_h) are the hazards and R is the risk set, and d is event.
#
#     We just compute a cumulative sum, and not the true Risk sets. This is a
#     limitation, but simple and fast.
#     """
#     if events.dtype is torch.bool:
#         events = events.float()
#     events = events.view(-1)
#     log_h = log_h.view(-1)
#     gamma = log_h.max()
#     log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
#     return - log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum())


def empirical_survival(durations, events):
    kmf = KaplanMeierFitter()
    kmf.fit(durations.numpy(), events.numpy())
    return torch.tensor(kmf.survival_function_.values.flatten(), dtype=torch.float32)


def baseline(log_h, df_target, max_duration, batch_size, eval_=True, num_workers=0):
    if max_duration is None:
        max_duration = np.inf

    duration_col = 'duration'
    event_col = 'event'
    # Here we are computing when expg when there are no events.
    # Could be made faster, by only computing when there are events.
    return (df_target
            .assign(expg=np.exp(log_h))
            .groupby(duration_col)
            .agg({'expg': 'sum', event_col: 'sum'})
            .sort_index(ascending=False)
            .assign(expg=lambda x: x['expg'].cumsum())
            .pipe(lambda x: x[event_col]/x['expg'])
            .fillna(0.)
            .iloc[::-1]
            .loc[lambda x: x.index <= max_duration]
            .rename('baseline_hazards'))


def the_baseline_hazard(df_target, durations_col='duration', events_col='event'):
    # Sort the data by duration (time)
    df_target_sorted = df_target.sort_values(by=durations_col)

    # Initialize the cumulative hazard (we'll compute this directly)
    df_target_sorted['cumulative_hazard'] = 0.0

    # Calculate the risk set at each time point
    # Risk set: All individuals still at risk at that time point
    for time in df_target_sorted[durations_col].unique():
        # Find the risk set at this time point (those who haven't yet experienced an event or been censored)
        risk_set = df_target_sorted[df_target_sorted[durations_col] >= time]

        # Calculate the number of events at this time point
        events_at_time = risk_set[events_col].sum()

        # Calculate the baseline hazard at this time
        baseline_hazard_at_time = events_at_time / len(risk_set)

        # Update the cumulative hazard
        df_target_sorted.loc[df_target_sorted[durations_col] ==
                             time, 'cumulative_hazard'] = baseline_hazard_at_time

    # Return the baseline hazards
    return df_target_sorted['cumulative_hazard']


def baseline(log_h, df_target, max_duration, batch_size, eval_=True, num_workers=0):
    if max_duration is None:
        max_duration = np.inf

    duration_col = 'duration'
    event_col = 'event'

    # Convert log_h to a tensor if it's not already
    log_h = torch.tensor(log_h, dtype=torch.float32)

    # Here we are computing when expg when there are no events.
    # Could be made faster, by only computing when there are events.
    df_target['expg'] = torch.exp(log_h)  # Keep everything in PyTorch

    # Group by duration and compute cumulative sum of 'expg' and the sum of 'event'
    grouped = df_target.groupby(duration_col).agg(
        {'expg': 'sum', event_col: 'sum'}).sort_index(ascending=False)

    # Cumulative sum of 'expg' and compute the baseline hazard
    grouped['expg_cumsum'] = grouped['expg'].cumsum()

    # Compute the baseline hazards
    grouped['baseline_hazards'] = grouped[event_col] / \
        grouped['expg_cumsum'].replace(0., np.nan)

    # Handle NaNs and reverse order for output
    grouped = grouped.fillna(0.).iloc[::-1]

    # Return the baseline hazards as a PyTorch tensor
    baseline_hazards = torch.tensor(
        grouped['baseline_hazards'].values.copy(), dtype=torch.float32)

    return baseline_hazards


def cox_ph_loss_sorted(log_h: torch.Tensor, events: torch.Tensor, durations: torch.Tensor, lambda_wass: float = 0.1, eps: float = 1e-7) -> torch.Tensor:
    """Simple Mean Squared Error (MSE) loss function."""
    if events.dtype is torch.bool:
        events = events.float()

    events = events.view(-1)
    log_h = log_h.view(-1)

    sorted_durations, sorted_indices = torch.sort(durations)
    sorted_events = events[sorted_indices]

    pred_surv = torch.exp(log_h)  # Simplified hazard to survival conversion

    df = pd.DataFrame({'duration': durations, 'event': events})
    base = baseline(log_h.detach().numpy(), df, None, 10)

    repeated_base = []
    previous = -np.inf
    pointer = 0
    base = base.detach().numpy()
    for i in range(len(sorted_durations)):

        if previous == sorted_durations[i]:
            pointer = pointer - 1

        repeated_base.append(base[pointer])
        previous = sorted_durations[i]
        pointer = pointer+1
    repeated_base_tensor = torch.tensor(repeated_base, dtype=torch.float32)

    pred_surv_tensor = repeated_base_tensor * pred_surv
    mse = torch.mean((pred_surv_tensor - sorted_events) ** 2)

    return mse


class Custom_CoxPH(tt.Model):
    """Cox proportional hazards model parameterized with a neural net.
    This is essentially the DeepSurv method [1].

    The loss function is not quite the partial log-likelihood, but close.
    The difference is that for tied events, we use a random order instead of
    including all individuals that had an event at that point in time.

    Arguments:
        net {torch.nn.Module} -- A pytorch net.

    Keyword Arguments:
        optimizer {torch or torchtuples optimizer} -- Optimizer (default: {None})
        device {str, int, torch.device} -- Device to compute on. (default: {None})
            Preferably pass a torch.device object.
            If 'None': use default gpu if available, else use cpu.
            If 'int': used that gpu: torch.device('cuda:<device>').
            If 'string': string is passed to torch.device('string').

    [1] Jared L. Katzman, Uri Shaham, Alexander Cloninger, Jonathan Bates, Tingting Jiang, and Yuval Kluger.
        Deepsurv: personalized treatment recommender system using a Cox proportional hazards deep neural network.
        BMC Medical Research Methodology, 18(1), 2018.
        https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-018-0482-1
    """

    duration_col = 'duration'
    event_col = 'event'

    def __init__(self, net, optimizer=None, device=None, loss=None):
        print("Custom_CoxPH")

        if loss is None:
            loss = CoxPHLoss()
        super().__init__(net, loss, optimizer, device)

    def _compute_baseline_hazards(self, input, df_target, max_duration, batch_size, eval_=True, num_workers=0):
        if max_duration is None:
            max_duration = np.inf

        # Here we are computing when expg when there are no events.
        #   Could be made faster, by only computing when there are events.
        return (df_target
                .assign(expg=np.exp(self.predict(input, batch_size, True, eval_, num_workers=num_workers)))
                .groupby(self.duration_col)
                .agg({'expg': 'sum', self.event_col: 'sum'})
                .sort_index(ascending=False)
                .assign(expg=lambda x: x['expg'].cumsum())
                .pipe(lambda x: x[self.event_col]/x['expg'])
                .fillna(0.)
                .iloc[::-1]
                .loc[lambda x: x.index <= max_duration]
                .rename('baseline_hazards'))

    def _predict_cumulative_hazards(self, input, max_duration, batch_size, verbose, baseline_hazards_,
                                    eval_=True, num_workers=0):
        max_duration = np.inf if max_duration is None else max_duration
        if baseline_hazards_ is self.baseline_hazards_:
            bch = self.baseline_cumulative_hazards_
        else:
            bch = self.compute_baseline_cumulative_hazards(set_hazards=False,
                                                           baseline_hazards_=baseline_hazards_)
        bch = bch.loc[lambda x: x.index <= max_duration]
        expg = np.exp(self.predict(input, batch_size, True, eval_,
                                   num_workers=num_workers)).reshape(1, -1)
        return pd.DataFrame(bch.values.reshape(-1, 1).dot(expg),
                            index=bch.index)

    def partial_log_likelihood(self, input, target, g_preds=None, batch_size=8224, eps=1e-7, eval_=True,
                               num_workers=0):
        '''Calculate the partial log-likelihood for the events in datafram df.
        This likelihood does not sample the controls.
        Note that censored data (non events) does not have a partial log-likelihood.

        Arguments:
            input {tuple, np.ndarray, or torch.tensor} -- Input to net.
            target {tuple, np.ndarray, or torch.tensor} -- Target labels.

        Keyword Arguments:
            g_preds {np.array} -- Predictions from `model.predict` (default: {None})
            batch_size {int} -- Batch size (default: {8224})
            eval_ {bool} -- If 'True', use 'eval' mode on net. (default: {True})
            num_workers {int} -- Number of workers in created dataloader (default: {0})

        Returns:
            Partial log-likelihood.
        '''
        df = self.target_to_df(target)
        if g_preds is None:
            g_preds = self.predict(
                input, batch_size, True, eval_, num_workers=num_workers)
        return (df
                .assign(_g_preds=g_preds)
                .sort_values(self.duration_col, ascending=False)
                .assign(_cum_exp_g=(lambda x: x['_g_preds']
                                    .pipe(np.exp)
                                    .cumsum()
                                    .groupby(x[self.duration_col])
                                    .transform('max')))
                .loc[lambda x: x[self.event_col] == 1]
                .assign(pll=lambda x: x['_g_preds'] - np.log(x['_cum_exp_g'] + eps))
                ['pll'])

    def fit(self, input, target, batch_size=256, epochs=1, callbacks=None, verbose=True,
            num_workers=0, shuffle=True, metrics=None, val_data=None, val_batch_size=8224,
            **kwargs):
        """Fit  model with inputs and targets. Where 'input' is the covariates, and
        'target' is a tuple with (durations, events).

        Arguments:
            input {np.array, tensor or tuple} -- Input x passed to net.
            target {np.array, tensor or tuple} -- Target [durations, events].

        Keyword Arguments:
            batch_size {int} -- Elements in each batch (default: {256})
            epochs {int} -- Number of epochs (default: {1})
            callbacks {list} -- list of callbacks (default: {None})
            verbose {bool} -- Print progress (default: {True})
            num_workers {int} -- Number of workers used in the dataloader (default: {0})
            shuffle {bool} -- If we should shuffle the order of the dataset (default: {True})
            **kwargs are passed to 'make_dataloader' method.

        Returns:
            TrainingLogger -- Training log
        """
        self.training_data = tt.tuplefy(input, target)
        return super().fit(input, target, batch_size, epochs, callbacks, verbose,
                           num_workers, shuffle, metrics, val_data, val_batch_size,
                           **kwargs)

    def target_to_df(self, target):
        durations, events = tt.tuplefy(target).to_numpy()
        df = pd.DataFrame(
            {self.duration_col: durations, self.event_col: events})
        return df

    def compute_baseline_hazards(self, input=None, target=None, max_duration=None, sample=None, batch_size=8224,
                                 set_hazards=True, eval_=True, num_workers=0):
        """Computes the Breslow estimates form the data defined by `input` and `target`
        (if `None` use training data).

        Typically call
        model.compute_baseline_hazards() after fitting.

        Keyword Arguments:
            input  -- Input data (train input) (default: {None})
            target  -- Target data (train target) (default: {None})
            max_duration {float} -- Don't compute estimates for duration higher (default: {None})
            sample {float or int} -- Compute estimates of subsample of data (default: {None})
            batch_size {int} -- Batch size (default: {8224})
            set_hazards {bool} -- Set hazards in model object, or just return hazards. (default: {True})

        Returns:
            pd.Series -- Pandas series with baseline hazards. Index is duration_col.
        """
        if (input is None) and (target is None):
            if not hasattr(self, 'training_data'):
                raise ValueError(
                    "Need to give a 'input' and 'target' to this function.")
            input, target = self.training_data
        df = self.target_to_df(target)  # .sort_values(self.duration_col)
        if sample is not None:
            if sample >= 1:
                df = df.sample(n=sample)
            else:
                df = df.sample(frac=sample)
        input = tt.tuplefy(input).to_numpy().iloc[df.index.values]
        base_haz = self._compute_baseline_hazards(input, df, max_duration, batch_size,
                                                  eval_=eval_, num_workers=num_workers)
        if set_hazards:
            self.compute_baseline_cumulative_hazards(
                set_hazards=True, baseline_hazards_=base_haz)
        return base_haz

    def compute_baseline_cumulative_hazards(self, input=None, target=None, max_duration=None, sample=None,
                                            batch_size=8224, set_hazards=True, baseline_hazards_=None,
                                            eval_=True, num_workers=0):
        """See `compute_baseline_hazards. This is the cumulative version."""
        if ((input is not None) or (target is not None)) and (baseline_hazards_ is not None):
            raise ValueError(
                "'input', 'target' and 'baseline_hazards_' can not both be different from 'None'.")
        if baseline_hazards_ is None:
            baseline_hazards_ = self.compute_baseline_hazards(input, target, max_duration, sample, batch_size,
                                                              set_hazards=False, eval_=eval_, num_workers=num_workers)
        assert baseline_hazards_.index.is_monotonic_increasing, \
            'Need index of baseline_hazards_ to be monotonic increasing, as it represents time.'
        bch = (baseline_hazards_
               .cumsum()
               .rename('baseline_cumulative_hazards'))
        if set_hazards:
            self.baseline_hazards_ = baseline_hazards_
            self.baseline_cumulative_hazards_ = bch
        return bch

    def predict_cumulative_hazards(self, input, max_duration=None, batch_size=8224, verbose=False,
                                   baseline_hazards_=None, eval_=True, num_workers=0):
        """See `predict_survival_function`."""
        if type(input) is pd.DataFrame:
            input = self.df_to_input(input)
        if baseline_hazards_ is None:
            if not hasattr(self, 'baseline_hazards_'):
                raise ValueError(
                    'Need to compute baseline_hazards_. E.g run `model.compute_baseline_hazards()`')
            baseline_hazards_ = self.baseline_hazards_
        assert baseline_hazards_.index.is_monotonic_increasing, \
            'Need index of baseline_hazards_ to be monotonic increasing, as it represents time.'
        return self._predict_cumulative_hazards(input, max_duration, batch_size, verbose, baseline_hazards_,
                                                eval_, num_workers=num_workers)

    def predict_surv_df(self, input, max_duration=None, batch_size=8224, verbose=False, baseline_hazards_=None,
                        eval_=True, num_workers=0):
        """Predict survival function for `input`. S(x, t) = exp(-H(x, t))
        Require computed baseline hazards.

        Arguments:
            input {np.array, tensor or tuple} -- Input x passed to net.

        Keyword Arguments:
            max_duration {float} -- Don't compute estimates for duration higher (default: {None})
            batch_size {int} -- Batch size (default: {8224})
            baseline_hazards_ {pd.Series} -- Baseline hazards. If `None` used `model.baseline_hazards_` (default: {None})
            eval_ {bool} -- If 'True', use 'eval' mode on net. (default: {True})
            num_workers {int} -- Number of workers in created dataloader (default: {0})

        Returns:
            pd.DataFrame -- Survival estimates. One columns for each individual.
        """
        return np.exp(-self.predict_cumulative_hazards(input, max_duration, batch_size, verbose, baseline_hazards_,
                                                       eval_, num_workers))

    def predict_surv(self, input, max_duration=None, batch_size=8224, numpy=None, verbose=False,
                     baseline_hazards_=None, eval_=True, num_workers=0):
        """Predict survival function for `input`. S(x, t) = exp(-H(x, t))
        Require compueted baseline hazards.

        Arguments:
            input {np.array, tensor or tuple} -- Input x passed to net.

        Keyword Arguments:
            max_duration {float} -- Don't compute estimates for duration higher (default: {None})
            batch_size {int} -- Batch size (default: {8224})
            numpy {bool} -- 'False' gives tensor, 'True' gives numpy, and None give same as input
                (default: {None})
            baseline_hazards_ {pd.Series} -- Baseline hazards. If `None` used `model.baseline_hazards_` (default: {None})
            eval_ {bool} -- If 'True', use 'eval' mode on net. (default: {True})
            num_workers {int} -- Number of workers in created dataloader (default: {0})

        Returns:
            pd.DataFrame -- Survival estimates. One columns for each individual.
        """
        surv = self.predict_surv_df(input, max_duration, batch_size, verbose, baseline_hazards_,
                                    eval_, num_workers)
        surv = torch.from_numpy(surv.values.transpose())
        return tt.utils.array_or_tensor(surv, numpy, input)

    def save_net(self, path, **kwargs):
        """Save self.net and baseline hazards to file.

        Arguments:
            path {str} -- Path to file.
            **kwargs are passed to torch.save

        Returns:
            None
        """
        path, extension = os.path.splitext(path)
        if extension == "":
            extension = '.pt'
        super().save_net(path+extension, **kwargs)
        if hasattr(self, 'baseline_hazards_'):
            self.baseline_hazards_.to_pickle(path+'_blh.pickle')

    def load_net(self, path, **kwargs):
        """Load net and hazards from file.

        Arguments:
            path {str} -- Path to file.
            **kwargs are passed to torch.load

        Returns:
            None
        """
        path, extension = os.path.splitext(path)
        if extension == "":
            extension = '.pt'
        super().load_net(path+extension, **kwargs)
        blh_path = path+'_blh.pickle'
        if os.path.isfile(blh_path):
            self.baseline_hazards_ = pd.read_pickle(blh_path)
            self.baseline_cumulative_hazards_ = self.baseline_hazards_.cumsum()

    def df_to_input(self, df):
        input = df[self.input_cols].values
        return input


# create the network
class Block(nn.Module):
    def __init__(self, in_features, out_features, bias=True, batch_norm=True, dropout=0., activation=nn.ReLU,
                 w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        if w_init_:
            w_init_(self.linear.weight.data)
        self.activation = activation()
        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else None
        if (batch_norm == None):
            print('no batch')

        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, input):
        input = self.activation(self.linear(input))
        if self.batch_norm:
            input = self.batch_norm(input)
        if self.dropout:
            input = self.dropout(input)
        return input


class MLP(nn.Module):
    def __init__(self, in_features, num_nodes, out_features, batch_norm=True, dropout=None, activation=nn.ReLU,
                 output_activation=None, output_bias=True,
                 w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        num_nodes = tuplefy(in_features, num_nodes).flatten()
        if not hasattr(dropout, '__iter__'):
            dropout = [dropout for _ in range(len(num_nodes)-1)]
        net = []
        for n_in, n_out, p in zip(num_nodes[:-1], num_nodes[1:], dropout):
            net.append(Block(n_in, n_out, True,
                       batch_norm, p, activation, w_init_))
        net.append(nn.Linear(num_nodes[-1], out_features, output_bias))
        if output_activation:
            net.append(output_activation)
        self.net = nn.Sequential(*net)

    def forward(self, input):
        return self.net(input)
