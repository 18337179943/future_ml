"""Utilities that are required by gplearn.

Most of these functions are slightly modified versions of some key utility
functions from scikit-learn that gplearn depends upon. They reside here in
order to maintain compatibility across different versions of scikit-learn.

"""

import numbers

import numpy as np
from joblib import cpu_count


class NotFittedError(ValueError, AttributeError):

    """Exception class to raise if estimator is used before fitting.

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.

    Examples
    --------
    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.exceptions import NotFittedError
    >>> try:
    ...     LinearSVC().predict([[1, 2], [2, 3], [3, 4]])
    ... except NotFittedError as e:
    ...     print(repr(e))
    ...                        # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    NotFittedError('This LinearSVC instance is not fitted yet',)
    .. versionchanged:: 0.18
       Moved from sklearn.utils.validation.

    """


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def _get_n_jobs(n_jobs):
    """Get number of jobs for the computation.

    This function reimplements the logic of joblib to determine the actual
    number of jobs depending on the cpu count. If -1 all CPUs are used.
    If 1 is given, no parallel computing code is used at all, which is useful
    for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
    Thus for n_jobs = -2, all CPUs but one are used.

    Parameters
    ----------
    n_jobs : int
        Number of jobs stated in joblib convention.

    Returns
    -------
    n_jobs : int
        The actual number of jobs as positive integer.

    Examples
    --------
    >>> from sklearn.utils import _get_n_jobs
    >>> _get_n_jobs(4)
    4
    >>> jobs = _get_n_jobs(-2)
    >>> assert jobs == max(cpu_count() - 1, 1)
    >>> _get_n_jobs(0)
    Traceback (most recent call last):
    ...
    ValueError: Parameter n_jobs == 0 has no meaning.

    """
    if n_jobs < 0:
        return max(cpu_count() + 1 + n_jobs, 1)
    elif n_jobs == 0:
        raise ValueError('Parameter n_jobs == 0 has no meaning.')
    else:
        return n_jobs

def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(_get_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = (n_estimators // n_jobs) * np.ones(n_jobs,
                                                              dtype=np.int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()


def winsorize(factor_input):
    # WINSORIZE 因子处理：中位数去极值
    factor_output = factor_input.copy()
    tmp_m = factor_input.median()  # 中位数
    tmp_m1 = (factor_input - factor_input.median()).abs().median()  # 与中位数之差的绝对值向量的中位数

    # 当数据质量太差时，可能出现大部分因子值都是零的情形，取中位数都是零，这种情形下需要把零先忽略掉
    if tmp_m1 == 0:
        tmp_is0 = factor_input == 0
        factor_input[tmp_is0] = np.nan  # 先把非0的部分去极值
        tmp_m = factor_input.median()
        tmp_m1 = (factor_input - factor_input.median()).abs().median()
        factor_output[factor_input > tmp_m + 5 * tmp_m1] = tmp_m + 5 * tmp_m1
        factor_output[factor_input < tmp_m - 5 * tmp_m1] = tmp_m - 5 * tmp_m1  # 中位数去极值
        factor_output[tmp_is0] = 0  # 最后把以前是0的部分恢复成0
    else:
        factor_output[factor_input > tmp_m + 5 * tmp_m1] = tmp_m + 5 * tmp_m1
        factor_output[factor_input < tmp_m - 5 * tmp_m1] = tmp_m - 5 * tmp_m1  # 中位数去极值

    return factor_output


def standardize(factor_input):
    # STANDARDIZE 因子处理：标准化
    factor_output = (factor_input - factor_input.mean())/factor_input.std()
    return factor_output
