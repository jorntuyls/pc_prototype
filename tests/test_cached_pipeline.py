
# Based on sklearn test cases:
#      https://github.com/scikit-learn/scikit-learn/blob/1e334657a6921a266d30f24dbb3cc1ca22b0f6c0/sklearn/tests/test_pipeline.py

import numpy as np
import time
import tempfile

from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_array_equal

from pc_smac.pc_smac.pipeline.cached_pipeline import CachedPipeline

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.externals.joblib import Memory

class NoFit(object):
    """Small class to test parameter dispatching.
    """

    def __init__(self, a=None, b=None):
        self.a = a
        self.b = b


class NoTrans(NoFit):

    def fit(self, X, y):
        return self

    def get_params(self, deep=False):
        return {'a': self.a, 'b': self.b}

    def set_params(self, **params):
        self.a = params['a']
        return self

class NoInvTransf(NoTrans):
    def transform(self, X, y=None):
        return X


class Transf(NoInvTransf):
    def transform(self, X, y=None):
        return X

    def inverse_transform(self, X):
        return X

class DummyTransf(Transf):
    """Transformer which store the column means"""

    def fit(self, X, y):
        self.means_ = np.mean(X, axis=0)
        # Store a timestamp such that we know
        # that we have a cache object
        self.timestamp = time.time()
        return self


def test_cached_pipeline():
    # Test the various methods of the pipeline (pca + svm).
    dr = "/Users/jorntuyls/Documents/workspaces/thesis/data/"
    cachedir = tempfile.mkdtemp(dir=dr, prefix="testcache_")

    iris = load_iris()
    X = iris.data
    y = iris.target
    # Create a Memory object
    memory = Memory(cachedir=cachedir, verbose=10)
    # Test with Transformer + SVC
    clf = SVC(probability=True, random_state=0)
    transf = DummyTransf()
    pipe = Pipeline([('transf', clone(transf)), ('svc', (clf))])
    cached_pipe = CachedPipeline([('transf', transf), ('svc', clf)],
                                 memory=memory)

    # Memoize the transformer at the first fit
    cached_pipe.fit(X, y)
    pipe.fit(X, y)
    # Get the time stamp of the tranformer in the cached pipeline
    ts = cached_pipe.named_steps['transf'].timestamp
    # Check if the results are similar
    assert_array_equal(pipe.predict(X), cached_pipe.predict(X))
    assert_array_equal(pipe.predict_proba(X), cached_pipe.predict_proba(X))
    assert_array_equal(pipe.predict_log_proba(X),
                       cached_pipe.predict_log_proba(X))
    assert_array_equal(pipe.score(X, y), cached_pipe.score(X, y))
    assert_array_equal(pipe.named_steps['transf'].means_,
                       cached_pipe.named_steps['transf'].means_)

    # Check that we are reading the cache while fitting
    # a second time
    cached_pipe.fit(X, y)
    # Check if the results are similar
    assert_array_equal(pipe.predict(X), cached_pipe.predict(X))
    assert_array_equal(pipe.predict_proba(X), cached_pipe.predict_proba(X))
    assert_array_equal(pipe.predict_log_proba(X),
                       cached_pipe.predict_log_proba(X))
    assert_array_equal(pipe.score(X, y), cached_pipe.score(X, y))
    assert_array_equal(pipe.named_steps['transf'].means_,
                       cached_pipe.named_steps['transf'].means_)
    assert_equal(ts, cached_pipe.named_steps['transf'].timestamp)

    # Create a new pipeline with cloned estimators
    # Check that we are reading the cache
    clf_2 = SVC(probability=True, random_state=0)
    transf_2 = DummyTransf()
    cached_pipe_2 = CachedPipeline([('transf', transf_2), ('svc', clf_2)],
                                   memory=memory)
    cached_pipe_2.fit(X, y)
    print("test")

    # Check if the results are similar
    assert_array_equal(pipe.predict(X), cached_pipe_2.predict(X))
    assert_array_equal(pipe.predict_proba(X), cached_pipe_2.predict_proba(X))
    assert_array_equal(pipe.predict_log_proba(X),
                       cached_pipe_2.predict_log_proba(X))
    assert_array_equal(pipe.score(X, y), cached_pipe_2.score(X, y))
    assert_array_equal(pipe.named_steps['transf'].means_,
                       cached_pipe_2.named_steps['transf'].means_)
    assert_equal(ts, cached_pipe_2.named_steps['transf'].timestamp)



if __name__ == "__main__":
    test_cached_pipeline()
