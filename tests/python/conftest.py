"""Every Python test runs three times: with the binned kernel (default), with the direct traversal
and with the binned model compiled to native code (experimental compiled mode)."""

import functools

import pytest

import lignum

KERNELS = [True, False, "compiled"]


@pytest.fixture(scope="session", autouse=True)
def compiled_cache(tmp_path_factory):
    # Compiled libraries go to a cache of this test session, not the user's cache.
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LIGNUM_CACHE_DIR", str(tmp_path_factory.mktemp("lignum_cache")))
        yield


@pytest.fixture(autouse=True, params=KERNELS, ids=["binned", "direct", "compiled"])
def kernel(request, monkeypatch):
    original = lignum.load
    if request.param != "compiled":
        monkeypatch.setattr(lignum, "load", functools.partial(original, binned=request.param))
        return request.param

    # Small blocks so models of a few trees still span several blocks, one of them partial.
    def load_compiled(*args, **kwargs):
        model = original(*args, **kwargs)
        if model.uses_binned:
            model.compile(jobs=2, block_trees=3)
        return model

    monkeypatch.setattr(lignum, "load", load_compiled)
    return request.param
