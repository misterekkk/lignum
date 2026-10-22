"""Every Python test runs three times: with the binned kernel (default), with the direct traversal
and, on POSIX, with the binned model compiled to native code (experimental compiled mode)."""

import functools
import sys

import pytest

import lignum

KERNELS = [True, False] + ([] if sys.platform == "win32" else ["compiled"])


@pytest.fixture(scope="session")
def compiled_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("compiled")


@pytest.fixture(autouse=True, params=KERNELS, ids=["binned", "direct", "compiled"][:len(KERNELS)])
def kernel(request, monkeypatch):
    original = lignum.load
    if request.param != "compiled":
        monkeypatch.setattr(lignum, "load", functools.partial(original, binned=request.param))
        return request.param

    # Small blocks so models of a few trees still span several blocks, one of them partial.
    work_dir = request.getfixturevalue("compiled_dir")

    def load_compiled(*args, **kwargs):
        model = original(*args, **kwargs)
        if model.uses_binned:
            model.compile(work_dir=str(work_dir), jobs=2, block_trees=3)
        return model

    monkeypatch.setattr(lignum, "load", load_compiled)
    return request.param
