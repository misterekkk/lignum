"""Every Python test runs twice: with the binned kernel (default) and with the direct traversal."""

import functools

import pytest

import lignum


@pytest.fixture(autouse=True, params=[True, False], ids=["binned", "direct"])
def kernel(request, monkeypatch):
    original = lignum.load
    monkeypatch.setattr(lignum, "load", functools.partial(original, binned=request.param))
    return request.param
