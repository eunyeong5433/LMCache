# SPDX-License-Identifier: Apache-2.0
"""SPECAGENT_LAZY_BULK_VIEWS: lazily built borrowed descriptors == eager ones."""
# Third Party
import pytest
import torch

# First Party
from lmcache.v1 import specagent_bulk_view as m
from tests.v1.test_specagent_bulk_view import owner


def _sig(d):
    return (d.tensor.data_ptr(), tuple(d.tensor.shape), tuple(d.tensor.stride()),
            d.metadata.address, tuple(d.metadata.shape), d.metadata.fmt)


@pytest.mark.parametrize("tokens", [17, 256])
def test_lazy_columns_match_eager(tokens, monkeypatch):
    owners_e = [owner(tokens)[2] for _ in range(3)]
    owners_l = [owner(tokens)[2] for _ in range(3)]
    eager = m.borrow_bulk_layer_views(owners_e, 4)
    monkeypatch.setenv("SPECAGENT_LAZY_BULK_VIEWS", "1")
    monkeypatch.setenv("SPECAGENT_LAZY_BULK_VIEWS_CHECK", "1")
    lazy = m.borrow_bulk_layer_views(owners_l, 4)
    assert len(lazy) == len(eager) == 4
    for ce, cl, oe, ol in zip(eager, lazy, [owners_e] * 4, [owners_l] * 4):
        assert len(ce) == len(cl)
        for de, dl, a, b in zip(ce, cl, oe, ol):
            # same layout relative to each owner's base
            assert _sig(de)[1:3] == _sig(dl)[1:3]
            assert (de.tensor.data_ptr() - a.tensor.data_ptr()
                    == dl.tensor.data_ptr() - b.tensor.data_ptr())
            assert torch.equal(de.tensor, dl.tensor)
    # reused on the next borrow, same objects each access
    again = m.borrow_bulk_layer_views(owners_l, 4)
    assert all(again[l][i] is lazy[l][i] for l in range(4) for i in range(3))


def test_lazy_bad_layer_count(monkeypatch):
    monkeypatch.setenv("SPECAGENT_LAZY_BULK_VIEWS", "1")
    with pytest.raises(ValueError):
        m.borrow_bulk_layer_views([owner()[2]], 5)


def test_lazy_invalid_owner(monkeypatch):
    monkeypatch.setenv("SPECAGENT_LAZY_BULK_VIEWS", "1")
    o = owner()[2]
    o.ref_count_down()
    with pytest.raises(ValueError):
        m.borrow_bulk_layer_views([o], 4)
