# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from lmcache.v1.memory_management import (
    TensorMemoryObj,
    MemoryObjMetadata,
    MemoryFormat,
)
from lmcache.v1 import specagent_bulk_view as m


class Alloc:
    def __init__(self):
        self.freed = []

    def free(self, obj, allocator_type=None):
        self.freed.append(obj)
        obj.invalidate()


def owner(tokens=17):
    a = Alloc()
    t = (
        torch.arange(2 * 4 * tokens * 8, dtype=torch.float32)
        .reshape(2, 4, tokens, 8)
        .to(torch.bfloat16)
    )
    meta = MemoryObjMetadata(
        shape=t.shape,
        dtype=t.dtype,
        address=t.data_ptr(),
        phy_size=t.numel() * 2,
        ref_count=1,
        fmt=MemoryFormat.KV_2LTD,
    )
    return a, t, TensorMemoryObj(t.view(torch.uint8).flatten(), meta, a)


@pytest.mark.parametrize("tokens", [17, 256])
def test_strided_view_is_exact_and_survives_owner_eviction(tokens):
    a, t, o = owner(tokens)
    v = m.BulkLayerView(o, 2)
    assert v.tensor.stride() == (8, 4 * tokens * 8, 1)
    assert not v.tensor.is_contiguous()
    assert torch.equal(v.tensor, t[:, 2].permute(1, 0, 2))
    o.ref_count_down()
    assert not a.freed and v.is_valid()
    v.ref_count_down()
    assert len(a.freed) == 1 and not o.is_valid() and v.tensor is None


def test_independent_layer_leases_release_parent_once():
    a, t, o = owner()
    views = [m.BulkLayerView(o, i) for i in range(4)]
    o.ref_count_down()
    for v in views[:-1]:
        v.ref_count_down()
        assert not a.freed
    views[-1].ref_count_up()
    views[-1].ref_count_down()
    assert not a.freed
    views[-1].ref_count_down()
    assert len(a.freed) == 1


def test_bad_layer_does_not_acquire_parent_reference():
    a, t, o = owner()
    with pytest.raises(IndexError):
        m.BulkLayerView(o, 4)
    assert o.metadata.ref_count == 1
    o.ref_count_down()
    assert len(a.freed) == 1


def test_batch_acquisition_rolls_back_prior_leases_on_bad_owner():
    a, t, good = owner()
    b, u, bad = owner()
    bad.metadata.fmt = MemoryFormat.KV_T2D
    with pytest.raises(ValueError):
        m.acquire_bulk_layer_views([good, bad], 4)
    assert good.metadata.ref_count == bad.metadata.ref_count == 1
    good.ref_count_down()
    bad.ref_count_down()
    assert len(a.freed) == len(b.freed) == 1


def test_prefix_merge_preserves_new_suffix_all_layers():
    a, src, old = owner(17)
    b, dst, new = owner(32)
    expected = dst.clone()
    expected[:, :, :17, :] = src
    m.merge_native_bulk_prefix(old, new, 17)
    assert torch.equal(new.tensor, expected)
    old.ref_count_down()
    new.ref_count_down()


def test_prefix_merge_rejects_incomplete_source():
    a, src, old = owner(17)
    b, dst, new = owner(32)
    expected = dst.clone()
    with pytest.raises(ValueError):
        m.merge_native_bulk_prefix(old, new, 16)
    assert torch.equal(new.tensor, expected)
    old.ref_count_down()
    new.ref_count_down()


def test_cached_tensor_views_do_not_pin_evicted_owner():
    a, t, o = owner()
    first = m.acquire_bulk_layer_views([o], 4)
    for layer in first:
        layer[0].ref_count_down()
    assert o.metadata.ref_count == 1
    o.tensor.add_(2)
    second = m.acquire_bulk_layer_views([o], 4)
    for index, layer in enumerate(second):
        assert torch.equal(layer[0].tensor, o.tensor[:, index].permute(1, 0, 2))
    o.ref_count_down()
    assert not a.freed
    for layer in second:
        layer[0].ref_count_down()
    assert len(a.freed) == 1 and not o.is_valid()


def test_cached_tensor_views_follow_owner_tensor_replacement():
    a, t, o = owner()
    old = m.BulkLayerView(o, 1)
    old.ref_count_down()
    replacement = torch.full_like(o.raw_data, 0)
    o.raw_data = replacement
    fresh = m.BulkLayerView(o, 1)
    assert torch.count_nonzero(fresh.tensor) == 0
    fresh.ref_count_down()
    o.ref_count_down()
    assert len(a.freed) == 1


def test_concurrent_batches_have_independent_leases():
    a, t, o = owner()
    first = m.acquire_bulk_layer_views([o], 4)
    second = m.acquire_bulk_layer_views([o], 4)
    assert all(x[0] is not y[0] for x, y in zip(first, second, strict=True))
    o.ref_count_down()
    for layer in first:
        layer[0].ref_count_down()
        assert layer[0].tensor is None
    assert not a.freed
    for index, layer in enumerate(second):
        assert torch.equal(layer[0].tensor, t[:, index].permute(1, 0, 2))
        layer[0].ref_count_down()
    assert len(a.freed) == 1


def test_incomplete_layer_count_leaves_owner_reference_unchanged():
    a, t, o = owner()
    with pytest.raises(ValueError, match="incomplete layer count"):
        m.acquire_bulk_layer_views([o], 3)
    assert o.get_ref_count() == 1
    o.ref_count_down()
    assert len(a.freed) == 1
