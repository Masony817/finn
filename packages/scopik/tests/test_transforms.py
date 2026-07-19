import math

import numpy as np
import pytest
from scopik.datamodel import ScopikError
from scopik.transforms import apply_transform, resolve_transform


def test_named_transforms():
    assert resolve_transform("rev_to_rad") == (2.0 * math.pi, 0.0)
    assert resolve_transform("us_to_s") == (1e-6, 0.0)
    assert resolve_transform("negate") == (-1.0, 0.0)
    assert resolve_transform(None) == (1.0, 0.0)


def test_mapping_transform():
    assert resolve_transform({"scale": 2.0, "offset": 1.0}) == (2.0, 1.0)


def test_chained_transforms_compose_in_order():
    # y = ((x * 2 + 1) * -1 + 0) = -2x - 1
    scale, offset = resolve_transform([{"scale": 2.0, "offset": 1.0}, "negate"])
    assert (scale, offset) == (-2.0, -1.0)


def test_apply_transform():
    values = np.array([0.0, 1.0, 2.0])
    out = apply_transform(values, {"scale": 3.0, "offset": -1.0})
    assert np.allclose(out, [-1.0, 2.0, 5.0])


def test_unknown_transform_raises():
    with pytest.raises(ScopikError, match="unknown transform"):
        resolve_transform("furlongs_to_rad")


def test_bad_mapping_keys_raise():
    with pytest.raises(ScopikError, match="unknown keys"):
        resolve_transform({"scale": 1.0, "gain": 2.0})
