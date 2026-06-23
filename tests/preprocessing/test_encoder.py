import numpy as np
import pytest

from recsys.preprocessing.encoder import IdEncoder


def test_encoder_contiguous_and_roundtrip():
    enc = IdEncoder().fit([10, 20, 20, 30])
    assert enc.n_ids == 3
    codes = enc.transform([10, 30, 20])
    assert set(np.unique(enc.transform([10, 20, 30]))) == {0, 1, 2}
    assert list(enc.inverse_transform(codes)) == [10, 30, 20]


def test_encoder_raises_on_unseen_id():
    enc = IdEncoder().fit([10, 20])
    with pytest.raises(KeyError):
        enc.transform([99])
