"""Testes da política de early stopping (usada no treino da rede neural)."""

from __future__ import annotations

from recsys.models.early_stopping import EarlyStopping


def test_does_not_stop_while_loss_improves() -> None:
    es = EarlyStopping(patience=2)
    losses = [1.0, 0.9, 0.8, 0.7, 0.6]
    assert not any(es.update(loss) for loss in losses)


def test_stops_after_patience_epochs_without_improvement() -> None:
    es = EarlyStopping(patience=2)
    # 1 melhora e depois estagna: para no 2º epoch sem melhora.
    assert es.update(1.0) is False
    assert es.update(0.9) is False  # melhora -> reseta contador
    assert es.update(0.9) is False  # 1ª sem melhora
    assert es.update(0.9) is True  # 2ª sem melhora -> para


def test_min_delta_ignores_insignificant_improvement() -> None:
    es = EarlyStopping(patience=1, min_delta=0.1)
    assert es.update(1.0) is False
    # queda de 0.05 < min_delta -> conta como "sem melhora" -> para
    assert es.update(0.95) is True


def test_tracks_best_loss() -> None:
    es = EarlyStopping(patience=3)
    for loss in [1.0, 0.7, 0.9]:
        es.update(loss)
    assert es.best == 0.7
