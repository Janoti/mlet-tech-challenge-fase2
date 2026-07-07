"""Testes do módulo `recsys.utils.seed`."""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

from recsys.utils.seed import set_global_seed


class TestSetGlobalSeed:
    """`set_global_seed` deve fixar todas as fontes de aleatoriedade."""

    def test_same_seed_produces_same_random_sequence(self) -> None:
        set_global_seed(123)
        first = [random.random() for _ in range(5)]
        set_global_seed(123)
        second = [random.random() for _ in range(5)]
        assert first == second

    def test_same_seed_produces_same_numpy_sequence(self) -> None:
        # Reprodutibilidade do numpy é o requisito mais importante aqui —
        # é onde o gerador de dataset puxa a aleatoriedade.
        set_global_seed(7)
        first = np.random.rand(10)
        set_global_seed(7)
        second = np.random.rand(10)
        np.testing.assert_array_equal(first, second)

    def test_same_seed_produces_same_torch_sequence(self) -> None:
        # A rede neural da Etapa 4 depende do RNG do torch (init dos embeddings
        # e shuffle do DataLoader). Sem isso, `dvc repro` não é determinístico.
        torch = pytest.importorskip("torch")
        set_global_seed(42)
        first = torch.rand(10)
        set_global_seed(42)
        second = torch.rand(10)
        assert torch.equal(first, second)

    def test_pythonhashseed_is_set(self) -> None:
        set_global_seed(99)
        assert os.environ["PYTHONHASHSEED"] == "99"

    def test_negative_seed_raises(self) -> None:
        # Validação no boundary — Clean Code cap. 7. Mensagem deve citar o
        # valor recebido para facilitar diagnóstico.
        with pytest.raises(ValueError, match="recebido: -1"):
            set_global_seed(-1)
