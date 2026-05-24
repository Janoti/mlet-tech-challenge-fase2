"""Valida que o ambiente está configurado corretamente para rodar o projeto.

Uso:
    poetry run python scripts/validate_env.py

Verifica:
    1. Variáveis de ambiente lidas pelo Pydantic Settings (via .env).
    2. Dependências críticas instaladas e acessíveis.
    3. Diretórios de dados existentes.

Retorna código 0 em sucesso e 1 se alguma verificação falhar.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


_OK = "  [OK]"
_FAIL = "  [FALHA]"
_REQUIRED_PACKAGES = ["torch", "sklearn", "mlflow", "dvc", "pydantic"]


def _check_settings() -> list[str]:
    """Valida as configurações do Pydantic Settings e retorna erros."""
    errors: list[str] = []
    try:
        from recsys.config import settings

        checks = {
            "RANDOM_SEED": settings.random_seed,
            "NUM_USERS": settings.num_users,
            "NUM_ITEMS": settings.num_items,
            "NUM_INTERACTIONS": settings.num_interactions,
            "MLFLOW_TRACKING_URI": settings.mlflow_tracking_uri,
            "MLFLOW_EXPERIMENT_NAME": settings.mlflow_experiment_name,
            "LOG_LEVEL": settings.log_level,
        }
        for name, value in checks.items():
            print(f"{_OK} {name}={value}")
    except Exception as exc:
        errors.append(f"Settings inválido: {exc}")
    return errors


def _check_packages() -> list[str]:
    """Verifica se os pacotes críticos estão instalados."""
    errors: list[str] = []
    for pkg in _REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "?")
            print(f"{_OK} {pkg} {version}")
        except ImportError:
            print(f"{_FAIL} {pkg} NÃO encontrado")
            errors.append(pkg)
    return errors


def _check_directories() -> list[str]:
    """Verifica se os diretórios de dados existem."""
    errors: list[str] = []
    project_root = Path(__file__).resolve().parent.parent
    dirs = ["data/raw", "data/interim", "data/processed", "models", "notebooks"]
    for d in dirs:
        path = project_root / d
        if path.exists():
            print(f"{_OK} {d}/")
        else:
            print(f"{_FAIL} {d}/ não existe")
            errors.append(d)
    return errors


def main() -> int:
    """Executa todas as verificações e retorna código de saída."""
    print("\n=== Validação do Ambiente ===\n")

    print("[ Configurações (.env) ]")
    settings_errors = _check_settings()

    print("\n[ Dependências ]")
    package_errors = _check_packages()

    print("\n[ Diretórios ]")
    dir_errors = _check_directories()

    total_errors = len(settings_errors) + len(package_errors) + len(dir_errors)
    print("\n" + "=" * 30)
    if total_errors == 0:
        print("Ambiente OK — pronto para rodar o pipeline.")
        return 0

    print(f"FALHOU — {total_errors} problema(s) encontrado(s). Corrija antes de continuar.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
