from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CODE_INSTANCE_DIR = DATA_DIR / "code_instances"
DOCS_DIR = PROJECT_ROOT / "docs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = ARTIFACTS_DIR / "results"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_output_path(value: str | Path) -> Path:
    path = resolve_path(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
