import tomllib
from pathlib import Path


def test_ci_and_package_metadata_target_supported_python_versions() -> None:
    root = Path(__file__).parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    workflow = (root / ".github/workflows/test.yml").read_text(encoding="utf-8")

    assert metadata["project"]["requires-python"] == ">=3.11,<3.13"
    assert 'python-version: ["3.11", "3.12"]' in workflow
