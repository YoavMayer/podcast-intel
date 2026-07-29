"""
Guards that a CORE install stays importable.

Optional extras (``[transcription]``, ``[analysis]``) must never be pulled in by a
package or subpackage ``__init__``. A module-level import of an extra turns
``import podcast_intel.<subpackage>`` into a ``ModuleNotFoundError`` for anyone who
installed the core package -- which is exactly the defect this test exists to catch.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "podcast_intel"

#: Distributions that are NOT core dependencies of podcast-intel.
EXTRA_ONLY_MODULES = {
    "faster_whisper",
    "torch",
    "librosa",
    "soundfile",
    "sklearn",
    "transformers",
    "scipy",
    "numpy",
}

INIT_FILES = sorted(SRC.rglob("__init__.py"))


def _module_level_imports(path: Path) -> set:
    """Top-level (non-nested) imported root module names in a source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in tree.body:  # module level only -- nested/lazy imports are fine
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_there_are_init_files_to_check():
    assert len(INIT_FILES) >= 6


@pytest.mark.parametrize("init_file", INIT_FILES, ids=lambda p: str(p.name))
def test_no_init_imports_an_optional_extra_at_module_level(init_file: Path):
    """Directly-imported extras."""
    leaked = _module_level_imports(init_file) & EXTRA_ONLY_MODULES
    assert not leaked, f"{init_file} imports optional extras at module level: {leaked}"


@pytest.mark.parametrize(
    "module",
    [
        "podcast_intel",
        "podcast_intel.analysis",
        "podcast_intel.config",
        "podcast_intel.ingestion",
        "podcast_intel.models",
        "podcast_intel.presets",
        "podcast_intel.transcription",
        "podcast_intel.triggers",
        "podcast_intel.triggers.providers",
    ],
)
def test_import_pulls_in_no_optional_extra(module: str):
    """
    Transitive leaks: import the module in a clean interpreter and assert none of
    the extra-only distributions ended up in sys.modules.

    This is what makes ``import podcast_intel.transcription`` safe on a core
    install: ``WhisperTranscriber`` is resolved lazily via PEP 562 ``__getattr__``.
    """
    code = (
        "import importlib, sys\n"
        f"importlib.import_module({module!r})\n"
        f"leaked = sorted({sorted(EXTRA_ONLY_MODULES)!r} and "
        f"[m for m in {sorted(EXTRA_ONLY_MODULES)!r} if m in sys.modules])\n"
        "print(','.join(leaked))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(SRC.parents[1]),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", (
        f"importing {module} pulled in optional extras: {result.stdout.strip()}"
    )


def test_whisper_transcriber_is_still_reachable_by_name():
    """Lazy must not mean gone: the export still resolves when the extra is present."""
    import podcast_intel.transcription as transcription

    assert "WhisperTranscriber" in transcription.__all__
    assert "WhisperTranscriber" in dir(transcription)

    pytest.importorskip("faster_whisper", reason="[transcription] extra not installed")
    assert transcription.WhisperTranscriber.__name__ == "WhisperTranscriber"


def test_unknown_attribute_still_raises_attribute_error():
    import podcast_intel.transcription as transcription

    with pytest.raises(AttributeError):
        transcription.NoSuchTranscriber
