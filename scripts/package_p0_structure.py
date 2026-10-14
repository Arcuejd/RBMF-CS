"""Seal the completed negative-result round without accessing holdout fields."""
import hashlib
import json
import platform
import shutil
import zipfile
from pathlib import Path

import numpy
import scipy

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    dest = ROOT / "outputs/p0_structure_delivery_v1"
    verification = json.loads((ROOT / "outputs/p0_structure_report_v1/verification.json").read_text())
    config = json.loads((ROOT / "configs/p0_strong_baselines_v1.json").read_text())
    selection = json.loads((ROOT / config["parameter_source"]).read_text())
    seal = json.loads((ROOT / "outputs/p0_holdout_v1/seal.json").read_text())
    assert verification["passed"] and not seal["release_authorized"]
    assert digest(ROOT / "outputs/p0_holdout_v1/manifest.json") == seal["manifest_sha256"]
    assert digest(ROOT / config["training_artifacts"]) == config["training_artifacts_sha256"]
    assert digest(ROOT / config["parameter_source"]) == config["parameter_source_sha256"]
    original_hash = digest(ROOT / "src/rbmf_cs/model.py")
    assert original_hash == "5e7173f7aca50511e4f1afff8a25150fea419eb9958008258b8b4781589c3168"
    for name, settings in config["methods"].items():
        assert settings["rank"] == selection["comparison"]["method_rank"][name]
        if "regularization" in settings:
            assert settings["regularization"] == selection["comparison"]["regularization"][name]
    log = ROOT / "outputs/p0_structure_report_v1/tests.log"
    log_text = log.read_bytes().decode("utf-16" if log.read_bytes().startswith(b"\xff\xfe") else "utf-8")
    assert "Ran 81 tests" in log_text and log_text.rstrip().endswith("OK")
    dest.mkdir(exist_ok=False)
    with zipfile.ZipFile(dest / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in ("src", "scripts", "tests", "configs", "docs"):
            for path in sorted((ROOT / folder).rglob("*")):
                if path.is_file() and path.suffix in (".py", ".md", ".json"):
                    archive.write(path, path.relative_to(ROOT).as_posix())
        for name in ("pyproject.toml", "requirements.txt"):
            if (ROOT / name).is_file():
                archive.write(ROOT / name, name)
    for path in list((ROOT / "docs").glob("p0_structure*.md")) + [ROOT / "configs/p0_strong_baselines_v1.json", log]:
        shutil.copy2(path, dest / path.name)
    artifacts = {}
    for folder in sorted((ROOT / "outputs").glob("p0_structure_*")):
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                artifacts[path.relative_to(ROOT).as_posix()] = {"sha256": digest(path), "bytes": path.stat().st_size}
    manifest = {
        "status": "negative_result_stop_rbmf_search",
        "primary_method": "pod_ridge",
        "independent_test_released": False,
        "tests_passed": 81,
        "verification": verification,
        "original_model_sha256": original_hash,
        "training_sha256": config["training_artifacts_sha256"],
        "holdout_manifest_sha256": seal["manifest_sha256"],
        "runtime": {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__},
        "artifacts": artifacts,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"delivery": str(dest), "files_hashed": len(artifacts), "status": manifest["status"]}))


if __name__ == "__main__":
    main()
