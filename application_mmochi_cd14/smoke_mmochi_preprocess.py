"""Smoke test: CD14 normalization, current code versus MMoCHi preprocessing.

Downloads the three 10x runs used in the MMoCHi landmark-registration tutorial
(10k, 5k, 1k PBMC protein v3) next to this script if they are missing, and
installs MMoCHi from https://github.com/donnafarberlab/MMoCHi (main) with pip
if it is not importable.

Compares, for the 5k run (CD14 > 0):
  A. current code: log(CP1k + 1) over the ADTs shared by the 1k and 5k runs;
  B. same formula over the ADTs shared by all three runs;
  C. mmochi.utils.preprocess_adatas on all three runs (default arguments).
If B and C agree, C can be reproduced without MMoCHi at run time.
Writes smoke_mmochi.txt next to this script.
"""

from __future__ import annotations

import os

# Recent scikit-learn, imported by MMoCHi through imbalanced-learn, refuses to
# load unless SciPy array-API support is enabled before SciPy is first imported.
os.environ.setdefault("SCIPY_ARRAY_API", "1")

import subprocess
import sys
import urllib.request
from importlib import metadata
from pathlib import Path

# MMoCHi 0.3.5 normalizes in place, which fails on the read-only arrays of
# pandas >= 3 ("output array is read-only"). Pin pandas < 3 and restart.
if int(metadata.version("pandas").split(".")[0]) >= 3:
    print("Installing pandas < 3 (required by MMoCHi) and restarting")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas>=2.2,<3"])
    os.execv(sys.executable, [sys.executable, *sys.argv])

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402

from modal_transport import fit_modal_configuration  # noqa: E402
from run_mmochi_cd14 import FILE_1K, FILE_5K, REGION, THRESHOLD, load_adt  # noqa: E402

FILE_10K = HERE / "pbmc_10k_protein_v3_filtered_feature_bc_matrix.h5"
OUT = HERE / "smoke_mmochi.txt"
BASE = "https://cf.10xgenomics.com/samples/cell-exp"
URLS = {FILE_10K: f"{BASE}/3.0.0/pbmc_10k_protein_v3/pbmc_10k_protein_v3_filtered_feature_bc_matrix.h5",
        FILE_5K: f"{BASE}/3.0.2/5k_pbmc_protein_v3/5k_pbmc_protein_v3_filtered_feature_bc_matrix.h5",
        FILE_1K: f"{BASE}/3.0.0/pbmc_1k_protein_v3/pbmc_1k_protein_v3_filtered_feature_bc_matrix.h5"}
MMOCHI_GIT = "git+https://github.com/donnafarberlab/MMoCHi.git"


def download(url: str, path: Path) -> None:
    # 10x rejects the default urllib user agent (HTTP 403).
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(path)


def ensure_files() -> None:
    for path, url in URLS.items():
        if not path.exists():
            print(f"Downloading {path.name}")
            try:
                download(url, path)
            except Exception:
                download(url.replace("https://", "http://"), path)


def ensure_mmochi() -> None:
    try:
        import mmochi  # noqa: F401
    except ImportError:
        print("Installing MMoCHi from GitHub")
        subprocess.check_call([sys.executable, "-m", "pip", "install", MMOCHI_GIT])


def log_cp1k(x: np.ndarray, names: list[str], panel: list[str]) -> np.ndarray:
    sub = x[:, [names.index(m) for m in panel]]
    total = sub.sum(axis=1, keepdims=True)
    cp1k = np.divide(1000.0 * sub, total, out=np.zeros_like(sub), where=total > 0)
    cd14 = np.log1p(cp1k)[:, panel.index("CD14")]
    return cd14[cd14 > 0]


def mmochi_cd14() -> tuple[np.ndarray, list[str], str]:
    import mmochi as mmc

    try:
        version = metadata.version("mmochi")
    except metadata.PackageNotFoundError:
        version = getattr(mmc, "__version__", "unknown")
    files = [str(FILE_10K), str(FILE_5K), str(FILE_1K)]
    adatas = mmc.utils.preprocess_adatas(files, log_CP_ADT=1e3, log_CP_GEX=1e4)
    ad5 = adatas[1]
    prot = ad5.obsm["protein"]
    cols = [str(c) for c in prot.columns]
    col = next(c for c in cols if c == "CD14" or c.startswith("CD14"))
    v = np.asarray(prot[col], dtype=float)
    return v[v > 0], cols, version


def describe(label: str, v: np.ndarray) -> list[str]:
    fit = fit_modal_configuration(v[:, None], analysis_region=REGION)
    q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
    return [f"{label}: n={len(v)}, frac>{THRESHOLD:g}={np.mean(v > THRESHOLD):.4f}, "
            f"quantiles(5,25,50,75,95)={np.round(q, 4).tolist()}, "
            f"modes={np.round(fit.modes.ravel(), 4).tolist()}"]


def main() -> None:
    ensure_files()
    ensure_mmochi()
    (x1, n1), (x5, n5), (x10, n10) = (load_adt(f) for f in (FILE_1K, FILE_5K, FILE_10K))
    panel2 = sorted(set(n1) & set(n5))
    panel3 = sorted(set(n1) & set(n5) & set(n10))
    a, b = log_cp1k(x5, n5, panel2), log_cp1k(x5, n5, panel3)
    lines = [f"pandas version: {metadata.version('pandas')}",
             f"panel 1k&5k ({len(panel2)}): {panel2}",
             f"panel 1k&5k&10k ({len(panel3)}): {panel3}"]
    lines += describe("A current", a) + describe("B three-run panel", b)
    try:
        c, cols, version = mmochi_cd14()
    except Exception as err:  # report A and B even if MMoCHi fails
        lines.append(f"C mmochi failed: {type(err).__name__}: {err}")
    else:
        lines += [f"mmochi version: {version}", f"mmochi protein columns ({len(cols)}): {cols}"]
        lines += describe("C mmochi", c)
        if len(b) == len(c):
            lines.append(f"max |sorted B - sorted C| = {np.max(np.abs(np.sort(b) - np.sort(c))):.3e}")
        else:
            lines.append("B and C differ in cell count; compare the summaries above.")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()