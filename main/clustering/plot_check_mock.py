#!/usr/bin/env python
"""Plot n(z) diagnostics for DESI data and Abacus QSO cutsky catalogs."""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


DATA_FN = "/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v2/nonKP/QSO_clustering.dat.fits"
ABACUS_DATA_FN = "/pscratch/sd/x/xryang/QSO_cutsky/ph000/merged_cutsky.dat"
ABACUS_RANDOM_FN = "/pscratch/sd/x/xryang/QSO_cutsky/ph000/merged_random.dat"

OUT_DIR = Path(__file__).resolve().parent / "results" / "mock_test"


def read_ascii_chunk(handle, max_rows: int) -> np.ndarray:
    """Read one whitespace ASCII chunk with columns RA, DEC, Z, Z_COSMO, NZ, STATUS, ..."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Input line .* contained no data")
        warnings.filterwarnings("ignore", message="loadtxt: input contained no data")
        chunk = np.loadtxt(handle, max_rows=max_rows)
    if chunk.ndim == 1 and chunk.size:
        chunk = chunk[None, :]
    return chunk

def stream_ascii_nz(
    fn: str,
    bins: np.ndarray,
    chunk_rows: int,
    max_rows: Optional[int],
    use_saved_nz: bool,
    label: str,
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray, int]:
    """Stream an Abacus ASCII catalog and return z counts plus saved-NZ bin means."""
    counts = np.zeros(len(bins) - 1, dtype=np.int64)
    saved_nz_sum = np.zeros(len(bins) - 1, dtype=float)
    saved_nz_count = np.zeros(len(bins) - 1, dtype=np.int64)
    total = 0
    rows_left = max_rows

    with open(fn, "r", encoding="utf-8") as handle:
        while rows_left is None or rows_left > 0:
            rows_to_read = chunk_rows if rows_left is None else min(chunk_rows, rows_left)
            chunk = read_ascii_chunk(handle, rows_to_read)
            if chunk.size == 0:
                break

            z = np.asarray(chunk[:, 2], dtype=float)
            status = np.asarray(chunk[:, 5], dtype=float)
            ok = np.isfinite(z) & (status != 0)
            counts += np.histogram(z[ok], bins=bins)[0]

            if use_saved_nz:
                saved_nz = np.asarray(chunk[:, 4], dtype=float)
                ok_nz = ok & np.isfinite(saved_nz)
                bin_idx = np.searchsorted(bins, z[ok_nz], side="right") - 1
                in_range = (bin_idx >= 0) & (bin_idx < len(counts))
                np.add.at(saved_nz_sum, bin_idx[in_range], saved_nz[ok_nz][in_range])
                np.add.at(saved_nz_count, bin_idx[in_range], 1)

            total += len(chunk)
            if rows_left is not None:
                rows_left -= len(chunk)

    saved_nz_mean = None
    if use_saved_nz:
        saved_nz_mean = np.divide(
            saved_nz_sum,
            saved_nz_count,
            out=np.full_like(saved_nz_sum, np.nan, dtype=float),
            where=saved_nz_count > 0,
        )

    print(f"{label}: processed {total:,} rows")
    return counts, saved_nz_mean, saved_nz_count, total


def read_fits_nz(fn: str, bins: np.ndarray, z_col: str, weight_col: Optional[str]) -> tuple[np.ndarray, int]:
    """Read DESI FITS Z values and return a possibly weighted n(z) histogram."""
    with fits.open(fn, memmap=True) as hdul:
        tab = hdul[1].data
        z = np.asarray(tab[z_col], dtype=float)
        ok = np.isfinite(z)

        weights = None
        if weight_col:
            weights = np.asarray(tab[weight_col], dtype=float)
            ok &= np.isfinite(weights)
            weights = weights[ok]

        counts = np.histogram(z[ok], bins=bins, weights=weights)[0]
        total = len(z)

    print(f"DATA_FN: processed {total:,} rows")
    return counts, total


def density_from_counts(counts: np.ndarray, bins: np.ndarray) -> np.ndarray:
    width = np.diff(bins)
    norm = np.sum(counts * width)
    if norm <= 0:
        return np.zeros_like(counts, dtype=float)
    return counts / norm


def plot_nz_comparison(
    bins: np.ndarray,
    data_counts: np.ndarray,
    abacus_data_counts: np.ndarray,
    abacus_data_saved_nz: np.ndarray,
    abacus_random_counts: np.ndarray,
    out_dir: Path,
) -> None:
    centers = 0.5 * (bins[:-1] + bins[1:])

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.step(centers, density_from_counts(data_counts, bins), where="mid", color="black", lw=1.8, label="DATA_FN n(z)")
    ax1.step(
        centers,
        density_from_counts(abacus_data_counts, bins),
        where="mid",
        color="darkred",
        lw=1.6,
        label="ABACUS_DATA_FN n(z)",
    )
    ax1.step(
        centers,
        density_from_counts(abacus_random_counts, bins),
        where="mid",
        color="royalblue",
        lw=1.6,
        label="ABACUS_RANDOM_FN n(z)",
    )
    ax1.set_xlabel("Redshift z")
    ax1.set_ylabel("Normalized histogram density")
    ax1.grid(alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(
        centers,
        abacus_data_saved_nz,
        color="darkorange",
        lw=1.6,
        ls="--",
        label="ABACUS_DATA_FN saved NZ column",
    )
    ax2.set_ylabel("Saved NZ column")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="best")
    ax1.set_title("QSO n(z) comparison")

    out_fn = out_dir / "nz_distribution_comparison.png"
    fig.tight_layout()
    fig.savefig(out_fn, dpi=200)
    plt.close(fig)
    print(f"wrote {out_fn}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-fn", default=DATA_FN)
    parser.add_argument("--abacus-data-fn", default=ABACUS_DATA_FN)
    parser.add_argument("--abacus-random-fn", default=ABACUS_RANDOM_FN)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--zmin", type=float, default=0.8)
    parser.add_argument("--zmax", type=float, default=2.1)
    parser.add_argument("--nbins", type=int, default=100)
    parser.add_argument("--chunk-rows", type=int, default=1_000_000)
    parser.add_argument("--max-rows", type=int, default=0, help="ASCII rows to read. Use 0 for all rows.")
    parser.add_argument("--data-z-col", default="Z")
    parser.add_argument("--data-weight-col", default="WEIGHT", help="Set to empty string for unweighted DATA_FN n(z).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bins = np.linspace(args.zmin, args.zmax, args.nbins + 1)
    max_rows = None if args.max_rows <= 0 else args.max_rows
    data_weight_col = args.data_weight_col or None

    data_counts, _ = read_fits_nz(args.data_fn, bins, args.data_z_col, data_weight_col)
    abacus_data_counts, abacus_data_saved_nz, _, _ = stream_ascii_nz(
        args.abacus_data_fn,
        bins,
        args.chunk_rows,
        max_rows,
        use_saved_nz=True,
        label="ABACUS_DATA_FN",
    )
    abacus_random_counts, _, _, _ = stream_ascii_nz(
        args.abacus_random_fn,
        bins,
        args.chunk_rows,
        max_rows,
        use_saved_nz=False,
        label="ABACUS_RANDOM_FN",
    )

    plot_nz_comparison(bins, data_counts, abacus_data_counts, abacus_data_saved_nz, abacus_random_counts, out_dir)


if __name__ == "__main__":
    main()
