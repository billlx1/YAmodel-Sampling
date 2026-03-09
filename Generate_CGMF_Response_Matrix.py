"""
CGMF Fragment Mass Transition Probability Matrix
=================================================
Computes P(A_post | A_pre): the probability of observing a post-neutron-emission
fragment mass A_post given a pre-neutron-emission fragment mass A_pre.

Also computes the relative statistical error on each probability element.

The matrix is defined as:
    P(A_post | A_pre) = N(A_pre -> A_post) / N(A_pre)

where N(A_pre -> A_post) is the number of fragments with pre-emission mass A_pre
that ended up with post-emission mass A_post = A_pre - nu_fragment (i.e. after
emitting nu_fragment neutrons).

Relative error (Poisson counting statistics):
    delta_rel(A_post | A_pre) = 1 / sqrt(N(A_pre -> A_post))

===============================================================================
USAGE EXAMPLES:
===============================================================================
Single file:
    python fragment_mass_matrix.py 98252sf.cgmf

Multiple files (pooled together — treated as independent realizations):
    python fragment_mass_matrix.py run1.cgmf run2.cgmf run3.cgmf

Custom output prefix:
    python fragment_mass_matrix.py 98252sf.cgmf --output cf252_matrix

Limit events read per file (useful for testing):
    python fragment_mass_matrix.py 98252sf.cgmf --nevents 5000

Label the system in plot titles:
    python fragment_mass_matrix.py histories.cgmf --system "235U(nth,f)"

Disable plotting:
    python fragment_mass_matrix.py 98252sf.cgmf --no-plot

===============================================================================
OUTPUT FILES:
===============================================================================
{PREFIX}_probability_matrix.csv   — Full P(A_post | A_pre) matrix (CSV)
{PREFIX}_relerr_matrix.csv        — Relative error matrix (CSV)
{PREFIX}_matrices.json            — Both matrices + metadata in JSON
{PREFIX}_matrix_plot.png          — Two-panel heatmap: P matrix + error matrix
{PREFIX}_nu_plot.png              — P(nu | A_pre) stacked area + mean-nu overlay
===============================================================================
"""

import sys
import os
import numpy as np
import json
import argparse
from datetime import datetime

# ========== CGMFTK PATH CONFIGURATION ==========
CGMF_INSTALL_PATH = os.environ.get("CGMFPATH", "")
COMMON_PATHS = [
    CGMF_INSTALL_PATH,
    os.path.expanduser("~/CGMF"),
    os.path.expanduser("~/Documents/CGMF"),
    "/usr/local/CGMF",
]

def find_cgmftk():
    """Locate CGMFtk in common installation locations."""
    for path in COMMON_PATHS:
        if path and os.path.exists(path):
            cgmftk_path = os.path.join(path, "tools")
            if os.path.exists(os.path.join(cgmftk_path, "CGMFtk")):
                return cgmftk_path
    return None

cgmftk_location = find_cgmftk()
if cgmftk_location and cgmftk_location not in sys.path:
    sys.path.insert(0, cgmftk_location)

try:
    from CGMFtk import histories as fh
    CGMFTK_AVAILABLE = True
except ImportError as e:
    print("✗ ERROR: CGMFtk not found in Python path")
    print(f"  Import error: {e}")
    print("\nTo fix this:")
    print("  1. Set environment variable: export CGMFPATH=/path/to/CGMF")
    print("  2. Or install CGMFtk: cd $CGMFPATH/tools/CGMFtk && pip install -e .")
    CGMFTK_AVAILABLE = False
    sys.exit(1)

# ========== MATPLOTLIB CONFIGURATION ==========
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.ticker import MultipleLocator, AutoMinorLocator
    from matplotlib.patches import Patch
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("⚠ WARNING: matplotlib not available — plotting disabled")
    MATPLOTLIB_AVAILABLE = False

# Colour palette for nu = 0..6: perceptually ordered grey → blue → yellow → red
NU_COLORS = [
    '#d0d0d0',  # nu=0  light grey
    '#4575b4',  # nu=1  blue
    '#74add1',  # nu=2  light blue
    '#fee090',  # nu=3  yellow
    '#fdae61',  # nu=4  orange
    '#f46d43',  # nu=5  red-orange
    '#d73027',  # nu=6  red
]


# ==============================================================================
# Data loading
# ==============================================================================

def load_histories(filepath, nevents=None):
    """
    Load a single CGMF histories file.

    Parameters
    ----------
    filepath : str
        Path to the .cgmf histories file.
    nevents : int or None
        If set, read only the first `nevents` events (useful for testing).

    Returns
    -------
    hist     : CGMFtk Histories object
    n_events : int
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"  Loading: {filepath}")
    if nevents:
        hist = fh.Histories(filepath, nevents=nevents)
    else:
        hist = fh.Histories(filepath)

    n_events = hist.getNumberEvents()
    print(f"  ✓ {n_events:,} fission events loaded")
    return hist, n_events


def extract_fragment_pairs(hist):
    """
    Extract (A_pre, A_post) pairs for every fission fragment.

    CGMFtk returns fragment-level arrays interleaved as:
        index 0, 2, 4, ... → light fragments of events 0, 1, 2, ...
        index 1, 3, 5, ... → heavy fragments of events 0, 1, 2, ...

    A_post = A_pre - nu_fragment  (neutrons emitted by that fragment only).

    Returns
    -------
    A_pre_all  : 1-D int array
    A_post_all : 1-D int array
    """
    A_pre   = np.array(hist.getA(),  dtype=int)
    nu_frag = np.array(hist.getNu(), dtype=int)
    A_post  = A_pre - nu_frag

    bad = A_post <= 0
    if bad.any():
        print(f"  ⚠ WARNING: {bad.sum()} fragments with A_post ≤ 0 — excluded")
        A_pre  = A_pre[~bad]
        A_post = A_post[~bad]

    return A_pre, A_post


# ==============================================================================
# Matrix construction
# ==============================================================================

def build_transition_matrix(A_pre_all, A_post_all):
    """
    Build count, probability, and relative-error matrices.

    Rows  → A_pre  (pre-neutron-emission mass)
    Cols  → A_post (post-neutron-emission mass)

    P[i,j]      = N(rows[i]→cols[j]) / N(rows[i])
    relerr[i,j] = 1/sqrt(N(rows[i]→cols[j]))  if N > 0, else 0

    Returns
    -------
    rows       : 1-D int array
    cols       : 1-D int array
    count_mat  : 2-D int64 array
    prob_mat   : 2-D float array
    relerr_mat : 2-D float array
    """
    rows = np.arange(A_pre_all.min(),  A_pre_all.max()  + 1, dtype=int)
    cols = np.arange(A_post_all.min(), A_post_all.max() + 1, dtype=int)

    row_idx = {a: i for i, a in enumerate(rows)}
    col_idx = {a: j for j, a in enumerate(cols)}

    count_mat = np.zeros((len(rows), len(cols)), dtype=np.int64)
    for a_pre, a_post in zip(A_pre_all, A_post_all):
        i = row_idx.get(a_pre)
        j = col_idx.get(a_post)
        if i is not None and j is not None:
            count_mat[i, j] += 1

    row_sums = count_mat.sum(axis=1, keepdims=True)

    with np.errstate(invalid='ignore', divide='ignore'):
        prob_mat   = np.where(row_sums > 0, count_mat / row_sums, 0.0)
        relerr_mat = np.where(count_mat > 0, 1.0 / np.sqrt(count_mat), 0.0)

    return rows, cols, count_mat, prob_mat, relerr_mat


def build_pnu(rows, cols, prob_mat, max_nu=6):
    """
    Derive P(nu | A_pre) from the transition probability matrix.

    nu = A_pre - A_post is a change of variable; this re-bins prob_mat
    into nu-space.

    Returns
    -------
    pnu      : 2-D float array, shape (len(rows), max_nu+1)
    nu_range : 1-D int array,   values 0 .. max_nu
    """
    nu_range = np.arange(0, max_nu + 1, dtype=int)
    pnu      = np.zeros((len(rows), max_nu + 1), dtype=float)

    for i, a_pre in enumerate(rows):
        for j, a_post in enumerate(cols):
            nu = int(a_pre) - int(a_post)
            if 0 <= nu <= max_nu:
                pnu[i, nu] = prob_mat[i, j]

    return pnu, nu_range


# ==============================================================================
# Export functions
# ==============================================================================

def export_matrix_csv(matrix, row_labels, col_labels, filepath, header_comment=""):
    """Write a labelled 2-D matrix to CSV."""
    with open(filepath, 'w') as f:
        if header_comment:
            for line in header_comment.strip().splitlines():
                f.write(f"# {line}\n")
        f.write("A_pre\\A_post," + ",".join(str(c) for c in col_labels) + "\n")
        for i, row_val in enumerate(row_labels):
            row_data = ",".join(f"{matrix[i, j]:.8e}" for j in range(len(col_labels)))
            f.write(f"{row_val},{row_data}\n")
    print(f"  ✓ CSV saved: {filepath}")


def export_json(rows, cols, count_mat, prob_mat, relerr_mat,
                source_files, total_events, total_fragments, filepath):
    """Export both matrices plus metadata as JSON."""
    data = {
        "metadata": {
            "timestamp":                datetime.now().isoformat(),
            "source_files":             [os.path.basename(f) for f in source_files],
            "total_fission_events":     int(total_events),
            "total_fragments_analysed": int(total_fragments),
            "description": (
                "P(A_post | A_pre): conditional probability that a fragment "
                "with pre-neutron-emission mass A_pre has post-emission mass "
                "A_post = A_pre - nu_fragment."
            )
        },
        "axes": {
            "A_pre":  rows.tolist(),
            "A_post": cols.tolist()
        },
        "count_matrix": {
            "description": "Raw fragment counts N(A_pre -> A_post)",
            "units":        "counts",
            "data":         count_mat.tolist()
        },
        "probability_matrix": {
            "description": "P(A_post | A_pre) — row-normalised probabilities",
            "units":        "dimensionless",
            "data":         prob_mat.tolist()
        },
        "relative_error_matrix": {
            "description": (
                "Relative statistical error = 1/sqrt(N(A_pre->A_post)). "
                "Zero where count is zero (undefined)."
            ),
            "units": "dimensionless",
            "data":  relerr_mat.tolist()
        }
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  ✓ JSON saved: {filepath}")


# ==============================================================================
# Plot 1 — Heatmap pair: probability matrix + relative error matrix
# ==============================================================================

def generate_matrix_plot(rows, cols, prob_mat, relerr_mat,
                         source_files, system_label, filepath):
    """
    Two-panel heatmap figure saved as {PREFIX}_matrix_plot.png:
      Left  — P(A_post | A_pre)      [log colour scale, inferno]
      Right — Relative error 1/√N    [log colour scale, viridis]
    """
    if not MATPLOTLIB_AVAILABLE:
        print("  ⚠ Skipping matrix plot (matplotlib unavailable)")
        return

    src_str  = ", ".join(os.path.basename(f) for f in source_files)
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    diag_min = max(int(rows[0]),  int(cols[0]))
    diag_max = min(int(rows[-1]), int(cols[-1]))

    # ── Left: probability ──────────────────────────────────────────────────
    ax = axes[0]
    p_disp = np.where(prob_mat > 0, prob_mat, np.nan)
    im0 = ax.imshow(
        p_disp, origin='lower', aspect='auto',
        norm=mcolors.LogNorm(vmin=float(np.nanmin(p_disp[p_disp > 0])), vmax=1.0),
        cmap='inferno',
        extent=[cols[0] - 0.5, cols[-1] + 0.5,
                rows[0] - 0.5, rows[-1] + 0.5]
    )
    fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04).set_label(
        r'$P(A_\mathrm{post}\,|\,A_\mathrm{pre})$', fontsize=12)
    ax.plot([diag_min, diag_max], [diag_min, diag_max],
            'w--', lw=1.0, alpha=0.65,
            label=r'$A_\mathrm{post}=A_\mathrm{pre}$  ($\nu=0$)')
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlabel(r'Post-emission Mass $A_\mathrm{post}$', fontsize=12)
    ax.set_ylabel(r'Pre-emission Mass $A_\mathrm{pre}$',   fontsize=12)
    ax.set_title(f'Transition Probability Matrix\n{system_label}  |  {src_str}',
                 fontsize=12, fontweight='bold')
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    # ── Right: relative error ──────────────────────────────────────────────
    ax = axes[1]
    r_disp = np.where(relerr_mat > 0, relerr_mat, np.nan)
    valid  = r_disp[~np.isnan(r_disp)]
    im1 = ax.imshow(
        r_disp, origin='lower', aspect='auto',
        norm=mcolors.LogNorm(vmin=float(valid.min()), vmax=float(valid.max())),
        cmap='viridis',
        extent=[cols[0] - 0.5, cols[-1] + 0.5,
                rows[0] - 0.5, rows[-1] + 0.5]
    )
    fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04).set_label(
        r'Relative Error $= 1/\sqrt{N}$', fontsize=12)
    ax.plot([diag_min, diag_max], [diag_min, diag_max],
            'w--', lw=1.0, alpha=0.65,
            label=r'$A_\mathrm{post}=A_\mathrm{pre}$  ($\nu=0$)')
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlabel(r'Post-emission Mass $A_\mathrm{post}$', fontsize=12)
    ax.set_ylabel(r'Pre-emission Mass $A_\mathrm{pre}$',   fontsize=12)
    ax.set_title(f'Relative Statistical Error Matrix\n{system_label}  |  {src_str}',
                 fontsize=12, fontweight='bold')
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Matrix plot saved: {filepath}")


# ==============================================================================
# Plot 2 — P(nu | A_pre) stacked area with mean-nu overlay
# ==============================================================================

def generate_nu_plot(rows, pnu, nu_range, source_files, system_label,
                     total_events, total_fragments, filepath):
    """
    Single-panel stacked area plot saved as {PREFIX}_nu_plot.png:
      Stacked coloured bands — P(nu | A_pre) for nu = 0, 1, 2, ...
      Black line on right axis — mean nu per fragment
    """
    if not MATPLOTLIB_AVAILABLE:
        print("  ⚠ Skipping nu plot (matplotlib unavailable)")
        return

    mean_nu = pnu @ nu_range.astype(float)

    fig, ax = plt.subplots(figsize=(16, 7))

    # ── Stacked area bands ─────────────────────────────────────────────────
    bottom = np.zeros(len(rows))
    for nu in nu_range:
        color = NU_COLORS[nu] if nu < len(NU_COLORS) else '#800026'
        top   = bottom + pnu[:, nu]
        ax.fill_between(rows, bottom, top, color=color, alpha=0.88, linewidth=0)
        ax.plot(rows, top, color='white', lw=0.5, alpha=0.55)
        bottom = top.copy()

    # ── Mean nu on twin right axis ─────────────────────────────────────────
    ax_r = ax.twinx()
    ax_r.plot(rows, mean_nu, color='black', lw=2.4, zorder=6,
              label=r'$\langle\nu\rangle$ per fragment')
    ax_r.fill_between(rows, 0, mean_nu, color='black', alpha=0.06, zorder=1)
    ax_r.set_ylabel(
        r'Mean neutrons emitted per fragment  $\langle\nu\rangle$', fontsize=12)
    ax_r.set_ylim(0, mean_nu.max() * 1.35)
    ax_r.yaxis.set_minor_locator(AutoMinorLocator())
    ax_r.legend(loc='upper right', fontsize=10, framealpha=0.9)

    # ── Axes cosmetics ─────────────────────────────────────────────────────
    ax.set_xlim(rows[0], rows[-1])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(
        r'Pre-neutron-emission Fragment Mass  $A_\mathrm{pre}$', fontsize=13)
    ax.set_ylabel(
        r'Probability  $P(\nu\,|\,A_\mathrm{pre})$', fontsize=13)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(2))

    legend_handles = [
        Patch(facecolor=NU_COLORS[k] if k < len(NU_COLORS) else '#800026',
              alpha=0.88,
              label=fr'$\nu={k}$ neutrons')
        for k in nu_range
    ]
    ax.legend(handles=legend_handles, loc='upper center',
              ncol=len(nu_range), fontsize=9.5, framealpha=0.88,
              bbox_to_anchor=(0.5, 0.985))

    src_str  = ", ".join(os.path.basename(f) for f in source_files)
    n_ev_str = f"{total_events:,}"    if isinstance(total_events,    int) else str(total_events)
    n_fr_str = f"{total_fragments:,}" if isinstance(total_fragments, int) else str(total_fragments)

    ax.set_title(
        fr'Neutron Multiplicity Distribution $P(\nu\,|\,A_{{\mathrm{{pre}}}})$'
        f' vs. Pre-emission Fragment Mass\n'
        f'{system_label}  |  {src_str}  |  '
        f'{n_ev_str} events · {n_fr_str} fragments',
        fontsize=12, fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Nu plot saved: {filepath}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build fragment mass transition probability matrices from CGMF histories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'histories_files',
        nargs='+',
        metavar='FILE',
        help='One or more CGMF histories files (.cgmf). Multiple files are pooled.'
    )
    parser.add_argument(
        '--output', '-o',
        type=str, default=None,
        help='Output file prefix. Default: derived from first input filename.'
    )
    parser.add_argument(
        '--system',
        type=str, default=None,
        help=(
            'Reaction label for plot titles, e.g. "235U(nth,f)" or "252Cf(sf)". '
            'Default: derived from first input filename.'
        )
    )
    parser.add_argument(
        '--nevents',
        type=int, default=None,
        help='Maximum events to read per file (useful for testing).'
    )
    parser.add_argument(
        '--max-nu',
        type=int, default=6,
        help='Maximum nu value shown in the stacked nu plot.'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Disable all plot generation.'
    )
    args = parser.parse_args()

    # ── Output prefix and system label ────────────────────────────────────
    base_name = os.path.splitext(os.path.basename(args.histories_files[0]))[0]
    prefix    = args.output if args.output else f"{base_name}_fragment_matrix"
    sys_label = args.system if args.system else base_name

    # ── Load and pool histories ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Loading CGMF Histories")
    print(f"{'='*70}")

    all_A_pre, all_A_post = [], []
    total_events = 0
    loaded_files = []

    for filepath in args.histories_files:
        try:
            hist, n_events = load_histories(filepath, nevents=args.nevents)
        except FileNotFoundError as exc:
            print(f"  ✗ Skipping {filepath}: {exc}")
            continue

        A_pre, A_post = extract_fragment_pairs(hist)
        all_A_pre.append(A_pre)
        all_A_post.append(A_post)
        total_events += n_events
        loaded_files.append(filepath)

    if not loaded_files:
        print("✗ No valid history files loaded. Exiting.")
        sys.exit(1)

    A_pre_all       = np.concatenate(all_A_pre)
    A_post_all      = np.concatenate(all_A_post)
    total_fragments = len(A_pre_all)

    print(f"\n✓ Total events loaded : {total_events:,}")
    print(f"✓ Total fragments     : {total_fragments:,}")
    print(f"✓ A_pre  range        : {A_pre_all.min()} – {A_pre_all.max()}")
    print(f"✓ A_post range        : {A_post_all.min()} – {A_post_all.max()}")

    # ── Build matrices ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Building Transition Matrices")
    print(f"{'='*70}")

    rows, cols, count_mat, prob_mat, relerr_mat = build_transition_matrix(
        A_pre_all, A_post_all
    )

    populated   = int((count_mat > 0).sum())
    total_cells = count_mat.size
    print(f"  Matrix dimensions : {len(rows)} A_pre × {len(cols)} A_post")
    print(f"  Populated cells   : {populated:,} / {total_cells:,} "
          f"({100*populated/total_cells:.1f}%)")
    print(f"  Max probability   : {prob_mat.max():.6f}")
    min_re = relerr_mat[relerr_mat > 0].min()
    print(f"  Min relative error: {min_re:.6f} "
          f"(= {int(round(1/min_re**2)):,} counts in best-sampled cell)")

    # ── Build P(nu | A_pre) ────────────────────────────────────────────────
    pnu, nu_range = build_pnu(rows, cols, prob_mat, max_nu=args.max_nu)

    mean_nu_overall = float((pnu @ nu_range.astype(float)).mean())
    print(f"\n  Mean nu per fragment (all A_pre): {mean_nu_overall:.4f}")
    print(f"  Implied mean nu per fission     : {2*mean_nu_overall:.4f}")

    # ── Export data files ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Exporting Results")
    print(f"{'='*70}")

    ts = datetime.now().isoformat()

    export_matrix_csv(
        prob_mat, rows, cols,
        f"{prefix}_probability_matrix.csv",
        header_comment=(
            f"Fragment mass transition probability matrix P(A_post | A_pre)\n"
            f"System: {sys_label}\n"
            f"Source files: {', '.join(os.path.basename(f) for f in loaded_files)}\n"
            f"Total fission events: {total_events}  |  Total fragments: {total_fragments}\n"
            f"Generated: {ts}\n"
            f"Rows = A_pre (pre-neutron-emission mass)\n"
            f"Cols = A_post (post-neutron-emission mass = A_pre - nu_fragment)"
        )
    )

    export_matrix_csv(
        relerr_mat, rows, cols,
        f"{prefix}_relerr_matrix.csv",
        header_comment=(
            f"Relative statistical error matrix  1/sqrt(N(A_pre->A_post))\n"
            f"System: {sys_label}\n"
            f"Source files: {', '.join(os.path.basename(f) for f in loaded_files)}\n"
            f"Zero entries = no counts (undefined error)\n"
            f"Generated: {ts}"
        )
    )

    export_json(
        rows, cols, count_mat, prob_mat, relerr_mat,
        loaded_files, total_events, total_fragments,
        f"{prefix}_matrices.json"
    )

    # ── Generate plots ─────────────────────────────────────────────────────
    if not args.no_plot:
        print(f"\n{'='*70}")
        print("Generating Plots")
        print(f"{'='*70}")

        generate_matrix_plot(
            rows, cols, prob_mat, relerr_mat,
            loaded_files, sys_label,
            f"{prefix}_matrix_plot.png"
        )

        generate_nu_plot(
            rows, pnu, nu_range,
            loaded_files, sys_label,
            total_events, total_fragments,
            filepath=f"{prefix}_nu_plot.png"
        )

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("✓ COMPLETE")
    print(f"{'='*70}")
    print(f"\nOutput files:")
    print(f"  • {prefix}_probability_matrix.csv")
    print(f"  • {prefix}_relerr_matrix.csv")
    print(f"  • {prefix}_matrices.json")
    if not args.no_plot and MATPLOTLIB_AVAILABLE:
        print(f"  • {prefix}_matrix_plot.png")
        print(f"  • {prefix}_nu_plot.png")
    print()


if __name__ == "__main__":
    if not CGMFTK_AVAILABLE:
        sys.exit(1)
    main()