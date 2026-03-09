"""
=============================================================================
STEP 1 — Load yield data and marginalise over charge number Z
=============================================================================

What this script does:
  1. Reads yield.csv (which has columns: Product ; Yield ; Error)
  2. Parses the nuclide name (e.g. "Kr92") into element symbol, Z, and A
  3. Groups all (Z, A) entries by mass number A and sums:
       Y(A)     = sum over Z of Y(A,Z)
       sigma(A) = sqrt( sum over Z of sigma(A,Z)^2 )
  4. Saves the result as a numpy .npz file for the next script to load

=============================================================================
NUMPY CONCEPTS INTRODUCED HERE:
  - np.array()       : create a NumPy array from a Python list
  - np.sqrt()        : square root of every element
  - np.sum()         : sum of all elements (or along an axis)
  - np.argsort()     : get the indices that would sort an array
  - array indexing   : using [i], [start:end], and boolean masks
=============================================================================
"""

import numpy as np          # numerical arrays and maths
import matplotlib           # plotting library (we configure it here)
matplotlib.use('Agg')       # use non-interactive backend (saves to file, no pop-up window needed)
import matplotlib.pyplot as plt  # the actual plotting functions
import os
import sys

# --- Element symbol → atomic number Z lookup table ---
# We need this because the CSV gives us "Kr92" but we need Z=36 for Krypton
ELEMENT_Z = {
    'H':1,  'He':2, 'Li':3, 'Be':4, 'B':5,  'C':6,  'N':7,  'O':8,
    'F':9,  'Ne':10,'Na':11,'Mg':12,'Al':13,'Si':14,'P':15, 'S':16,
    'Cl':17,'Ar':18,'K':19, 'Ca':20,'Sc':21,'Ti':22,'V':23, 'Cr':24,
    'Mn':25,'Fe':26,'Co':27,'Ni':28,'Cu':29,'Zn':30,'Ga':31,'Ge':32,
    'As':33,'Se':34,'Br':35,'Kr':36,'Rb':37,'Sr':38,'Y':39, 'Zr':40,
    'Nb':41,'Mo':42,'Tc':43,'Ru':44,'Rh':45,'Pd':46,'Ag':47,'Cd':48,
    'In':49,'Sn':50,'Sb':51,'Te':52,'I':53, 'Xe':54,'Cs':55,'Ba':56,
    'La':57,'Ce':58,'Pr':59,'Nd':60,'Pm':61,'Sm':62,'Eu':63,'Gd':64,
    'Tb':65,'Dy':66,'Ho':67,'Er':68,'Tm':69,'Yb':70,'Lu':71,'Hf':72,
    'Ta':73,'W':74, 'Re':75,'Os':76,'Ir':77,'Pt':78,'Au':79,'Hg':80,
    'Tl':81,'Pb':82,'Bi':83,'Po':84,'At':85,'Rn':86,'Fr':87,'Ra':88,
    'Ac':89,'Th':90,'Pa':91,'U':92, 'Np':93,'Pu':94,'Am':95,'Cm':96,
    'Bk':97,'Cf':98,'Es':99,'Fm':100
}

def parse_nuclide(name):
    """
    Parse a nuclide name into (symbol, Z, A, isomer_tag), correctly handling isomers.

    Nuclear isomers are long-lived excited states of a nucleus.  They share
    the same Z and A as the ground state but carry a suffix in their name:
        Kr85   = ground state of Krypton-85
        Kr85m  = first metastable isomer  (same Z=36, same A=85)
        Kr85m2 = second metastable isomer (same Z=36, same A=85)
        In116n = a different isomer notation used in some libraries
        In116m = yet another notation

    For MASS YIELD purposes these are physically the same fragment.
    A fission event that produces Kr85m still produces a fragment with A=85.
    The metastable state will eventually decay to the ground state — the mass
    does not change.  So we strip the isomer suffix and add the yield to the
    same (Z, A) bin as the ground state.

    Parsing strategy (processes name left-to-right in three phases):
        Phase 1: collect leading letters        → element symbol
        Phase 2: collect digits after symbol    → mass number A
        Phase 3: discard trailing letters/digits → isomer tag (m, n, m2, ...)

    Examples:
        "Kr85"    → ("Kr", 36, 85, "")    ground state
        "Kr85m"   → ("Kr", 36, 85, "m")   isomer — same Z, A as above
        "Nb100m"  → ("Nb", 41, 100, "m")  isomer stripped
        "In116n"  → ("In", 49, 116, "n")  n-isomer
        "Pm152m2" → ("Pm", 61, 152, "m2") second metastable

    Parameters
    ----------
    name : str
        Nuclide name, possibly with isomer suffix

    Returns
    -------
    (symbol, Z, A, isomer_tag) or None if the name cannot be parsed
    """
    name = name.strip()

    # ── Phase 1: extract the element symbol (leading letters only) ────────────
    symbol = ""
    i = 0
    while i < len(name) and name[i].isalpha():
        symbol += name[i]
        i += 1

    # ── Phase 2: extract the mass number (digits immediately after symbol) ────
    digits = ""
    while i < len(name) and name[i].isdigit():
        digits += name[i]
        i += 1

    # ── Phase 3: remainder is the isomer tag — discard it silently ───────────
    isomer_tag = name[i:]   # e.g. "m", "n", "m2", "" for ground state

    # Validate: both symbol and digits must be non-empty
    if not symbol or not digits:
        return None

    if symbol not in ELEMENT_Z:
        return None

    Z = ELEMENT_Z[symbol]
    A = int(digits)

    # Return the isomer tag too so the caller can log statistics
    return (symbol, Z, A, isomer_tag)


def load_and_marginalise(csv_path):
    """
    Read the yield CSV, skip header/comment lines, parse each nuclide,
    group by mass number A, and return arrays of A, Y(A), sigma(A).
    """
    print("=" * 60)
    print("STEP 1: Loading yield data from:", csv_path)
    print("=" * 60)

    if not os.path.exists(csv_path):
        print(f"ERROR: Cannot find file '{csv_path}'")
        print("  Make sure yield.csv is in the same folder as this script.")
        sys.exit(1)

    # We'll accumulate data in a Python dictionary keyed by mass number A.
    # For each A we keep a list of (yield, error) pairs from different Z values.
    #
    # dict structure:  { A: {'yields': [y1, y2, ...], 'errors': [e1, e2, ...]} }
    data_by_A = {}

    skipped_header  = 0
    skipped_zero    = 0
    parsed_ok       = 0
    parsed_isomers  = 0
    parse_failed    = 0

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        # encoding='utf-8-sig' strips the BOM character (\ufeff) that Excel
        # sometimes writes at the start of CSV files.
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            if not line:
                continue

            parts = [p.strip() for p in line.split(';')]

            if len(parts) < 3:
                print(f"  [Line {line_number}] Skipping (not enough columns): {line!r}")
                skipped_header += 1
                continue

            product_str, yield_str, error_str = parts[0], parts[1], parts[2]

            try:
                yield_val = float(yield_str)
                error_val = float(error_str)
            except ValueError:
                print(f"  [Line {line_number}] Skipping header/non-numeric row: {line!r}")
                skipped_header += 1
                continue

            if yield_val == 0.0:
                skipped_zero += 1
                continue

            # parse_nuclide now returns (symbol, Z, A, isomer_tag)
            # Isomers (m, n, m2 ...) are folded into the same A bin as the ground state.
            result = parse_nuclide(product_str)
            if result is None:
                print(f"  [Line {line_number}] WARNING: Could not parse nuclide '{product_str}', skipping.")
                parse_failed += 1
                continue

            symbol, Z, A, isomer_tag = result
            parsed_ok += 1
            if isomer_tag:
                parsed_isomers += 1

            if A not in data_by_A:
                data_by_A[A] = {'yields': [], 'errors': []}
            data_by_A[A]['yields'].append(yield_val)
            data_by_A[A]['errors'].append(error_val)

    print(f"\n  Parsing summary:")
    print(f"    Header/non-numeric rows skipped           : {skipped_header}")
    print(f"    Zero-yield rows skipped                   : {skipped_zero}")
    print(f"    Ground-state nuclides parsed              : {parsed_ok - parsed_isomers}")
    print(f"    Isomeric states folded into A bins        : {parsed_isomers}")
    print(f"    Total rows contributing to Y(A)           : {parsed_ok}")
    print(f"    Rows that truly failed to parse           : {parse_failed}")
    print(f"    Unique mass numbers A found               : {len(data_by_A)}")

    if len(data_by_A) == 0:
        print("\nERROR: No valid data was parsed. Check your CSV format.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # MARGINALISATION
    # -------------------------------------------------------------------------
    # For each mass number A, sum over all Z contributions:
    #   Y(A)      = sum_Z  Y(A,Z)
    #   sigma(A)  = sqrt( sum_Z  sigma(A,Z)^2 )
    #
    # This is the Z-marginalisation step described in the project specification.
    # We assume yields at different Z are uncorrelated (standard assumption when
    # no covariance information is available from the evaluated library).
    # -------------------------------------------------------------------------

    A_list     = []
    Y_list     = []
    sigma_list = []

    for A in sorted(data_by_A.keys()):
        yields = data_by_A[A]['yields']
        errors = data_by_A[A]['errors']

        # Convert to NumPy arrays so we can do array maths
        # np.array() turns a plain Python list into a NumPy array
        yields_arr = np.array(yields)   # shape: (number of Z contributors,)
        errors_arr = np.array(errors)

        # Sum all yields at this A
        # np.sum() adds up every element in the array
        Y_A = np.sum(yields_arr)

        # Propagate errors in quadrature: sigma = sqrt( sum of sigma_i^2 )
        # errors_arr**2  squares every element
        # np.sum(...)    sums all those squares
        # np.sqrt(...)   takes the square root of the total
        sigma_A = np.sqrt(np.sum(errors_arr**2))

        A_list.append(A)
        Y_list.append(Y_A)
        sigma_list.append(sigma_A)

    # Convert our lists to NumPy arrays — from here on we always work with arrays
    A_arr     = np.array(A_list,     dtype=float)
    Y_arr     = np.array(Y_list,     dtype=float)
    sigma_arr = np.array(sigma_list, dtype=float)

    print(f"\n  After marginalisation:")
    print(f"    Mass range      : A = {int(A_arr.min())} to {int(A_arr.max())}")
    print(f"    Number of points: {len(A_arr)}")
    print(f"    Sum of Y(A)     : {np.sum(Y_arr):.6f}  (should be ~2.0 for U-235 thermal fission)")

    # Quick sanity check: fission yields should sum to 2.0 (two fragments per event)
    total = np.sum(Y_arr)
    if abs(total - 2.0) > 0.15:
        print(f"  WARNING: Sum of Y(A) = {total:.4f}, expected ~2.0.")
        print(f"           This could mean many small yields were filtered as zero.")
        print(f"           The MCMC will still run but treat this result with caution.")

    return A_arr, Y_arr, sigma_arr


def plot_mass_yields(A_arr, Y_arr, sigma_arr, output_path="step1_mass_yields.png"):
    """
    Plot the marginalised mass yields Y(A) with 1-sigma error bars.

    MATPLOTLIB CONCEPTS:
      plt.figure()        : create a new figure (the canvas)
      plt.errorbar()      : plot points with error bars
      plt.xlabel/ylabel() : axis labels
      plt.title()         : plot title
      plt.yscale('log')   : use logarithmic scale on y-axis (useful for yields
                            that span many orders of magnitude)
      plt.grid()          : show grid lines
      plt.tight_layout()  : auto-adjust spacing so labels don't overlap
      plt.savefig()       : save the figure to a file
      plt.close()         : close the figure (free memory)
    """
    print(f"\n  Plotting mass yields → {output_path}")

    # Create a figure.  figsize=(width, height) in inches.
    fig, ax = plt.subplots(figsize=(12, 5))

    # errorbar() plots Y(A) as points with vertical error bars of ± sigma(A)
    #   fmt='o'     : data points shown as circles
    #   markersize  : circle size
    #   capsize     : width of the horizontal caps on the error bars
    #   linewidth   : thickness of the error bar lines
    ax.errorbar(
        A_arr, Y_arr,
        yerr=sigma_arr,
        fmt='o',
        markersize=3,
        capsize=2,
        linewidth=0.8,
        color='steelblue',
        label='Y(A) from evaluated library'
    )

    ax.set_xlabel("Mass number A", fontsize=13)
    ax.set_ylabel("Y(A)  [fissions per fission]", fontsize=13)
    ax.set_title("Y(A) Mass Yields", fontsize=14)

    # Use log scale on y-axis: yields span many orders of magnitude
    ax.set_yscale('log')

    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.legend()

    # tight_layout adjusts subplot parameters to avoid clipping labels
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"  Plot saved to: {output_path}")


def save_data(A_arr, Y_arr, sigma_arr, output_path="step1_output.npz"):
    """
    Save the marginalised data to a NumPy .npz file.

    .npz is NumPy's compressed archive format — think of it like a zip file
    that stores multiple named NumPy arrays. It's much faster and more reliable
    to load than a CSV when working with numerical data.

    np.savez() takes:
      - output_path : where to save the file
      - keyword args: name=array  (you choose the names)
    """
    np.savez(output_path, A=A_arr, Y=Y_arr, sigma=sigma_arr)
    print(f"\n  Data saved to: {output_path}")
    print(f"  Arrays stored: A (mass numbers), Y (yields), sigma (uncertainties)")


# =============================================================================
# MAIN — runs when you execute this script directly
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Step 1 — Load fission yield data, marginalise over Z, save Y(A)"
    )
    parser.add_argument(
        "--input", "-i",
        default="yields.csv",
        help="Path to the input CSV file (default: yields.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        default="step1_output.npz",
        help="Path to save the .npz output file (default: step1_output.npz)"
    )
    parser.add_argument(
        "--plot",
        default="step1_mass_yields.png",
        help="Path to save the plot (default: step1_mass_yields.png)"
    )

    args = parser.parse_args()

    print(f"Using input file:  {args.input}")
    print(f"Output data:       {args.output}")
    print(f"Output plot:       {args.plot}\n")

    # Run the pipeline
    A_arr, Y_arr, sigma_arr = load_and_marginalise(args.input)
    plot_mass_yields(A_arr, Y_arr, sigma_arr, output_path=args.plot)
    save_data(A_arr, Y_arr, sigma_arr, output_path=args.output)

    print("\n" + "=" * 60)
    print("STEP 1 COMPLETE")
    print(f"  Output data : {args.output}")
    print(f"  Output plot : {args.plot}")
    print("  Next step   : run step2_forward_model.py")
    print("=" * 60)
