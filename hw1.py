import numpy as np
import pandas as pd

def generate_pam(x, input_path, output_path):
    # Step 1: Read mutation metric & Align columns to index order
    df = pd.read_csv(input_path, sep=r"\s+", index_col=0, engine="python", comment="#")
    aa_order = df.index.tolist()

    # Step 2: Fixed AA frequencie、
    AA_FREQ = {
        "G": 0.089, "A": 0.087, "L": 0.085, "K": 0.081, "S": 0.070,
        "V": 0.065, "T": 0.058, "P": 0.051, "E": 0.050, "D": 0.047,
        "R": 0.041, "N": 0.040, "F": 0.040, "Q": 0.038, "I": 0.037,
        "H": 0.034, "C": 0.033, "Y": 0.030, "M": 0.015, "W": 0.010
    }
    f = np.array([AA_FREQ[a] for a in aa_order], dtype=float)  # row-wise f_i

    # Step 3: Normalization: Divide by 10,000
    M1 = df.values.astype(float) / 10000.0

    # Step 4: Integer power M^x
    Mx = np.linalg.matrix_power(M1, int(x))

    # Step 5: Non-symmetric scoring with row frequency in the denominator
    # S_ij = round(10*log10(Mx_ij / f_i))
    eps = 1e-300  # prevent log10(0)
    denom_i = np.maximum(f[:, None], 1e-12)
    S = 10.0 * np.log10(np.maximum(Mx / denom_i, eps))
    S = np.rint(S).astype(int)

    # Step 6: Output
    pam_df = pd.DataFrame(S, index=aa_order, columns=aa_order)
    pam_df.to_csv(output_path, sep="\t", index=True)
