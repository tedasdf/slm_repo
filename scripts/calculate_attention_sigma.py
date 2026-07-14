from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add attention entropy-bound sigma columns to an exported diagnostics Excel file."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Excel/CSV file from export_attention_diagnostics_to_excel.py.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output Excel path. Defaults to *_with_sigma.xlsx next to input.",
    )
    args = parser.parse_args()

    import pandas as pd

    src = Path(args.input)
    if src.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if src.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(src, sep=sep)
    else:
        df = pd.read_excel(src)

    required = [
        "attention_logit_multiplier",
        "head_dim",
        "xxt_norm_2_mean",
        "xxt_norm_2_max",
        "wk_wqT_norm_2",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SystemExit(
            "Missing required columns: "
            + ", ".join(missing)
            + "\nRe-export after running with the latest attention diagnostics."
        )

    scale = df["attention_logit_multiplier"] / (df["head_dim"] ** 0.5)
    df["sigma_bound_mean"] = scale * df["xxt_norm_2_mean"] * df["wk_wqT_norm_2"]
    df["sigma_bound_max"] = scale * df["xxt_norm_2_max"] * df["wk_wqT_norm_2"]

    if "score_norm_bound_mean" in df.columns:
        df["sigma_bound_mean_logged_delta"] = (
            df["sigma_bound_mean"] - df["score_norm_bound_mean"]
        )
    if "score_norm_bound_max" in df.columns:
        df["sigma_bound_max_logged_delta"] = (
            df["sigma_bound_max"] - df["score_norm_bound_max"]
        )

    if "seq_len" in df.columns:
        # For a bounded logit range with one dominant score, a simple worst-case
        # entropy lower bound is computed from p_max under a gap of sigma.
        # p_max <= exp(sigma) / (exp(sigma) + n - 1)
        import numpy as np

        for sigma_col, out_col in [
            ("sigma_bound_mean", "entropy_lower_bound_from_sigma_mean"),
            ("sigma_bound_max", "entropy_lower_bound_from_sigma_max"),
        ]:
            sigma = df[sigma_col].astype(float)
            n = df["seq_len"].astype(float)
            exp_s = np.exp(np.minimum(sigma, 700.0))
            p = exp_s / (exp_s + n - 1.0)
            q = (1.0 - p) / (n - 1.0)
            entropy = -(p * np.log(np.clip(p, 1e-300, 1.0)))
            entropy -= (n - 1.0) * q * np.log(np.clip(q, 1e-300, 1.0))
            df[out_col] = entropy
            df[out_col + "_normalized"] = entropy / np.log(n)

    leading = [
        "change",
        "layer",
        "step",
        "sigma_bound_mean",
        "sigma_bound_max",
        "entropy_lower_bound_from_sigma_mean_normalized",
        "entropy_lower_bound_from_sigma_max_normalized",
    ]
    cols = [col for col in leading if col in df.columns]
    cols += [col for col in df.columns if col not in cols]
    df = df[cols]

    out = Path(args.out) if args.out else src.with_name(src.stem + "_with_sigma.xlsx")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out, index=False)

    print(f"Wrote {len(df)} rows to {out}")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
