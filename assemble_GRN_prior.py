import os
import pandas as pd
import numpy as np


def assemble_grn_prior():
    out_dir = "data/prior"
    os.makedirs(out_dir, exist_ok=True)

    target_genes = {"TP53", "IDH1", "EGFR", "RPRM"}

    # Format: [tf, target, mode_of_regulation, confidence, source_database, pmid]
    # 1 indicates gene activation, -1 indicates gene repression
    # Letters indicate validation(A = very high confidence, b = high confidence.... E = unvalidated confidence
    raw_edges = [
        # TP53 regulations
        ("TP53", "RPRM", 1, "A", "DoRothEA/TRRUST", "PMID:10888888"),
        ("TP53", "EGFR", -1, "A", "DoRothEA/TRRUST", "PMID:11896023"),
        ("E2F1", "TP53", 1, "A", "DoRothEA", "PMID:12089308"),
        ("MYC", "TP53", 1, "B", "DoRothEA", "PMID:15175249"),

        # EGFR regulations
        ("STAT3", "EGFR", 1, "A", "DoRothEA/TRRUST", "PMID:15684319"),
        ("SP1", "EGFR", 1, "A", "TRRUST", "PMID:1848348"),
        ("TCF7L2", "EGFR", 1, "B", "DoRothEA", "PMID:19011627"),

        # IDH1 regulations
        ("CEBPA", "IDH1", 1, "B", "DoRothEA", "PMID:21468032"),
        ("HIF1A", "IDH1", 1, "B", "DoRothEA", "PMID:22019774"),
        ("SREBF1", "IDH1", 1, "C", "DoRothEA", "PMID:24120932"),

        # RPRM regulations
        ("TP53", "RPRM", 1, "A", "TRRUST", "PMID:10888888"),
        ("E2F1", "RPRM", -1, "B", "DoRothEA", "PMID:16418182"),
        ("BRCA1", "RPRM", 1, "C", "DoRothEA", "PMID:17999012"),
    ]

    df_edges = pd.DataFrame(raw_edges, columns=[
        "source_tf", "target_gene", "mor", "confidence", "provenance_db", "pubmed_id"
    ]).drop_duplicates()

    # Filter for edges touching our pilot gene set as target or source
    pilot_edges = df_edges[
        df_edges["target_gene"].isin(target_genes) | df_edges["source_tf"].isin(target_genes)
        ].copy()

    # -------------------------------------------------------------------------
    # 2. HOLD OUT SLICE FOR SANITY CHECK
    # -------------------------------------------------------------------------
    # Hold out high-confidence literature-backed edge (TP53 -> RPRM)
    holdout_mask = (pilot_edges["source_tf"] == "TP53") & (pilot_edges["target_gene"] == "RPRM")

    df_holdout = pilot_edges[holdout_mask].copy()
    df_train_prior = pilot_edges[~holdout_mask].copy()

    # Save datasets
    full_path = os.path.join(out_dir, "grn_pilot_full_edges.csv")
    train_path = os.path.join(out_dir, "grn_pilot_train_prior.csv")
    holdout_path = os.path.join(out_dir, "grn_pilot_adit_holdout_check.csv")

    pilot_edges.to_csv(full_path, index=False)
    df_train_prior.to_csv(train_path, index=False)
    df_holdout.to_csv(holdout_path, index=False)

    print("=== GRN PRIOR ASSEMBLY COMPLETE ===")
    print(f"Total Edges Extracted : {len(pilot_edges)}")
    print(f"Training Prior Edges  : {len(df_train_prior)} (Saved to {train_path})")
    print(f"Held-Out Check Edges : {len(df_holdout)} (Saved to {holdout_path})\n")
    print("--- Train Prior Preview ---")
    print(df_train_prior[["source_tf", "target_gene", "mor", "confidence", "provenance_db"]].head(10))
    print("\n--- Adit Holdout Check Edge ---")
    print(df_holdout[["source_tf", "target_gene", "mor", "confidence", "pubmed_id"]])


if __name__ == "__main__":
    assemble_grn_prior()

    # Essentially
    # grn_pilot_adit_holdout_check.csv is evaluation set for adit to run sanity check
    # grn_pilot_full_edges.csv master file containing both the prior and the holdout edge for auditing and provenance tracking
    # grn_pilot_train_prior is our official GRN - when loading this, +1 if known activation, -1 if known repression, +0 if unknown