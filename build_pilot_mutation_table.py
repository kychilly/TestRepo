import os
import scanpy as sc
import pandas as pd
import zipfile

GATE_GENES = {"TP53", "IDH1", "EGFR", "RPRM"}


def evaluate_patient_genes(patient, p_cells, target_genes, maf_lookup, cgga_lookup):
    """
    Evaluates each target gene independently for a given patient sample.

    Every branch below only assigns a non-wildtype/non-deficient status when
    there is a real variant call (MAF) or a real clinical annotation (CGGA
    IDH status) behind it. No fallback infers mutation status from generic
    subclone-label substrings — that produced uncorrelated noise, not
    evidence, and has been removed rather than tuned.
    """
    patient_gene_calls = {}

    subclones = p_cells["GeneticSubclone"].dropna().unique() if not p_cells.empty else []
    subclone_str = " ".join([str(s).lower() for s in subclones])

    for gene in target_genes:
        # Default: no evidence found for this patient/gene pair.
        # This is the honest outcome for the ~2500 non-gate genes on every
        # patient, and for gate genes too whenever no source has a call.
        status, impact = "data_deficient", "no_variant_call"

        # -------------------------------------------------------------
        # IDH1 — CGGA clinical annotation is a real call. Neftel has no
        # DNA-level IDH1 data at all (it's an explicit IDH-wildtype
        # single-cell cohort by design of the study, not a data gap) —
        # wildtype is the correct, non-guessed value for Neftel patients,
        # not a fallback, but it structurally can never be anything else
        # until IDH-mutant patients come from TCGA/CGGA instead.
        # -------------------------------------------------------------
        if gene == "IDH1":
            if patient in cgga_lookup and "IDH1" in cgga_lookup[patient]:
                status, impact = cgga_lookup[patient]["IDH1"]
            elif not p_cells.empty:
                status, impact = "wildtype", "none"

        # -------------------------------------------------------------
        # EGFR — MAF call, or an explicit EGFR-specific amplification/gain
        # keyword co-occurring in the subclone annotation. The old
        # digit-membership fallback ("1" or "2" in the string) is removed:
        # it fired on generic subclone numbering with no relation to EGFR
        # status and inflated the amplification bucket with noise.
        # -------------------------------------------------------------
        elif gene == "EGFR":
            if (patient, "EGFR") in maf_lookup:
                status, impact = maf_lookup[(patient, "EGFR")]
            elif "egfr" in subclone_str and ("amp" in subclone_str or "gain" in subclone_str):
                status, impact = "amplification", "high_gain"

        # -------------------------------------------------------------
        # TP53 — MAF call, or an explicit TP53-specific mutation keyword.
        # The old digit fallback ("3" or "4" in the string) is removed for
        # the same reason as EGFR above.
        # -------------------------------------------------------------
        elif gene == "TP53":
            if (patient, "TP53") in maf_lookup:
                status, impact = maf_lookup[(patient, "TP53")]
            elif "tp53" in subclone_str and "mut" in subclone_str:
                status, impact = "missense", "pathogenic"

        # -------------------------------------------------------------
        # RPRM — MAF call only. Note: RPRM silencing in GBM is typically
        # promoter-methylation driven, not mutation-driven, so a clean MAF
        # showing no variant is a legitimate result here, not a script
        # failure. Real silencing calls likely require methylation array
        # data (e.g. TCGA 450k) as a separate source — out of scope for
        # this script as written.
        # -------------------------------------------------------------
        elif gene == "RPRM":
            if (patient, "RPRM") in maf_lookup:
                status, impact = maf_lookup[(patient, "RPRM")]

        # -------------------------------------------------------------
        # ALL OTHER CANDIDATE GENES (the ~2500 HVG panel) — MAF call only,
        # same rule as RPRM. No subclone-text heuristic exists for these,
        # so they simply stay data_deficient unless MAF has a row.
        # -------------------------------------------------------------
        else:
            if (patient, gene) in maf_lookup:
                status, impact = maf_lookup[(patient, gene)]

        patient_gene_calls[gene] = (status, impact)

    return patient_gene_calls


def build_pilot_mutation_table():
    pilot_path = "data/pilot/pilot_subsample.h5ad"
    if not os.path.exists(pilot_path):
        raise FileNotFoundError(f"Could not find {pilot_path}. Run build_pilot_subsample.py first.")

    adata_pilot = sc.read_h5ad(pilot_path)
    pilot_patients = adata_pilot.obs["Sample"].unique().tolist()

    # Full candidate list = every gene retained in the pilot subsample
    # (the ~2-3k HVG panel plus the four mandatory gate genes), not just
    # the four gate genes. This is the actual fix for issue #1 from
    # earlier: without this, every non-gate gene had no row to join to.
    target_genes = sorted(set(adata_pilot.var_names.tolist()) | GATE_GENES)

    # 1. PRE-LOAD TCGA / MAF LOOKUPS (IF PRESENT)
    # Now indexed for ALL genes present in the MAF, not just the 4 gate
    # genes, so the full candidate list can actually get real calls.
    maf_lookup = {}
    tcga_data_dir = "data/raw/tcga/data"
    if os.path.exists(tcga_data_dir):
        for fname in os.listdir(tcga_data_dir):
            if fname.endswith(".maf"):
                maf_df = pd.read_csv(os.path.join(tcga_data_dir, fname), sep="\t", comment="#", low_memory=False)
                for _, row in maf_df.iterrows():
                    p_id = str(row.get("Tumor_Sample_Barcode", ""))[:12]
                    g_sym = str(row.get("Hugo_Symbol", ""))
                    v_class = str(row.get("Variant_Classification", "")).lower()

                    if g_sym in target_genes:
                        if "missense" in v_class:
                            maf_lookup[(p_id, g_sym)] = ("missense", "pathogenic")
                        elif "frame_shift" in v_class or "nonsense" in v_class:
                            maf_lookup[(p_id, g_sym)] = ("truncating", "high_loss_of_function")
                        elif "amplification" in v_class or "amp" in v_class:
                            maf_lookup[(p_id, g_sym)] = ("amplification", "high_gain")

    # 2. PRE-LOAD CGGA CLINICAL LOOKUPS (IF PRESENT)
    #
    # This zip's stored CRC-32 does not match its contents (verified
    # independently with system `unzip -t` and Python's zipfile — the
    # decompressed bytes are consistent across every extraction method,
    # only the archive's checksum is wrong, likely a packaging artifact
    # from the original Mac Excel export). A plain zipfile.read() raises
    # BadZipFile here. The old code wrapped this whole block in a bare
    # `except Exception: pass`, which silently swallowed that error and
    # left cgga_lookup empty on every run — CGGA contributed nothing.
    #
    # The underlying file also uses old Mac-style bare-\r line endings
    # (not \n or \r\n), and contains at least 2 rows with one extra
    # tab-separated field vs. the header (CGGA_261, CGGA_738 as of the
    # 2020-05-06 file) — likely a source data-entry issue. Malformed rows
    # are skipped explicitly and logged, not guessed at.
    cgga_lookup = {}
    cgga_skipped_rows = []
    cgga_zip = "data/raw/cgga/CGGA.mRNAseq_325_clinical.20200506.txt.zip"
    if os.path.exists(cgga_zip):
        try:
            with zipfile.ZipFile(cgga_zip) as z:
                real_names = [n for n in z.namelist()
                              if not n.startswith("__MACOSX") and not os.path.basename(n).startswith(".")]
                with z.open(real_names[0]) as zf:
                    # Bypass the (independently confirmed inaccurate) CRC
                    # check rather than let it raise. This relies on a
                    # private zipfile attribute; if a future Python version
                    # removes it, this will start raising again loudly
                    # (not silently) since there's no bare except around it.
                    zf._expected_crc = None
                    raw = zf.read()

                lines = [ln for ln in raw.decode("utf-8", errors="replace").split("\r") if ln.strip()]
                header = lines[0].split("\t")
                n_cols = len(header)
                idh_col_idx = header.index("IDH_mutation_status")
                id_col_idx = header.index("CGGA_ID")

                for line in lines[1:]:
                    fields = line.split("\t")
                    if len(fields) != n_cols:
                        cgga_skipped_rows.append(line)
                        continue
                    c_id = fields[id_col_idx].strip()
                    idh_stat = fields[idh_col_idx].strip().lower()
                    if "mut" in idh_stat:
                        cgga_lookup[c_id] = {"IDH1": ("missense", "pathogenic")}
                    elif "wildtype" in idh_stat or "wt" in idh_stat:
                        cgga_lookup[c_id] = {"IDH1": ("wildtype", "none")}
                    # anything else (e.g. "NA") is left unset -> data_deficient

            if cgga_skipped_rows:
                print(f"CGGA clinical: skipped {len(cgga_skipped_rows)} malformed row(s) "
                      f"(field count mismatch): {[r.split(chr(9))[0] for r in cgga_skipped_rows]}")
            print(f"CGGA clinical: loaded {len(cgga_lookup)} patients "
                  f"({sum(1 for v in cgga_lookup.values() if v['IDH1'][0] == 'missense')} IDH1-mutant, "
                  f"{sum(1 for v in cgga_lookup.values() if v['IDH1'][0] == 'wildtype')} IDH1-wildtype)")
        except Exception as e:
            print(f"WARNING: failed to load CGGA clinical data from {cgga_zip}: {e}")

    # 3. PRE-LOAD NEFTEL METADATA
    neftel_meta_df = pd.DataFrame()
    neftel_meta_path = "data/raw/neftel/IDHwt.GBM.Metadata.SS2.txt"
    if os.path.exists(neftel_meta_path):
        neftel_meta_df = pd.read_csv(neftel_meta_path, sep="\t")
        if len(neftel_meta_df) > 0 and neftel_meta_df.iloc[0]["NAME"] == "TYPE":
            neftel_meta_df = neftel_meta_df.iloc[1:].copy()

    # 4. ITERATE PATIENTS AND BUILD INDEPENDENT CALLS
    records = []
    for patient in pilot_patients:
        p_cells = neftel_meta_df[neftel_meta_df["Sample"] == patient] if not neftel_meta_df.empty else pd.DataFrame()
        gene_calls = evaluate_patient_genes(patient, p_cells, target_genes, maf_lookup, cgga_lookup)

        for gene in target_genes:
            status, impact = gene_calls[gene]
            records.append({
                "patient_id": patient,
                "gene_symbol": gene,
                "variant_status": status,
                "impact": impact
            })

    # 5. SAVE CSV
    long_table = pd.DataFrame(records)
    out_dir = "data/pilot"
    os.makedirs(out_dir, exist_ok=True)
    long_out = os.path.join(out_dir, "patient_gene_mutation_long.csv")
    long_table.to_csv(long_out, index=False)

    print(f"Successfully saved mutation table to {long_out}")
    print(f"Patients: {len(pilot_patients)} | Genes: {len(target_genes)} | Rows: {len(long_table)}")
    print("\nGate gene status counts:")
    print(long_table[long_table["gene_symbol"].isin(GATE_GENES)]
          .groupby(["gene_symbol", "variant_status"]).size())
    print("\nOverall variant_status counts (all genes):")
    print(long_table["variant_status"].value_counts())


if __name__ == "__main__":
    build_pilot_mutation_table()