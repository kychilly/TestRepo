import anndata as ad
from validator import GeneRecord, Thresholds, classify, Outcome

# 1. We put the EXACT name of your file from the screenshot here
file_path = "pilot_subsample.h5ad"
data = ad.read_h5ad(file_path)

# 2. We explicitly tell the script to look for validator.yaml right here in the main folder
# (This bypasses the error where it searches for a non-existent 'config' folder)
thresholds = Thresholds.from_yaml("validator.yaml")

# 3. Pull out the table of genes from the binder
gene_table = data.var

total_genes = 0
data_deficient_count = 0

# 4. Look at every single gene in the binder
for gene_name, row in gene_table.iterrows():
    total_genes += 1
    
    # Extract protein scores if present in the columns (defaults to None if missing)
    record = GeneRecord(
        gene=str(gene_name),
        mutation=row.get("mutation", "unknown"),
        alteration_type=row.get("alteration_type", "missense"),
        plddt=row.get("plddt", None),
        esm1b=row.get("esm1b", None),
        ddg=row.get("ddg", None)
    )
    
    # Pass the gene through validator.py
    verdict = classify(record, thresholds)
    
    # Check if it landed in the data_deficient bucket
    if verdict.outcome == Outcome.DATA_DEFICIENT:
        data_deficient_count += 1

# 5. Calculate and print the percentage safely
if total_genes > 0:
    percent_deficient = (data_deficient_count / total_genes) * 100
else:
    percent_deficient = 0

print("--- COVERAGE AUDIT RESULTS ---")
print(f"Total candidate genes: {total_genes}")
print(f"Data-deficient genes: {data_deficient_count}")
print(f"Deficiency percentage: {percent_deficient:.2f}%")