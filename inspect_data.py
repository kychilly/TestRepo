import scanpy as sc
import matplotlib.pyplot as plt


# Generate Visual plots/graphics, might not be needed but just in case
adata = sc.read_h5ad("data/processed/neftel_qc.h5ad")

sc.settings.figdir = "docs/figures"
sc.settings.set_figure_params(dpi=150, facecolor="white")

print("Computing PCA and UMAP projections...")
sc.tl.pca(adata, svd_solver="arpack")
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)

sc.pl.highly_variable_genes(adata, save="_hvgs.png", show=False)

# Plot UMAP colored by QC metrics (e.g. total counts and gene counts)
sc.pl.umap(adata, color=["n_genes_by_counts", "total_counts"], save="_qc_umap.png", show=False)

print("Plots successfully saved to docs/figures/")