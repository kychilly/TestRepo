from __future__ import annotations

from pathlib import Path

import pytest

from scripts.build_stage34_inputs_from_combined_data import _protein_evidence


def test_protein_evidence_parses_nullable_scores(tmp_path: Path) -> None:
    path = tmp_path / "protein.csv"
    path.write_text(
        "gene,mutation,plddt,esm1b,ddg,evidence_source\ntp53,R175H,92.1,-8.2,,AlphaFold DB 2025\n",
        encoding="utf-8",
    )
    rows = _protein_evidence(path)
    assert rows["TP53"]["plddt"] == 92.1
    assert rows["TP53"]["esm1b"] == -8.2
    assert rows["TP53"]["ddg"] is None


def test_protein_evidence_rejects_ambiguous_gene_variants(tmp_path: Path) -> None:
    path = tmp_path / "protein.jsonl"
    path.write_text(
        '{"gene":"TP53","mutation":"R175H","plddt":90,"esm1b":-8,"ddg":2}\n'
        '{"gene":"TP53","mutation":"R248Q","plddt":91,"esm1b":-9,"ddg":3}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="multiple rows"):
        _protein_evidence(path)
