"""Write a simple, mandatory human explanation beside a JSON artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def companion_path(json_path: Path) -> Path:
    return json_path.with_suffix(".txt")


def _items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def explain(payload: Mapping[str, Any], *, source: str) -> str:
    """Explain status, meaning, concerns, and actions without technical shorthand."""
    status = str(payload.get("status", "unknown"))
    if status in {"completed", "passed"}:
        happened = "The job finished and saved its result."
    elif status == "completed_with_blockers":
        happened = "The job did the parts it could do, but some parts could not run."
    elif status in {"blocked", "failed"}:
        happened = "The job could not finish. It stopped instead of making up an answer."
    else:
        happened = f"The file says the job status is {status}."

    reasons = _items(payload.get("reason")) + _items(payload.get("blockers"))
    reasons += _items(payload.get("failures"))
    reasons += _items(payload.get("known_issues"))
    reasons += [
        f"Method did not complete: {name}" for name in _items(payload.get("methods_not_completed"))
    ]
    runs = payload.get("runs")
    if isinstance(runs, list):
        run_reasons = {
            str(run.get("reason")) for run in runs if isinstance(run, Mapping) and run.get("reason")
        }
        reasons += sorted(run_reasons)
    for key in ("feasibility", "comparison"):
        component = payload.get(key)
        if isinstance(component, Mapping) and str(component.get("status", "")).startswith(
            "blocked"
        ):
            reasons.append(f"{key}: {component.get('reason', component.get('status'))}")
    why = reasons or ["The JSON file did not give a special reason."]

    important: list[str] = []
    for key, label in (
        ("completed_runs", "Runs that finished"),
        ("blocked_runs", "Runs that stopped"),
        ("file_count", "Required files found"),
        ("verified_file_count", "Files whose size and fingerprint matched"),
        ("total_bytes", "Total required bytes"),
        ("seeds", "Seeds"),
        ("checkpoint_sha256", "Checkpoint fingerprint"),
        ("vocabulary_sha256", "Vocabulary fingerprint"),
    ):
        if key in payload:
            important.append(f"{label}: {payload[key]}")
    scope = payload.get("backbone_scope")
    if isinstance(scope, Mapping):
        important.append(f"Backbones selected: {scope.get('backbones')}")
        important.append(f"Scope decision: {scope.get('reason')}")
    data = payload.get("data")
    if isinstance(data, Mapping):
        for key, label in (
            ("tcga_samples", "TCGA patients"),
            ("neftel_pseudobulk_samples", "Neftel patient summaries"),
            ("cgga_labeled_samples", "Labeled CGGA patients"),
            ("confirmed_genes", "Confirmed genes"),
            ("unconfirmed_gene_count", "Unconfirmed genes tested"),
        ):
            if key in data:
                important.append(f"{label}: {data[key]}")
    headline = payload.get("headline")
    if isinstance(headline, Mapping):
        important.extend(
            [
                f"Confirmed-gene internal-to-external drop: {headline.get('confirmed_gene_drop')}",
                f"Unconfirmed-gene internal-to-external drop: {headline.get('unconfirmed_gene_drop')}",
                f"Confirmed-gene drop advantage: {headline.get('confirmed_drop_advantage')}",
                f"Shuffled-control drop: {headline.get('shuffled_control_drop')}",
                f"Did confirmed beat shuffled? {headline.get('confirmed_beats_shuffled')}",
                f"Result call: {headline.get('result_call')}",
                str(headline.get("interpretation", "")),
            ]
        )
    proposal = payload.get("proposal_metrics")
    if isinstance(proposal, Mapping):
        for name, value in proposal.items():
            if isinstance(value, Mapping):
                important.append(
                    f"{name}: AUROC={value.get('auroc')}; {value.get('limitation', '')}"
                )
            else:
                important.append(f"{name}: {value}")
    if payload.get("scientific_note"):
        important.append(str(payload["scientific_note"]))
    silhouette = payload.get("silhouette")
    if isinstance(silhouette, Mapping):
        for dimension_key, dimension_label in (
            ("two_dimensions", "2D PCA"),
            ("twenty_dimensions", "20D PCA"),
        ):
            scores = silhouette.get(dimension_key)
            if isinstance(scores, Mapping):
                important.append(
                    f"{dimension_label}: donor silhouette={scores.get('donor')}; "
                    f"state silhouette={scores.get('state')}"
                )
    for key, label in (
        ("n_cells", "Cells measured"),
        ("n_donors", "Donors measured"),
        ("n_cell_assignment_groups", "State groups measured"),
        ("batch_risk_interpretation", "Batch-effect conclusion"),
    ):
        if key in payload:
            important.append(f"{label}: {payload[key]}")
    bucket_counts = payload.get("bucket_counts")
    if isinstance(bucket_counts, Mapping):
        important.append(
            "Outcome buckets: "
            + ", ".join(f"{name}={value}" for name, value in bucket_counts.items())
        )
    feasibility = payload.get("feasibility")
    if isinstance(feasibility, Mapping):
        important.append(
            f"Feasibility branch: {feasibility.get('branch')} "
            f"(decision count {feasibility.get('decision_count')}, "
            f"minimum {feasibility.get('minimum_required')})"
        )
    for key, label in (
        ("code_contract_ready", "Is the code/data contract runnable?"),
        ("full_scientific_matrix_ready", "Is the full scientific matrix ready?"),
    ):
        if key in payload:
            important.append(f"{label} {payload[key]}")
    rows = payload.get("rows")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping) or "method" not in row:
                continue
            message = f"{row['method']}: {row.get('status')}"
            evaluation = row.get("evaluation")
            if isinstance(evaluation, Mapping):
                message += (
                    f"; macro-F1={evaluation.get('macro_f1')}; "
                    f"balanced accuracy={evaluation.get('balanced_accuracy')}"
                )
            if row.get("reason"):
                message += f"; reason={row['reason']}"
            important.append(message)
    if not important:
        important.append("The JSON file is the exact machine-readable record.")

    concerns = _items(payload.get("warnings")) + reasons
    concerns += _items(payload.get("limitations"))
    concerns += [f"Missing: {item}" for item in _items(payload.get("missing"))]
    concerns += [f"Fingerprint mismatch: {item}" for item in _items(payload.get("mismatches"))]
    if payload.get("state_label_warning"):
        concerns.append(str(payload["state_label_warning"]))
    if not concerns and status in {"completed", "passed"}:
        concerns = ["No blocking problem was recorded in this file."]
    elif not concerns:
        concerns = ["Read the JSON for details before using this result."]

    actions = _items(payload.get("next_actions"))
    if not actions:
        if status == "completed":
            actions = ["Check the saved files, then continue to the next planned step."]
        else:
            actions = [
                "Fix the reason listed above, then run the same command again. The job can resume from its checkpoint."
            ]

    def section(title: str, lines: list[str]) -> str:
        return title + "\n" + "\n".join(f"- {line}" for line in lines)

    return (
        "\n\n".join(
            (
                f"PLAIN-ENGLISH EXPLANATION\nJSON file: {source}",
                section("WHAT HAPPENED", [happened]),
                section("WHY", why),
                section("WHAT IS IMPORTANT", important),
                section("WHAT IS CONCERNING", concerns),
                section("NEXT ACTIONS", actions),
            )
        )
        + "\n"
    )


def write_json_with_explanation(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    text_path = companion_path(path)
    text_temporary = text_path.with_suffix(text_path.suffix + ".tmp")
    text_temporary.write_text(explain(payload, source=str(path)), encoding="utf-8")
    text_temporary.replace(text_path)


def write_jsonl_explanation(
    path: Path,
    *,
    row_count: int,
    description: str,
    status: str = "completed",
    next_actions: list[str] | None = None,
) -> None:
    payload = {
        "status": status,
        "row_count": row_count,
        "reason": description,
        "next_actions": next_actions
        or ["Use these rows only with the matching run manifest, seed, and provenance."],
    }
    companion_path(path).write_text(explain(payload, source=str(path)), encoding="utf-8")
