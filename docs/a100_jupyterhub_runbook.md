# A100 JupyterHub runbook

The prior version of this page described an expired allocation and an obsolete
streaming-only workflow. Use the maintained, dataset-specific procedure in
[team_week2_to_week4_execution.md](team_week2_to_week4_execution.md), section 5.

The A100 run must use an isolated torch 2.3.x / torchtext 0.18.x environment,
the local 6,576-cell Neftel H5AD, the frozen patient split, checkpoint-matched
`args.json`, and persistent checkpoint/output storage.
