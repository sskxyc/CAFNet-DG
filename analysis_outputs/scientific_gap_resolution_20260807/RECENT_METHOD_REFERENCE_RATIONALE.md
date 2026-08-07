# Recent-method reference rationale

## Rare and novel-drug ADR modeling

- *Informative relational learning for adverse reaction prediction with
  enhanced generalization to novel drugs* (Bioinformatics, 2026) combines ADR
  hierarchy/co-occurrence, drug structure and ATC information, a dual
  mixture-of-experts module, and domain adaptation. Its most transferable idea
  for the present single-modality benchmark is expert specialization with
  explicit routing diagnostics. The ATC, MolFormer, ADReCS hierarchy, and
  target-domain inputs are not silently imported because they change the input
  information budget.
  <https://academic.oup.com/bioinformatics/article/42/7/btag494/8723702>
- DeepADR (Briefings in Bioinformatics, 2025) and DSNet (Knowledge-Based
  Systems, 2025) motivate multimodal and dual-graph future extensions, but they
  are not directly inserted into the current model because their inputs and
  task definitions differ from the frozen 750 x 994 benchmark.
  <https://academic.oup.com/bib/article/26/6/bbaf695/8408361>
  <https://doi.org/10.1016/j.knosys.2025.113537>

## Independent validation

- CT-ADE (Scientific Data, 2025) provides controlled monopharmacy clinical
  trial labels, explicit negative cases, and raw event proportions. It supports
  independent association-prioritization and ordinal-frequency diagnostics,
  while its patient/regimen context remains outside CAFNet-DG's inputs.
  <https://doi.org/10.1038/s41597-025-04718-1>
- OnSIDES (Med, 2025; official v3.1.1 release in 2026) provides international
  regulatory-label ADR pairs. Exact drug/PT mapping and SIDER-overlap removal
  are required because the resource is label-derived and automatically
  extracted.
  <https://doi.org/10.1016/j.medj.2025.100642>
  <https://doi.org/10.5281/zenodo.19701431>
- FDA quarterly AEMS/FAERS safety-signal pages provide a temporal regulatory
  stress test. FDA explicitly states that listing is not a causal conclusion;
  these entries must be described as potential signals only.
  <https://www.fda.gov/drugs/fda-adverse-event-monitoring-system-aems/new-safety-information-or-potential-signals-serious-risks-identified-fda-adverse-event-monitoring>

## Reproducibility

- PyTorch documents that deterministic-algorithm mode can substitute supported
  kernels or raise an error for unsupported operations, and that the switch
  alone does not guarantee reproducibility across releases or platforms. The
  present audit therefore records strict-mode support, repeated-output hashes,
  and numerical differences instead of promising universal bitwise identity.
  <https://docs.pytorch.org/docs/stable/notes/randomness.html>
  <https://docs.pytorch.org/docs/main/generated/torch.use_deterministic_algorithms.html>
