# GitHub Release Checklist

Use this checklist before pushing the public repository.

## 1. Repository Hygiene

- [ ] Confirm `.local_pkgs/`, `.conda_envs/`, `.idea/`, `__pycache__/`, and old `submission_flat/` snapshots are not committed.
- [ ] Confirm trained checkpoints (`*.pt`, `*.pth`, `*.model`) are not committed unless intentionally uploaded through LFS or a release artifact.
- [ ] Confirm root `README.md` describes CAFNet-DG rather than the older A3Net-only project.
- [ ] Confirm `environment.yml` is present.
- [ ] Confirm `.gitattributes` is configured before adding large files.

## 2. Data

- [ ] Upload benchmark data in `data/`.
- [ ] Track `.mat` and `.npy` files through Git LFS or publish them as release assets.
- [ ] Include `drug_SMILES_750.csv`, `drug_SMILES_750.txt`, and `frequency_data.txt`.
- [ ] Include `mask_mat_750.mat`, `blind_mask_mat_750.mat`, and `scaffold_mask_mat_750.mat`.

## 3. Code

- [ ] Upload `Net.py`, `utils.py`, `vector.py`, `cold-scence.py`, and `warm-scence.py`.
- [ ] Upload analysis scripts needed to regenerate tables.
- [ ] Upload external-validation scripts under `analysis_outputs/cafnet_dg_external_validation/`.
- [ ] Upload per-drug, hot-side-effect, and statistics analysis scripts.

## 4. Result Data

- [ ] Upload main cold-start predictions for CAFNet, A3Net, CAFNet-D, and CAFNet-DG.
- [ ] Upload baseline result folders for RF, XGBoost, HSTrans, and global popularity.
- [ ] Upload curated analysis outputs in `analysis_outputs/statistics_jbhi/`.
- [ ] Upload CAFNet-DG hot-side-effect, per-drug, and external-validation summaries.
- [ ] Upload large OFFSIDES scored-pair CSVs through Release/LFS, not ordinary Git.

## 5. Documentation

- [ ] Include `docs/github_release/UPLOAD_MANIFEST.md`.
- [ ] Include `docs/github_release/RESULT_DATA_MANIFEST.md`.
- [ ] Include this checklist.
- [ ] Add a release note explaining which files are required for table reproduction and which are optional large artifacts.

## 6. Final Verification Commands

```powershell
git lfs install
git lfs track "*.mat"
git lfs track "*.npy"
git lfs track "*.npz"
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.model"
git status
```

Optional integrity checks:

```powershell
Get-FileHash data/raw_frequency_750.mat -Algorithm SHA256
Get-FileHash result_ICS/10cafnet_dg_ensemble06_cafnetd04_cafnet/blind_pred.csv -Algorithm SHA256
```
