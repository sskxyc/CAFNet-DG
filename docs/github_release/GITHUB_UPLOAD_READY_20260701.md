# CAFNet-DG GitHub Upload Package (2026-07-01)

This repository has been staged for public release.

## Prepared Local Package

```text
D:\CAFNet-DG-github-release-staging
D:\CAFNet-DG-github-release-staging_20260701.zip
```

Package summary:

```text
Files: 276
Uncompressed size: 295.46 MB
Zip size: 42.86 MB
SHA256: see companion file `D:\CAFNet-DG-github-release-staging_20260701.zip.sha256`
```

The staging directory has been checked for common temporary files and Python cache files. The keyword scan did not identify real credentials; matches such as `secretin`, `secretion`, and `token_set` are biomedical terms or code identifiers.

## Recommended Upload Route

Use a public GitHub repository for the code and curated data package, then attach the zip file as a GitHub Release asset.

Recommended repository name:

```text
CAFNet-DG
```

Recommended first release tag:

```text
v1.0-manuscript
```

Recommended release asset:

```text
D:\CAFNet-DG-github-release-staging_20260701.zip
```

## Git Upload Commands

Run these commands from a machine where Git is installed:

```bash
cd /path/to/CAFNet-DG-github-release-staging
git init
git add .
git commit -m "Initial CAFNet-DG manuscript release"
git branch -M main
git remote add origin https://github.com/<USER_OR_ORG>/CAFNet-DG.git
git push -u origin main
```

If Git LFS is available, initialize it before `git add .`:

```bash
git lfs install
git lfs track "*.mat" "*.npy" "*.npz" "*.pt" "*.pth" "*.model"
git add .gitattributes
```

## GitHub Web Upload Alternative

If command-line Git is unavailable:

1. Create a new GitHub repository named `CAFNet-DG`.
2. Upload the contents of `D:\CAFNet-DG-github-release-staging` through GitHub Desktop or the web interface.
3. Create a release named `v1.0-manuscript`.
4. Attach `D:\CAFNet-DG-github-release-staging_20260701.zip` as the release asset.
5. Copy the repository URL into the manuscript before final submission if the repository is public before submission.

## Files That Should Not Be Uploaded

Do not upload:

```text
submission_flat/
old manuscript snapshots
local conda environments
__pycache__/
*.pyc
*.aux, *.log, *.synctex.gz
temporary training logs
```

## Manuscript Link Placeholder

If the public repository URL is not ready at submission time, keep the manuscript wording as "will be publicly released through a GitHub repository." Once the repository is created, replace that phrase with the actual URL.
