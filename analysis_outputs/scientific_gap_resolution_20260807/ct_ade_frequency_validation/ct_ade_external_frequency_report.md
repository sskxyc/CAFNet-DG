# CT-ADE external ordinal-frequency diagnostics

{
  "source": "CT-ADE-PT public test_frequencies split",
  "source_url": "https://huggingface.co/datasets/anthonyyazdaniml/CT-ADE-PT",
  "source_sha256": "21adb9ad1d9049796b1e35ba1880f40539771f9ba5222d79603a103edef09a13",
  "mapped_trial_groups": 254,
  "exact_side_terms": 982,
  "non_sider_nonzero_frequency_pairs": 2781,
  "mapped_drugs_with_frequency_pairs": 30,
  "aggregation": "at-risk weighted mean trial proportion by drug-side-effect pair",
  "class_bands": {
    "1": "<0.01%",
    "2": "0.01-0.1%",
    "3": "0.1-1%",
    "4": "1-10%",
    "5": ">=10%"
  },
  "claim_boundary": "external ordinal calibration diagnostic; no patient-specific or causal interpretation"
}

```text
             model    n      qwk  within_one_accuracy  exact_accuracy  spearman     rmse      mae
CAFNet-D_frequency 2781 0.287838             0.755124        0.268968  0.346791 1.390939 1.126012
Training-side-mean 2781 0.261245             0.893563        0.405250  0.360013 0.900674 0.727174
```