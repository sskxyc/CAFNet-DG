# CT-ADE-PT independent clinical-trial validation

- Exact structure mapping: 31 drugs across 254 trial groups.
- Exact MedDRA-PT mapping: 982/994 side effects.
- After SIDER exclusion: 1336 positives over 29 drugs.
- The outcome is CT-ADE's clinically significant >=1% label, not the original 1--5 ordinal-frequency target.

```text
            model metric   pooled  per_drug_mean  per_drug_std  n_drugs
           CAFNet  AUROC 0.712352       0.785899      0.145456       29
           CAFNet   AUPR 0.168940       0.210916      0.174895       29
         CAFNet-D  AUROC 0.748266       0.829793      0.114531       29
         CAFNet-D   AUPR 0.190013       0.270481      0.236252       29
        CAFNet-DG  AUROC 0.754449       0.835760      0.107114       29
        CAFNet-DG   AUPR 0.198570       0.272774      0.240433       29
Global popularity  AUROC 0.740768       0.811981      0.106419       29
Global popularity   AUPR 0.170402       0.239269      0.181799       29
```

## Prevalence-controlled results

```text
        scope             model metric   pooled  per_drug_mean  per_drug_std  n_evaluable_drugs  n_pairs  n_positive
all_non_sider            CAFNet  AUROC 0.712352       0.785899      0.145456                 29    28874        1336
all_non_sider            CAFNet   AUPR 0.168940       0.210916      0.174895                 29    28874        1336
all_non_sider          CAFNet-D  AUROC 0.748266       0.829793      0.114531                 29    28874        1336
all_non_sider          CAFNet-D   AUPR 0.190013       0.270481      0.236252                 29    28874        1336
all_non_sider         CAFNet-DG  AUROC 0.754449       0.835760      0.107114                 29    28874        1336
all_non_sider         CAFNet-DG   AUPR 0.198570       0.272774      0.240433                 29    28874        1336
all_non_sider Global popularity  AUROC 0.740768       0.811981      0.106419                 29    28874        1336
all_non_sider Global popularity   AUPR 0.170402       0.239269      0.181799                 29    28874        1336
         rare            CAFNet  AUROC 0.559891       0.569021      0.187211                 20     9950         174
         rare            CAFNet   AUPR 0.034871       0.068697      0.071206                 20     9950         174
         rare          CAFNet-D  AUROC 0.602450       0.624275      0.223856                 20     9950         174
         rare          CAFNet-D   AUPR 0.028118       0.071975      0.077244                 20     9950         174
         rare         CAFNet-DG  AUROC 0.610037       0.632929      0.227474                 20     9950         174
         rare         CAFNet-DG   AUPR 0.029165       0.070756      0.075621                 20     9950         174
         rare Global popularity  AUROC 0.530673       0.471137      0.205664                 20     9950         174
         rare Global popularity   AUPR 0.020828       0.037109      0.040994                 20     9950         174
       middle            CAFNet  AUROC 0.578788       0.644252      0.209705                 20     9992         270
       middle            CAFNet   AUPR 0.056709       0.097212      0.119554                 20     9992         270
       middle          CAFNet-D  AUROC 0.615878       0.642677      0.216903                 20     9992         270
       middle          CAFNet-D   AUPR 0.061678       0.117713      0.126086                 20     9992         270
       middle         CAFNet-DG  AUROC 0.620154       0.664972      0.184886                 20     9992         270
       middle         CAFNet-DG   AUPR 0.070829       0.121497      0.146659                 20     9992         270
       middle Global popularity  AUROC 0.571286       0.604800      0.180639                 20     9992         270
       middle Global popularity   AUPR 0.037875       0.068190      0.068756                 20     9992         270
     frequent            CAFNet  AUROC 0.689910       0.743315      0.170436                 29     8932         892
     frequent            CAFNet   AUPR 0.229363       0.264476      0.215966                 29     8932         892
     frequent          CAFNet-D  AUROC 0.706791       0.779776      0.127528                 29     8932         892
     frequent          CAFNet-D   AUPR 0.252848       0.320022      0.267352                 29     8932         892
     frequent         CAFNet-DG  AUROC 0.713664       0.784969      0.135169                 29     8932         892
     frequent         CAFNet-DG   AUPR 0.262232       0.319238      0.269466                 29     8932         892
     frequent Global popularity  AUROC 0.671668       0.744102      0.133401                 29     8932         892
     frequent Global popularity   AUPR 0.221102       0.285596      0.219882                 29     8932         892
```

```text
                    scope             model      metric   pooled  per_drug_mean  per_drug_std  n_evaluable_drugs  n_pairs  n_positive
prevalence_matched_1_to_5            CAFNet       AUROC 0.649060       0.611030      0.165107                 29     8011        1336
prevalence_matched_1_to_5            CAFNet        AUPR 0.295789       0.335634      0.161429                 29     8011        1336
prevalence_matched_1_to_5          CAFNet-D       AUROC 0.672612       0.643384      0.188840                 29     8011        1336
prevalence_matched_1_to_5          CAFNet-D        AUPR 0.317175       0.386711      0.206545                 29     8011        1336
prevalence_matched_1_to_5         CAFNet-DG       AUROC 0.678956       0.622887      0.169791                 29     8011        1336
prevalence_matched_1_to_5         CAFNet-DG        AUPR 0.321081       0.349347      0.165779                 29     8011        1336
prevalence_matched_1_to_5 Global popularity       AUROC 0.619877       0.571222      0.134423                 29     8011        1336
prevalence_matched_1_to_5 Global popularity        AUPR 0.266575       0.301265      0.106236                 29     8011        1336
prevalence_matched_1_to_5            CAFNet concordance 0.659780       0.625805      0.174672                 29     8011        1336
prevalence_matched_1_to_5          CAFNet-D concordance 0.719960       0.663858      0.223495                 29     8011        1336
prevalence_matched_1_to_5         CAFNet-DG concordance 0.723403       0.649629      0.185033                 29     8011        1336
prevalence_matched_1_to_5 Global popularity concordance 0.540469       0.534599      0.127917                 29     8011        1336
```

## Interpretation boundary

This analysis is an independent controlled-monopharmacy clinical-trial prioritization test. Because CAFNet-DG has no patient or regimen input, repeated trial groups are aggregated by drug. The result must not be described as patient-specific prediction, calibrated incidence, or causal confirmation.