###Evaluation Results
Our evaluation on the **IMDb Large Movie Review Dataset** highlights a significant trade-off between model complexity and computational overhead.

###Model Performance Comparison| Model | Accuracy | Precision | Recall | F1 Score | Total Time (s) |
| --- | --- | --- | --- | --- | --- |
| **XLNet** | 0.9338 | 0.9321 | 0.9357 | **0.9339** | 4134.58 |
| **RoBERTa** | 0.9334 | 0.9319 | 0.9351 | 0.9335 | 2868.78 |
| **BART** | 0.9292 | 0.9228 | 0.9366 | 0.9297 | 3496.43 |
| **BERT** | 0.9185 | 0.9112 | 0.9274 | 0.9192 | 2826.24 |
| **LR** | 0.8820 | 0.8826 | 0.8811 | 0.8819 | **6.19** |
| **SVM** | 0.8813 | 0.8865 | 0.8746 | 0.8805 | 248.83 |
| **NB** | 0.8302 | 0.8742 | 0.7714 | 0.8196 | **0.06** |
| **RF** | 0.8222 | 0.8192 | 0.8269 | 0.8230 | 7633.23 |

###Summary of Findings* **Transformers (SOTA):** **XLNet** achieved the highest F1 score (0.9339). These models excel at capturing context but require significantly more GPU time (up to 69 minutes).
* **Traditional ML (Efficiency):** **Logistic Regression (LR)** and **Naive Bayes (NB)** are incredibly fast. NB completed the task in 0.06 seconds, making it the most cost-effective for simple applications.
* **The Random Forest Anomaly:** **Random Forest (RF)** was the least efficient model, taking over **127 minutes** while failing to outperform simpler linear models like SVM or LR.
