###Evaluation Results
Our evaluation on the **IMDb Large Movie Review Dataset** highlights a significant trade-off between model complexity and computational overhead.

###Model Performance Comparison| Model | Accuracy | Precision | Recall | F1 Score | Total Time (s) |
### Model Performance Comparison

<table role="table">
  <thead>
    <tr>
      <th>Model</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1 Score</th>
      <th>Total Time (s)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>XLNet</strong></td>
      <td>0.9338</td>
      <td>0.9321</td>
      <td>0.9357</td>
      <td><strong>0.9339</strong></td>
      <td>4134.58</td>
    </tr>
    <tr>
      <td><strong>RoBERTa</strong></td>
      <td>0.9334</td>
      <td>0.9319</td>
      <td>0.9351</td>
      <td>0.9335</td>
      <td>2868.78</td>
    </tr>
    <tr>
      <td><strong>BART</strong></td>
      <td>0.9292</td>
      <td>0.9228</td>
      <td>0.9366</td>
      <td>0.9297</td>
      <td>3496.43</td>
    </tr>
    <tr>
      <td><strong>BERT</strong></td>
      <td>0.9185</td>
      <td>0.9112</td>
      <td>0.9274</td>
      <td>0.9192</td>
      <td>2826.24</td>
    </tr>
    <tr>
      <td><strong>LR</strong></td>
      <td>0.8820</td>
      <td>0.8826</td>
      <td>0.8811</td>
      <td>0.8819</td>
      <td><strong>6.19</strong></td>
    </tr>
    <tr>
      <td><strong>SVM</strong></td>
      <td>0.8813</td>
      <td>0.8865</td>
      <td>0.8746</td>
      <td>0.8805</td>
      <td>248.83</td>
    </tr>
    <tr>
      <td><strong>NB</strong></td>
      <td>0.8302</td>
      <td>0.8742</td>
      <td>0.7714</td>
      <td>0.8196</td>
      <td><strong>0.06</strong></td>
    </tr>
    <tr>
      <td><strong>RF</strong></td>
      <td>0.8222</td>
      <td>0.8192</td>
      <td>0.8269</td>
      <td>0.8230</td>
      <td>7633.23</td>
    </tr>
  </tbody>
</table>

Summary of Findings* **Transformers (SOTA):** **XLNet** achieved the highest F1 score (0.9339). These models excel at capturing context but require significantly more GPU time (up to 69 minutes).
* **Traditional ML (Efficiency):** **Logistic Regression (LR)** and **Naive Bayes (NB)** are incredibly fast. NB completed the task in 0.06 seconds, making it the most cost-effective for simple applications.
* **The Random Forest Anomaly:** **Random Forest (RF)** was the least efficient model, taking over **127 minutes** while failing to outperform simpler linear models like SVM or LR.
