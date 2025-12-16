Evaluation Results

Our evaluation on the IMDb Large Movie Review Dataset highlights a significant trade-off between model complexity and computational overhead.
<table style="width: 100%; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; border-collapse: collapse;">
<thead>
<tr style="background-color: #f6f8fa; border-bottom: 2px solid #d0d7de;">
<th style="padding: 10px; text-align: left;">Model</th>
<th style="padding: 10px; text-align: center;">Accuracy</th>
<th style="padding: 10px; text-align: center;">Precision</th>
<th style="padding: 10px; text-align: center;">Recall</th>
<th style="padding: 10px; text-align: center;">F1 Score</th>
<th style="padding: 10px; text-align: right;">Total Time (s)</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom: 1px solid #d0d7de;">
<td style="padding: 10px;"><strong>XLNet</strong></td>
<td style="padding: 10px; text-align: center;">0.9338</td>
<td style="padding: 10px; text-align: center;">0.9321</td>
<td style="padding: 10px; text-align: center;">0.9357</td>
<td style="padding: 10px; text-align: center;"><strong>0.9339</strong></td>
<td style="padding: 10px; text-align: right;">4134.58</td>
</tr>
<tr style="border-bottom: 1px solid #d0d7de;">
<td style="padding: 10px;"><strong>RoBERTa</strong></td>
<td style="padding: 10px; text-align: center;">0.9334</td>
<td style="padding: 10px; text-align: center;">0.9319</td>
<td style="padding: 10px; text-align: center;">0.9351</td>
<td style="padding: 10px; text-align: center;">0.9335</td>
<td style="padding: 10px; text-align: right;">2868.78</td>
</tr>
<tr style="border-bottom: 1px solid #d0d7de;">
<td style="padding: 10px;"><strong>BART</strong></td>
<td style="padding: 10px; text-align: center;">0.9292</td>
<td style="padding: 10px; text-align: center;">0.9228</td>
<td style="padding: 10px; text-align: center;">0.9366</td>
<td style="padding: 10px; text-align: center;">0.9297</td>
<td style="padding: 10px; text-align: right;">3496.43</td>
</tr>
<tr style="border-bottom: 1px solid #d0d7de;">
<td style="padding: 10px;"><strong>BERT</strong></td>
<td style="padding: 10px; text-align: center;">0.9185</td>
<td style="padding: 10px; text-align: center;">0.9112</td>
<td style="padding: 10px; text-align: center;">0.9274</td>
<td style="padding: 10px; text-align: center;">0.9192</td>
<td style="padding: 10px; text-align: right;">2826.24</td>
</tr>
<tr style="border-bottom: 1px solid #d0d7de;">
<td style="padding: 10px;"><strong>LR</strong></td>
<td style="padding: 10px; text-align: center;">0.8820</td>
<td style="padding: 10px; text-align: center;">0.8826</td>
<td style="padding: 10px; text-align: center;">0.8811</td>
<td style="padding: 10px; text-align: center;">0.8819</td>
<td style="padding: 10px; text-align: right;"><strong>6.19</strong></td>
</tr>
<tr style="border-bottom: 1px solid #d0d7de;">
<td style="padding: 10px;"><strong>SVM</strong></td>
<td style="padding: 10px; text-align: center;">0.8813</td>
<td style="padding: 10px; text-align: center;">0.8865</td>
<td style="padding: 10px; text-align: center;">0.8746</td>
<td style="padding: 10px; text-align: center;">0.8805</td>
<td style="padding: 10px; text-align: right;">248.83</td>
</tr>
<tr style="border-bottom: 1px solid #d0d7de;">
<td style="padding: 10px;"><strong>NB</strong></td>
<td style="padding: 10px; text-align: center;">0.8302</td>
<td style="padding: 10px; text-align: center;">0.8742</td>
<td style="padding: 10px; text-align: center;">0.7714</td>
<td style="padding: 10px; text-align: center;">0.8196</td>
<td style="padding: 10px; text-align: right;"><strong>0.06</strong></td>
</tr>
<tr style="border-bottom: 1px solid #d0d7de;">
<td style="padding: 10px;"><strong>RF</strong></td>
<td style="padding: 10px; text-align: center;">0.8222</td>
<td style="padding: 10px; text-align: center;">0.8192</td>
<td style="padding: 10px; text-align: center;">0.8269</td>
<td style="padding: 10px; text-align: center;">0.8230</td>
<td style="padding: 10px; text-align: right;">7633.23</td>
</tr>
</tbody>
</table>

Summary of Findings

    Transformers (SOTA): XLNet and RoBERTa achieved the highest F1 scores (0.93+), demonstrating the power of self-attention mechanisms for deep contextual understanding.

    Traditional ML (Efficiency): Logistic Regression (LR) and Naive Bayes (NB) are the efficiency leaders. NB finished in 0.06 seconds, while LR provided a strong balance of 88% accuracy in just 6 seconds.

    The Random Forest Anomaly: Random Forest (RF) was highly inefficient for this text classification task, taking over 127 minutes while failing to outperform the much faster linear models (SVM/LR) due to the high dimensionality of TF-IDF features.
