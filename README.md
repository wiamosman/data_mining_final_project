Evaluation: Traditional ML vs. Transformer-Based Models

This project conducts a comparative analysis between traditional machine learning models and modern transformer architectures for sentiment classification. Using the IMDb Large Movie Review Dataset (50,000 polar reviews), we evaluate the trade-offs between predictive accuracy and computational efficiency.
Evaluated Models

    Traditional (TF-IDF): Support Vector Machines (SVM), Logistic Regression (LR), Random Forest (RF), and Naive Bayes (NB).

    Transformers: BERT, RoBERTa, XLNet, and BART.

Performance Summary

The following table summarizes the results obtained on an NVIDIA A100 GPU:
Model	Accuracy	F1 Score	Total Time (s)	Efficiency Rank
XLNet	0.9338	0.9339	4134.58	Low
RoBERTa	0.9334	0.9335	2868.78	Low
BART	0.9292	0.9297	3496.43	Low
BERT	0.9185	0.9192	2826.24	Medium-Low
LR	0.8820	0.8819	6.19	High
SVM	0.8813	0.8805	248.83	Medium
NB	0.8302	0.8196	0.06	Ultra-High
RF	0.8222	0.8230	7633.23	Very Low

Key Findings

    The Accuracy Gap: Transformer-based models (specifically XLNet and RoBERTa) set the performance ceiling with F1 scores exceeding 0.93. They provide superior contextual understanding compared to traditional methods.

    The Efficiency Leader: Logistic Regression and Naive Bayes offer the best performance-to-cost ratio. Logistic Regression achieved ~88% accuracy in just 6 seconds, making it ideal for resource-constrained environments.

    The Random Forest Outlier: Random Forest proved highly inefficient for this task, taking over 127 minutes to run while yielding lower accuracy than simpler linear models due to the high-dimensional sparsity of TF-IDF vectors.

    Conclusion: While Transformers are essential for high-precision tasks, traditional linear models remain highly competitive for large-scale, time-sensitive applications.
