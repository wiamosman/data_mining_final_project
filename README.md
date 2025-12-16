📊 Evaluation: Traditional ML vs. Transformers

This project provides a comprehensive comparative analysis of sentiment classification using the IMDb Large Movie Review Dataset (50,000 polar reviews). We evaluate the trade-offs between predictive accuracy and computational efficiency across two distinct modeling paradigms.
Evaluated Models

    Traditional ML (TF-IDF): Naive Bayes, Logistic Regression, Support Vector Machines (SVM), and Random Forest.

    Transformer-Based: BERT, RoBERTa, XLNet, and BART.

Key Results

The study reveals a significant performance-efficiency gap. While Transformers achieve state-of-the-art accuracy, traditional linear models offer massive speed advantages.
Model Class	Top Performer	F1 Score	Total Time (Training + Inference)
Transformer	XLNet	0.9339	~68.9 minutes
Traditional ML	Logistic Regression	0.8819	~6.2 seconds
Major Findings

    The Accuracy Leader: XLNet and RoBERTa achieved the highest F1 scores (>0.93), outperforming traditional methods by approximately 5–11%.

    The Efficiency King: Logistic Regression and Naive Bayes are remarkably efficient. Logistic Regression provides a strong baseline (0.88 F1) in a fraction of the time required for deep learning.

    The Random Forest Outlier: Random Forest proved unsuitable for this task, taking the longest time to train (~127 minutes) while yielding lower accuracy (0.82 F1) due to the high-dimensional sparsity of the TF-IDF features.

    The Trade-off: Model selection should be driven by resource constraints. Transformers are ideal for high-precision needs, while Logistic Regression remains the most cost-effective choice for large-scale, real-time applications.
