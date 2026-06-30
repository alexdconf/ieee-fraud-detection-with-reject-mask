# Introduction
The purpose of this project is to further evaluate an Uncertainty Quantification (UQ) technique that I developed for my MSCS thesis, Reject Mask (formerly Bayes Mask), on a tabular, time-series, fraud detection data set. 

# Overview
## Data
The data is time series and tabular. It has a lot of missing values (key point of research for this data). There is a primary transactions csv file along with an identity csv file. The identity csv file is meant to be supplemental to the transactions csv file; they join on the column "TransactionID". The timestamp column, "TransactionDT", is monotonically increasing. The target distribution for the training set (train_transactions.csv) has ~3% fraudulent examples, which is a large imbalance.

Addison Howard, Bernadette Bouchon-Meunier, IEEE CIS, inversion, John Lei, Lynn@Vesta, Marcus2010, and Prof. Hussein Abbass. IEEE-CIS Fraud Detection. https://kaggle.com/competitions/ieee-fraud-detection, 2019. Kaggle.

## Process
I trained three models, XGBoost, NN and NN+RejectMask. XGBoost was like a control and the NN for both RejectMask and without RejectMask were the same otherwise, same model same weights, RejectMask is a post-inference capability. I wanted to explore the efficacy of the RejectMask technique on this data set. In order to do this, I chose PR-AUC as the target metric because it's useful for fraud data sets and also because it's a rank metric. This is good because the prediction probabilities for the BDL NN are the focus of the technique but also the prior for the NN is modified and not comparable to the prediction probabilities for the XGBoost model as raw numbers, not an issue for a rank metric, it allows the models to be compared. Several metrics, in addition to PR-AUC were surfaced in the comparison data.

Each of these baselines had two flavors, the first is a left join on TransactionID such that transactions df is left and identity df is right and the second is just the transactions df (no supplemental data). The reason for these "flavors" of baselines was to get an idea for whether or not the identity information was necessary. The number of merged columns is 434 and the number of transaction columns is 394 (difference of 40). The reason for potentially excluding the identity information was because all joined columns (from the identity join) have a null percentage of greater than 50%. 174 of the 394 columns in the transaction table also have a null percentage of greater than 50%. Due to the high level of NaNs, the identity information may just not be worth using -- hence two flavors of every experiment to keep track of that hypothesis.

See the `README.md` for technical implementation details. See the `notes/` for a decision and problem solving blow-by-blow.

I first built the training pipeline, then I built an evaluation script. The `main.py` runs a training job and the evaluation script leverages existing saved models. I am on a 12 Gen Intel i7, GeForce RTX 3060 Mobile (6GB VRAM) and 32GB RAM. For this training configuration It took almost three hours to train both models and RAM was the bottleneck. I didn't go above 40% utilization of the GPU and used less than 2GB VRAM, but essentially all RAM was utilized to the point that I had to keep an eye on swap (8GB) -- Ubuntu 24.04. There is room to improve these models, I just wanted reasonable performance to get more data about the Reject Mask technique, see how it performs on this data set while also trying to characterize what it can do. 

You can find any generated reports and information in the `reports/` directory.

## BDL NN with Reject Mask -- Technique Overview

### Components
ANN with MC-Dropout for BDL

Rule metric for abstention

Performance metric(s)

### Workflow
Given a trained model,

infer a datum

determine if the inferred datum's prediction should be rejected based upon the rule metric

rejected datums should be dealt with appropriately depending upon the system.

# Evaluation and Results
Refer to the data file here `reports/20260625-155136_transactions_only/recompare/risk_coverage.json`. There are other data files, but they are deprecated. The other data files are closer to exploratory benchmarks than experiment data for evaluating the Reject Mask technique.

Coverage is how BALD was discovered as the rule for this data set. The Reject Mask technique is essentially an abstention technique, it excludes samples from prediction (after the fact). This can be framed as testing for efficacy at different coverage levels for the BDL NN model in order to determine the most effective abstention rule.

Once BALD was discovered for this data set for Reject Mask, PR-AUC is the performance metric that is invariant to sample size (abstention alters sample size, so certain metrics can become meaningless in the context of a model rejecting data points before metrics are calculated). Included in those metrics are precision and recall. The main findings of this effort come from looking at precision and recall changes, given the same coverage level and given abstention rules.

# Conclusion
There were two findings for this data set. The first is that the BALD abstention rule works best out of the abstention rules considered. The second finding is that using BALD as a Reject Mask abstention rule for this data set is significant; precision can increase substantially over not having the Reject Mask post-processing step for the BDL NN. When inspecting results between BDL NN (with or without Reject Mask) using non-PR-AUC metrics, `coverage` must be the same.

# Discussion
The entire point of this technique is to improve model performance by calculating whether or not predictions are significant with as little human tuning as possible. Aside from requiring BDL with some other modifications to what is essentially a regular ANN, everything Reject Mask related happens after a prediction.

# Improvements and Next Steps
Try more data sets
Better performing models here, to see if there's any meaningful change in results
More reject metrics and methods to try for new data sets (probably find a new abstention rule per data set)

# Lessons Learned
