# Introduction
The purpose of this project is to further evaluate an Uncertainty Quantification (UQ) technique that I developed for my MSCS thesis, Reject Mask (formerly Bayes Mask), on a tabular, time-series, fraud detection data set. 

# Overview
## Data
The data is time series and tabular. It has a lot of missing values (key point of research for this data). There is a primary transactions csv file along with an identity csv file. The identity csv file is meant to be supplemental to the transactions csv file; they join on the column "TransactionID".

Addison Howard, Bernadette Bouchon-Meunier, IEEE CIS, inversion, John Lei, Lynn@Vesta, Marcus2010, and Prof. Hussein Abbass. IEEE-CIS Fraud Detection. https://kaggle.com/competitions/ieee-fraud-detection, 2019. Kaggle.
## Process
I started with baselining the data on XGBoost, NN and NN+RejectMask. Each of these baselines had two flavors, the first is a left join on TransactionID such that transactions df is left and identity df is right and the second is just the transactions df (no supplemental data). The reason for these "flavors" of baselines was to get an idea for whether or not the identity information was necessary. The number of merged columns is 434 and the number of transaction columns is 394 (difference of 40). The reason for potentially excluding the identity information was because all joined columns (from the identity join) have a null percentage of greater than 50%. 174 of the 294 columns in the transaction table also have a null percentage of greater than 50%. Due to the high level of NaNs, the identity information may just not be worth using -- hence two flavors of every experiment to keep track of that hypothesis.