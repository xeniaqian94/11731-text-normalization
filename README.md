# 11731-text-normalization

To Run step 1 and 2:

pip install editdistance

python data/utils.py # currently main() executes Experiment Setup 1 & 2

exec_CRF.sh # For similar example, crfpp-master/example/chunking

TODO List:

Step 1.reading social media abbr list

Step 2.edit distance feature (which takes a bit time to compute) -- we could ignore for now

Step 3 (all) in test_truth_multiline_CAND.pred, normalize CAND token


Other notes:

Re-implement IHS_RD baseline in unconstrained mode?

Baseline: https://groups.google.com/forum/#!topic/lexical-normalisation-for-english-tweets/hVKo0k3PYuQ

Training data:
MD5 (./train_data.json) = 7180cfe1fc751df9a8d772c5b39c188e

Test data:
MD5 (./test_data.json) = 10dfbc18945886935e58b4fc51740277

Ground truth data:
MD5 (./test_truth.json) = 2cc5e518fc50d7da634f9af9c60f508c

Constrained Mode:

MD5 (NCSU_SAS_WOOKHEE.cm.json) = 3c63d1da5d8770fb90ae7ae1e2329f9e

Evaluating NCSU_SAS_WOOKHEE.cm.json

precision: 0.9136

recall:    0.7398

F1:        0.8175

Unconstrained Mode:

MD5 (IHS_RD.um.json) = 0f1814e911cdabfae4a1e2288c8256bc

Evaluating IHS_RD.um.json

precision: 0.8469

recall:    0.8083

F1:        0.8272

