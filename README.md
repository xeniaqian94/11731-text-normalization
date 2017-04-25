# 11731-text-normalization

TODO List:

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
F1:        0.8175
MD5 (NCSU_SAS_SAM.cm.json) = 460c2034107a0d8372730cdf9649001c
Evaluating NCSU_SAS_SAM.cm.json
precision: 0.9012
recall:    0.7437
F1:        0.8149
MD5 (iitp.cm.json) = f3d64e8949e23041d5b30906fc9c78d8
Evaluating iitp.cm.json
precision: 0.9026
recall:    0.7191
F1:        0.8005
MD5 (DCU-Adapt.cm.json) = 76435acf2d30c825c698025b73ecd179
Evaluating DCU-Adapt.cm.json
precision: 0.819
recall:    0.5509
F1:        0.6587
MD5 (lysgroup.cm.json) = 2b65c4fad42653e9bed93c21b92838e1
Evaluating lysgroup.cm.json
precision: 0.4646
recall:    0.6281
F1:        0.5341

Unconstrained Mode:
MD5 (IHS_RD.um.json) = 0f1814e911cdabfae4a1e2288c8256bc
Evaluating IHS_RD.um.json
precision: 0.8469
recall:    0.8083
F1:        0.8272
MD5 (USZEGED.um.json) = b8494566913f7dcd9223c5df238fb7c7
Evaluating USZEGED.um.json
precision: 0.8606
recall:    0.7564
F1:        0.8052
MD5 (bekli.um.json) = 257e128bec63dcbe79e252ee35e69f0d
Evaluating bekli.um.json
precision: 0.7732
recall:    0.7416
F1:        0.7571
MD5 (gigo.um.json) = 4b8b1218037b51525bdbee31e8aa012e
Evaluating gigo.um.json
precision: 0.7593
recall:    0.6963
F1:        0.7264
MD5 (lysgroup.um.json) = d68b3658d3f0b33462afcfa43ce486d6
Evaluating lysgroup.um.json
precision: 0.4592
recall:    0.6296
F1:        0.531

Cheers,
Bo
