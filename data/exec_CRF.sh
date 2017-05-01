#!/usr/bin/env bash

../crfpp-master/crf_learn -c 4.0 crf_template train_data_multiline_CAND crf_model
../crfpp-master/crf_test -m crf_model test_truth_multiline_CAND > test_truth_multiline_CAND.pred