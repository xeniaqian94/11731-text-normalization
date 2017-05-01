'''
    File name: utils.py
    Author: xin
    Date created: 4/29/17 3:35 PM
'''

# !/usr/bin/env python
import sys
import argparse

import re
from collections import defaultdict
import numpy as np
import editdistance
import itertools


try:
    import ujson as json
except ImportError:
    import json


def create_feature_and_CANDlabel(file_input,norm_dict,canonical_dict):
    output = open(file_input.split(".")[0] + "_CAND", "w")

    pred_list = json.load(open(file_input))
    count=0
    norm_dict_split=[norm_word.split() for norm_word in norm_dict] # "A L W A Y S"
    norm_dict_subword = list(itertools.chain(*norm_dict_split))  # A L W A Y S
    for pred in pred_list:
        count+=1
        print count*1.0/len(pred_list)
        input_tokens = pred["input"]
        output_tokens = pred["output"]
        sent_length = len(input_tokens)
        for token_ind in range(sent_length):
            this_token=input_tokens[token_ind].lower()
            output_token=output_tokens[token_ind].lower()
            tag = False
            alphanumeric = True
            in_normalized_lexicon=False
            in_canonical_dict=False
            if this_token!= output_token:
                tag = True
            if not re.match('^[a-zA-Z0-9]+[a-zA-Z0-9\']*$', this_token):
                alphanumeric = False
            if this_token in norm_dict_subword:
                in_normalized_lexicon=True
            if this_token in canonical_dict:
                in_canonical_dict=True
            word_length=len(this_token)
            num_vowels = 0
            vowels = set("aeiou")
            for letter in this_token:
                if letter in vowels:
                    num_vowels += 1

            # edit_distance_within2=(np.asarray([editdistance.eval(word, this_token) for word in canonical_dict])<=2).any()

            # features = [str(alphanumeric),str(in_normalized_lexicon),str(in_canonical_dict),str(word_length),str(num_vowels),str(edit_distance_within2)]
            features = [str(alphanumeric), str(in_normalized_lexicon), str(in_canonical_dict), str(word_length),
                        str(num_vowels)]
            feature_string = this_token
            for feat_ind in range(len(features)):
                feature_string += " " + features[feat_ind]
            feature_string += " " + ("CAND" if tag else "NOT_CAND")
            output.write(feature_string + "\n")
        output.write("\n")
    output.close()


def training_data_normalization_lexicon(training_file):
    pred_list = json.load(open(training_file))
    norm_dict = defaultdict(lambda: defaultdict(lambda: 0))
    line_count = 0

    for pred in pred_list:
        line_count += 1

        input_tokens = pred["input"]
        output_tokens = pred["output"]
        sent_length = len(input_tokens)
        i = 0
        while i < sent_length:
            if input_tokens[i].lower() != output_tokens[i].lower() and (re.match('^[a-zA-Z0-9]+[a-zA-Z0-9\']*$',
                                                                                input_tokens[i]) is not None):
                low = i
                high = i+1
                while high < sent_length:
                    if input_tokens[high].lower() != output_tokens[high].lower() and re.match(
                            '^[a-zA-Z0-9]+[a-zA-Z0-9\']*$',
                            input_tokens[high]) and output_tokens[high] == "":
                        high = high + 1
                    else:
                        break
                i = max(high, low + 1)
                if (i-low)>1:
                    # print input_tokens[oldi-1].lower()
                    # print output_tokens[oldi-1].lower()
                    # print (input_tokens[oldi-1].lower() != output_tokens[oldi-1].lower() and (re.match('^[a-zA-Z0-9]+[a-zA-Z0-9\']*$',
                    #                                                             input_tokens[oldi-1]) is not None))
                    print " ".join(input_tokens[low:i])+" -> "+output_tokens[low]
                    # print pred

                norm_dict[" ".join(input_tokens[low:i]).lower()][output_tokens[low].lower()] += 1

            else:
                i = i + 1
    print "Many to one mapping"
    norm_file = open(training_file.split(".")[0] + "_dict", "w")
    for input in norm_dict.keys():
        for output in norm_dict[input].keys():
            norm_file.write(input + "\t" + output + "\t" + str(norm_dict[input][output]) + "\n")
    norm_file.close()
    return norm_dict



def read_incremental_dict(file, norm_dict):

    print "Before "+file+" Original norm dict size "+str(len(norm_dict.keys()))
    f = open(file,"r")
    for line in f.readlines():
        norm_dict[line.split("\t")[0]][line.split("\t")[1]]=int(line.split("\t")[2])
    f.close()
    print "After "+file+" New norm dict size " + str(len(norm_dict.keys()))
    return norm_dict


def main():
    parser = argparse.ArgumentParser(description="Generate CRF CAND training data (in original order)")
    parser.add_argument("--train_file", default="train_data_multiline.json")
    parser.add_argument("--test_truth",default="test_truth_multiline.json")

    args = parser.parse_args()

    norm_dict=training_data_normalization_lexicon(args.train_file)
    norm_dict=read_incremental_dict("../resource/Han2011_clean.txt",norm_dict)
    norm_dict = read_incremental_dict("../resource/Liu2011_clean.txt", norm_dict)
    canonical_dict=[unicode(line.strip(), errors='ignore')  for line in open('../resource/canonical_dict')]
    # create_feature_and_CANDlabel(args.train_file,norm_dict,canonical_dict)
    create_feature_and_CANDlabel(args.test_truth, norm_dict, canonical_dict)


if __name__ == "__main__":
    main()
