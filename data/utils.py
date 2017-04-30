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

try:
    import ujson as json
except ImportError:
    import json

feature_num = 0


def create_feature(file_input):
    output = open(file_input.split(".")[0] + "_CAND", "w")

    pred_list = json.load(open(file_input))

    for pred in pred_list:
        input_tokens = pred["input"]
        output_tokens = pred["output"]
        sent_length = len(input_tokens)
        for i in range(sent_length):
            tag = False
            alphanumeric = True
            if input_tokens[i].lower() != output_tokens[i].lower():
                tag = True
            if not re.match('^[a-zA-Z0-9]+[a-zA-Z0-9\']*$', input_tokens[i]):
                alphanumeric = False
            features = [str(alphanumeric), ]
            feature_string = input_tokens[i]
            for i in range(feature_num):
                feature_string += " " + features[i]
            feature_string += " " + " " + ("CAND" if tag else "NOT_CAND")
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
            if input_tokens[i].lower() != output_tokens[i].lower() and re.match('^[a-zA-Z0-9]+[a-zA-Z0-9\']*$',
                                                                                input_tokens[i]):
                low = i
                high = i
                while high < sent_length:
                    if input_tokens[high].lower() != output_tokens[high].lower() and re.match(
                            '^[a-zA-Z0-9]+[a-zA-Z0-9\']*$',
                            input_tokens[high]) and output_tokens[high] == "":
                        high = high + 1
                    else:
                        break
                i = max(high, low + 1)
                if (i-low)>1:
                    print " ".join(input_tokens[low:i])+" "+output_tokens[low]
                    print pred

                norm_dict[" ".join(input_tokens[low:i])][output_tokens[low]] += 1



            else:
                i = i + 1
    norm_file = open(training_file.split(".")[0] + "_dict", "w")
    for input in norm_dict.keys():
        for output in norm_dict[input].keys():
            norm_file.write(input + "\t" + output + "\t" + str(norm_dict[input][output]) + "\n")
    norm_file.close()


def main():
    parser = argparse.ArgumentParser(description="Generate CRF CAND training data (in original order)")
    parser.add_argument("--file", default="train_data_multiline.json")
    args = parser.parse_args()

    # evaluate(args.file)
    training_data_normalization_lexicon(args.file)


if __name__ == "__main__":
    main()
