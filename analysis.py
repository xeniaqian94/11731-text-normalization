import json, sys, os
if 'data' not in sys.path:
    sys.path.append('data')
from evaluation import evaluate

pred_file = 'output/valid3.json'
oracle_file = 'data/valid_data.json'

evaluate(pred, oracle)

with open(pred_file,'r') as f:
    pred_list = json.load(f)
with open(oracle_file, 'r') as f:
    oracle_list = json.load(f)
    
for pred, oracle in zip(pred_list, oracle_list):
    try:
        assert(pred["tid"] == oracle["tid"])
        input_tokens = pred["input"]
        pred_rules = pred["rule"]
        pred_tokens = pred["output"]
        oracle_tokens = oracle["output"]
        if pred_tokens == oracle_tokens:
            continue
        logic = None
        for wid in range(len(input_tokens)):
            win, wpred, woracle, rule = input_tokens[wid].lower(),pred_tokens[wid],oracle_tokens[wid],pred_rules[wid]
            if win != woracle and wpred==win: # and rule=='ink':
                logic = wid
        if logic is not None:
            print('------------------------', logic)
            print('input : ', '|'.join(input_tokens))
            print('pred  : ', '|'.join(pred_tokens))
            print('oracle: ', '|'.join(oracle_tokens))
            print('rule:   ', '|'.join(pred_rules))

    except AssertionError:
        print("Invalid data format")
