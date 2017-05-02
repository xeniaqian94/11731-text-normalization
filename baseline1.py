
# coding: utf-8

# In[2]:

import json, copy, requests, pickle, re
from nltk.corpus import words
from multiprocessing import Pool
from collections import Counter, defaultdict
from autocorrect import spell


# In[3]:
which = 'valid'
output_fn = 'myoutput'

if which == 'valid':
    train_prefix, test_prefix = 'train', 'valid'
elif which == 'test':
    train_prefix, test_prefix = 'train_valid', 'test'
    

with open('data/'+test_prefix+'_data.json','r') as f:
    test_in = json.load(f)


# In[4]:

wordlist = words.words()



with open('./resource/pairs_'+train_prefix+'.pickle', 'rb') as f:
    pairs = pickle.load( f)



with open('resource/Han2011.txt', 'r') as f:
    lines = f.readlines()
han2011 = {}
for l in lines:
    k, v = l.strip().split()
    if re.match('[0-9]+', k) is  None:  
        # Han201.txt includes unecessary modifications, e.g. 20million -> million
        han2011[k]=v


# In[8]:

with open('resource/Liu2011.txt', 'r') as f:
    lines = f.readlines()
liu2011 = {}
for l in lines:
    l = l.strip().split()
    l = [ll for ll in l if ll!='|']
    freq, key, vs = int(l[0]), l[1], l[2:]
    liu2011[key]=vs


# In[ ]:




# In[9]:

# baseline 2 = baseline 1 + Liu2011 + Han2011
exclude_pattern = ['^@', '^#', '^http', ]
candidate = defaultdict(list)
excluded = []
output_data = copy.deepcopy(test_in)
for d in output_data:
    out, rule = [], []
    for wid, w in enumerate(d['input']):
        w = w.lower()
        exclude = False
        for pattern in exclude_pattern:
            if re.match(pattern, w) is not None:
                exclude=True
                break
        if exclude:
            #excluded.append((w, d['output'][wid]))
            out.append(w)
            rule.append('exl')
            continue
        if w in pairs:
            keys = sorted(pairs[w][1].keys(), key=lambda x:-pairs[w][1][x])
            out.append(keys[0])
            rule.append('dic')
            continue
        
        if w in wordlist:
            out.append(w)
            rule.append('wls')
            continue
        if w in liu2011:
            out.append(liu2011[w][0])
            rule.append('liu')
            continue
        if w in han2011:
            out.append(han2011[w])
            rule.append('han')
            continue
        
        else:
            out.append(w)
            rule.append('ink')
    #d['oracle'] = d['output']
    d['output'] = out
    d['rule'] = rule


# In[408]:


with open('output/'+output_fn+'.json','w') as f:
    json.dump(output_data, f)

# python data/evaluation.py --pred output/myoutput.json --oracle data/valid_data.json
# python data/evaluation.py --pred output/myoutput.json --oracle data/test_truth.json