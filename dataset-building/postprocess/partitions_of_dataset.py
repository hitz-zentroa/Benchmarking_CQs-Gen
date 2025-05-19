import json
from collections import Counter
import random

random.seed(2024)

def main():
    with open('data/benchmark/by_intervention_with_schemes.json') as f:
        data = json.load(f)

    print(len(data.keys()))

    # if utiles only stand for the half of the questions, there are all types of questions, and they are not splitted texts, keep it
    keep_for_test = []
    put_in_valid = []
    potentially_in_sample = []
    for line in data.values():
        count_proportions = []
        for cq in line['cqs']:
            count_proportions.append(cq['label'])
            if cq['novalid_reason'] == 'Nuevo concepto' and cq['postedited']: # put new concept with postedition as Useful, with the edited version
                cq['cq'] = cq['postedited']
                cq['label'] = 'Useful'
        c = Counter(count_proportions)
        if c['Invalid']+c['Unhelpful'] >= c['Useful'] and len(line['intervention_id'].split('_')) == 2 and c['Invalid'] and c['Unhelpful'] and c['Useful'] and line['intervention_id'] != 'CLINTON_235':
            keep_for_test.append(line)
        elif c['Invalid']+c['Unhelpful'] >= c['Useful'] and c['Useful'] > 7:
            potentially_in_sample.append(line['intervention_id'])
            put_in_valid.append(line)
        else:
            put_in_valid.append(line)

    print(len(keep_for_test))
    dict_out = {}
    for line in keep_for_test:
        new_line = {key: val for key, val in line.items() if key not in ['cqs']}
        new_line['cqs'] = []
        for cq in line['cqs']:
            new_dict = {key: val for key, val in cq.items() if key not in ['annotator', 'postedited', 'comment', 'lsid', 'novalid_reason', 'type']}
            new_line['cqs'].append(new_dict)
        
        dict_out[line['intervention_id']] = new_line


    with open('shared_task/data_splits/test.json', 'w') as o:
        json.dump(dict_out, o, indent=5)

    # I MANUALLY REMOVED CF_32 AND TRUMP_226 FROM TEST SINCE THEY WERE VERY SHORT AND HAD ERRONEOUS ANNOTATIONS

    print(len(put_in_valid))
    dict_out = {}
    for line in put_in_valid:
        new_line = {key: val for key, val in line.items() if key not in ['cqs']}
        new_line['cqs'] = []
        for cq in line['cqs']:
            new_dict = {key: val for key, val in cq.items() if key not in ['annotator', 'postedited', 'comment', 'lsid', 'novalid_reason', 'type']}
            new_line['cqs'].append(new_dict)
        
        dict_out[line['intervention_id']] = new_line


    with open('shared_task/data_splits/validation.json', 'w') as o:
        json.dump(dict_out, o, indent=4)

    sample_list = random.sample(potentially_in_sample, 5)
    sample = {}
    for key in sample_list:
        sample[key] = dict_out[key]
        print('\n'+dict_out[key]['intervention'])
        #for cq in dict_out[key]['cqs']:
        #    print(cq['cq'])
    with open('shared_task/data_splits/sample.json', 'w') as o:
        json.dump(sample, o, indent=4)





if __name__ == "__main__":
    main()


