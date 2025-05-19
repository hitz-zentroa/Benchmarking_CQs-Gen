import csv
import sys
import re
import json
import glob
import os

def main():
    dataset = sys.argv[1]
    model = sys.argv[2]

    with open('data/llm_generation/'+model+"/"+dataset+'.csv') as f:
        r = csv.reader(f)
        data = []
        next(r)
        for line in r:
            data.append(line)

    # deduplicated_data=[]
    # already_seen = []
    # for line in data:
    #     # print(line)
    #     if line[0]+line[1]+line[2] in already_seen:
    #         #print(line[0])
    #         deduplicated_data.append(line)
    #     else:
    #         already_seen.append(line[0]+line[1]+line[2])
    #print(len(data), len(deduplicated_data))

    to_save = []
    sum_valid = 0
    for i, line in enumerate(data):
        valid = []
        not_valid = []
        cqs = line[4]
        cqs_list = cqs.split('\n')
        for cq in cqs_list:
            if re.match('.*\?(\")?( )?(\([a-zA-Z0-9\.\'\-,\? ]*\))?([a-zA-Z \.,\"\']*)?(\")?$', cq):
                valid.append(cq)
                sum_valid += 1
            else:
                not_valid.append(cq)

        still_not_valid = []
        for text in not_valid:
            new_cqs = re.split("\?\"", text+'end')
            if len(new_cqs) > 1:
                for cq in new_cqs[:-1]:
                    valid.append(cq+'?\"')
            else:
                still_not_valid.append(text)

        for i, cq in enumerate(valid):
            occurrence = re.search(r'[A-Z]', cq)
            if occurrence:
                cq = cq[occurrence.start():]
            else:
                continue
            identification = str(line[0])+'_LLM_'+dataset+'_D_'+line[1].replace('/', '_')+'_'+str(i)+'_'+line[2][0]
            out = {'data':{'id':identification, 'prompt':line[2], 'intervention': line[3], 'cq':cq, 'intervention_id': str(line[0])}}
            to_save.append(out)

    print(len(to_save), sum_valid)

    with open("data/llm_generation/"+model+"/splitted_cqs_" + dataset +'.json', 'w') as o:
        json.dump(to_save, o, indent=5)



if __name__ == "__main__":
        main()