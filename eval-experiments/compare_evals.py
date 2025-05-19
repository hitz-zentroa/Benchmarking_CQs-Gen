import json
import sys
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
from collections import Counter

def main():
    results_to_compare1 = sys.argv[1]
    results_to_compare2 = sys.argv[2]
    binary = int(sys.argv[3])

    with open(results_to_compare1) as f:
        reference = json.load(f)

    with open(results_to_compare2) as f:
        comparison = json.load(f)

    print(len(reference), len(comparison))

    # TODO: crec que hauria de restringir que ignori els que no estiguin a test

    r_label = []
    c_label = []
    for key,line in reference.items():
        for c,cq in enumerate(line['cqs']):
            try:
                if not binary:
                    r_label.append(cq['label'])
                else:
                    if cq['label'] != 'Useful':
                        r_label.append('Not-Useful') 
                    else:
                        r_label.append(cq['label'])
            
                c_label.append(comparison[key]['cqs'][c]['label'])
            except KeyError:
                print('error:', cq)



    print('Labels in reference', Counter(r_label))
    print('Labels in prediction', Counter(c_label))
    print(cohen_kappa_score(r_label, c_label))
    if not binary:
        print(confusion_matrix(r_label, c_label, labels=['Invalid', 'Unhelpful', 'Useful', 'not_able_to_evaluate']))
    else:
        print(confusion_matrix(r_label, c_label, labels=['Not-Useful', 'Useful', 'not_able_to_evaluate']))
    
    print(classification_report(r_label, c_label))

    #for key,line in reference.items():
    #    for c,cq in enumerate(line['cqs']):
    #        if cq['label']=='Useful' and comparison[key]['cqs'][c]['label'] == 'Unhelpful':
    #             print(cq['cq'])
            
    



if __name__ == "__main__":
        main()