import json
import csv

def merge_two_dicts(x, y):
    z = x.copy()   # start with keys and values of x
    z.update(y)    # modifies z with keys and values of y
    return z

def main():
    print()
    # TODO: load the outcome dataset by intervention
    with open('data/benchmark/by_intervention.json') as f:
        data = json.load(f)

    # print the theoretical no-valid cqs, see if I see a pattern
    # out_cqs = []
    # for _id, intervention in data.items():
    #     for cq in intervention['cqs']:
    #         if cq['label'] == 'No Válida' and cq['type'] == 'theory' and not cq['postedited']:
    #             out_cqs.append([cq['id'], intervention['intervention'], cq['cq'],cq['annotator'],cq['novalid_reason'], cq['comment']])
    #         #elif cq['label'] == 'No Válida' and cq['type'] == 'theory': 
    #         #    print('POSTEDITED:', cq['cq'])


    # out_cqs.sort()

    # with open('data/benchmark/inspect_theory_novalid.csv', 'w') as o:
    #     w = csv.writer(o)
    #     w.writerows(out_cqs)

    # TODO: load the original by intervention dataset
    datasets = ['moral_maze_schemes', 'US2016', 'us2016reddit', 'rrd']#, 'EO_PC']
    original_data = {}
    for dataset in datasets:

        with open("data/theory_cqs_by_intervention/"+dataset+'.json') as f:
            original_data = merge_two_dicts(original_data, json.load(f))

    # assign the argumentation schemes to each intervention
    #print(list(original_data.keys()))

    for key in data:
        if key in original_data.keys():
            #print(original_data[key]['SCHEMES'])
            data[key]['schemes'] = original_data[key]['SCHEMES']
    
    with open('data/benchmark/by_intervention_with_schemes.json', 'w') as o:
        json.dump(data, o, indent=4)
    


if __name__ == "__main__":
    main()