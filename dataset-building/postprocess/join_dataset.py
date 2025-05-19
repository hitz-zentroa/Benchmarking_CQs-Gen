import json
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

def extract_labels(annotation, _id):
    if not annotation:
        print('Missing annotation', _id)
        return '', '', '', ''
    labels = []
    texts = []
    for a in annotation:
        try: 
            labels.append((a['value']['choices'][0], a['from_name']))
        except KeyError:
            pass
        try:
            #print((a['value']['text'][0], a['from_name']))
            texts.append((a['value']['text'][0], a['from_name']))
        except KeyError:
            pass
    
    main_lab = ''
    no_valid_label = ''
    postedicion = ''
    needs_postedition = False
    comment = ''
    #print(len(labels), len(texts))
    if labels:
        for lab in labels:
            if lab[1] == 'usefulness':
            #if lab in ['Útil', 'Poco Útil', 'No Válida']:
                main_lab = lab[0]
            elif lab[1] == 'reason': #in ['Nuevo concepto', 'Muy General', 'No crítica', 'Sin relación', 'Mal razonamiento']:
                no_valid_label = lab[0]
            elif lab[1] == 'postedition': 
                needs_postedition = True
            
        for text in texts:
            if text[1] == 'comentario':
                comment = text[0]
            elif text[1] == 'new_cq':
                postedicion = text[0]

    if not main_lab and no_valid_label:
        print('Correcting error in', _id)
        main_lab = 'No Válida'

    if no_valid_label and main_lab != 'No Válida':
        print('Correcting error in', _id)
        main_lab = 'No Válida'

    if needs_postedition and not postedicion:
        print('Missing postedition', _id)

    if needs_postedition and not main_lab:
        print('Correcting error in', _id)
        main_lab = 'No Válida'

    if not main_lab:
        print('Missing label. Reason:', comment, _id) #TODO: the comment disapears in this step
        #print(annotation)

    return main_lab, no_valid_label, postedicion, comment


def main():

    # load the 6 parts and join
    all = []
    datasets = ['bloque_1_celia', 'bloque_1_pablo', 'bloque_2_celia', 'bloque_2_pablo', 'bloque_3_celia', 'bloque_3_pablo']
    for d in datasets:
        annotator = d.split('_')[-1]
        with open('scripts/evaluations/usefulness/data/'+d+'.json') as f:
            data = json.load(f)
            for line in data:
                line['annotator'] = annotator
            all.extend(data)

    print(len(all))
    intervention_ids = [line['data']['intervention_id'] for line in all]


    # load the annotation test and decide how to make it final, also join
    
    for proba in ['1', '3']: # trial 2 has same data as trial 3, entities in trial 1 should be reviewed, because iaa was low
        for ann in ['1']: # for now, just keep one of the annotators: Celia
            count = 0
            with open('scripts/evaluations/usefulness/data/proba_'+proba+'_ann_'+ann+'.json') as f:
                data = json.load(f)

            for line in data:
                line['annotator'] = 'pablo'

            #for line in data:
            #    if line['data']['intervention_id'] not in intervention_ids:
            #        count+=1

            all.extend(data)
            intervention_ids = [line['data']['intervention_id'] for line in all]

            #print(proba, ann, len(data), count)


    # keep only important data and extract basic stats
    #print(all[0])
    #print(all[0]['annotations'][-1]['result'][-1]['value']['choices'])
    all_simplified = []
    for line in all:
        #print(line)
        label, no_valid_label, postedited, comment = extract_labels(line['annotations'][0]['result'], line['id'])
        if '_T_' in line['data']['id']:
            origin = 'theory'
        else:
            origin = 'LLMs'

        if label:
            all_simplified.append({ 
                'lsid':line['id'], 'id':line['data']['id'], 'intervention_id':line['data']['intervention_id'], 
                'intervention':line['data']['intervention'], 'cq':line['data']['cq'],
                'label': label,
                'novalid_reason': no_valid_label,
                'postedited': postedited,
                'type': origin,
                #'dataset_origin': select_dataset_origin(line['data']['intervention']), # I'm doing it later
                'annotator': line['annotator'],
                'comment': comment
                })

    labels = [line['label'] for line in all_simplified]
    print(Counter(labels))
    reason = [line['novalid_reason'] for line in all_simplified if line['novalid_reason']]
    print(Counter(reason))
    print('Proportion of valid', (Counter(labels)['Útil']+Counter(labels)['Poco Útil'])/len(labels))
    print('Proportion of usefull', (Counter(labels)['Útil']/len(labels)))
    postedits = [line['postedited'] for line in all_simplified if line['postedited']]
    print('Postedits', len(postedits))
    reason_postedits = [line['novalid_reason'] for line in all_simplified if line['postedited']]
    print('Reason of postedits', Counter(reason_postedits))

    # theory_labels = []
    # llm_labels = []
    # reason_tehory_labels = []
    # reason_llm_labels = []
    # for line in all_simplified:
    #     if line['type'] == 'theory':
    #         theory_labels.append(line['label'])
    #         reason_tehory_labels.append(line['novalid_reason'])
    #     else:
    #         llm_labels.append(line['label'])
    #         reason_llm_labels.append(line['novalid_reason'])

    # print('\nComparison Theory vs LLM CQs')
    # print('Proportion of valid', (Counter(llm_labels)['Útil']+Counter(llm_labels)['Poco Útil'])/len(llm_labels))
    # print('Proportion of usefull', (Counter(llm_labels)['Útil']/len(llm_labels)))
    # print(Counter(reason_llm_labels))
    # print('Proportion of valid', (Counter(theory_labels)['Útil']+Counter(theory_labels)['Poco Útil'])/len(theory_labels))
    # print('Proportion of usefull', (Counter(theory_labels)['Útil']/len(theory_labels)))
    # print(Counter(reason_tehory_labels))

    by_intervention = {}
    for line in all_simplified:
        #if line['type'] == 'theory'
        if line['intervention_id'] not in list(by_intervention.keys()):
            by_intervention[line['intervention_id']] = {'Útil':0, 'Poco Útil':0, 'No Válida':0}
        try:
            by_intervention[line['intervention_id']][line['label']] += 1
        except KeyError:
            continue

    print('\nAverage per intervention')
    averages = {'Útil':0, 'Poco Útil':0, 'No Válida':0}
    for line in by_intervention.values():
        for key in averages.keys():
            averages[key] += line[key]
    print('N interventions', len(by_intervention.keys()))
    for key in averages.keys():
        print(key, averages[key]/len(by_intervention.keys()))

    # output the whole dataset and place it in data/benchmnark
    with open('data/benchmark/whole_dataset.json', 'w') as o:
        json.dump(all_simplified, o, indent=5)


    datasets = ['US2016', 'us2016reddit', 'rrd', 'moral_maze_schemes']
    change_labels = {'Útil': 'Useful','Poco Útil':'Unhelpful','No Válida':'Invalid'}
    by_intervention_dataset = {}
    for line in all_simplified:
        dataset = ''
        for d in datasets:
            if d in line['id']:
                dataset = d
        if line['intervention_id'] not in list(by_intervention_dataset.keys()): 
            by_intervention_dataset[line['intervention_id']] = {'intervention_id':line['intervention_id'], 'intervention':line['intervention'], 'dataset': dataset, 'cqs':[]}
        try:
            if dataset:
                by_intervention_dataset[line['intervention_id']]['dataset'] = dataset # I need to do this because I did not keep the origin of the questions in the theoruy ones
            by_intervention_dataset[line['intervention_id']]['cqs'].append({
                'lsid':line['lsid'], 
                'id':line['id'],
                'cq':line['cq'],
                'label': change_labels[line['label']],
                'novalid_reason': line['novalid_reason'],
                'postedited': line['postedited'],
                'type': line['type'],
                'annotator': line['annotator'],
                'comment': line['comment']
            })
        except KeyError:
            continue

    # output the whole dataset and place it in data/benchmnark
    with open('data/benchmark/by_intervention.json', 'w') as o:
        json.dump(by_intervention_dataset, o, indent=5)


    # output the distribution of the three labels per dataset of origins, use boxplots
    counts_per_dataset = {'US2016':{'Useful':[], 'Unhelpful':[], 'Invalid':[]}, 
                          'us2016reddit':{'Useful':[], 'Unhelpful':[], 'Invalid':[]}, 
                          'rrd':{'Useful':[], 'Unhelpful':[], 'Invalid':[]}, 
                          'moral_maze_schemes':{'Useful':[], 'Unhelpful':[], 'Invalid':[]}}
    #counts = {'Useful':[], 'Unhelpful':[], 'Invalid':[]}
    for intervention in by_intervention_dataset.values():
        util_labels = 0
        pocoutil_labels = 0
        novalid_labels = 0
        for cq in intervention['cqs']:
            if cq['label'] == 'Useful':
                util_labels += 1
            elif cq['label'] == 'Unhelpful':
                pocoutil_labels += 1
            elif cq['label'] == 'Invalid':
                novalid_labels += 1
        counts_per_dataset[intervention['dataset']]['Useful'].append(util_labels)
        counts_per_dataset[intervention['dataset']]['Unhelpful'].append(pocoutil_labels)
        counts_per_dataset[intervention['dataset']]['Invalid'].append(novalid_labels)



    for dataset in datasets:
        print('\n', dataset)
        total_count = [sum(counts_per_dataset[dataset][label]) for label in counts_per_dataset[dataset].keys()]
        total_dataset = sum(total_count)
        fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
        print('interventions', len(counts_per_dataset[dataset]['Useful']))
        print('cqs', total_dataset)
        for i,label in enumerate(counts_per_dataset[dataset].keys()):
            axs[i].hist(counts_per_dataset[dataset][label], bins=5)
            print(label, #sum(counts_per_dataset[dataset][label])/len(counts_per_dataset[dataset][label]),
                  sum(counts_per_dataset[dataset][label])/total_dataset*100)
        fig.savefig('scripts/postprocess/distribution_'+dataset+'.png') 

        

        #plt.savefig('scripts/postprocess/distribution.png')
        



if __name__ == "__main__":
    main()