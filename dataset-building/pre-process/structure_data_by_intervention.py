import sys
from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

import csv
import json
import re
from utils.whole_debate import load_moral_maze, load_US2016, load_rrd, load_reddit
from utils.load_data import load_dataset, load_mapping
import re

def fusion_interventions(args_per_intervention, to_id, from_id):
    if 'went_to' in args_per_intervention[to_id]['comment'].keys():
        to_id = args_per_intervention[to_id]['comment']['went_to']
    args_per_intervention[to_id]['ARGS'].extend(args_per_intervention[from_id]['ARGS']) # aquí se añaden los args hast ahora
    if 'FULL_ARGS' in args_per_intervention[from_id].keys():
        args_per_intervention[to_id]['FULL_ARGS'].extend(args_per_intervention[from_id]['FULL_ARGS'])
    args_per_intervention[to_id]['SCHEMES'].extend(args_per_intervention[from_id]['SCHEMES'])
    args_per_intervention[to_id]['IDS'].extend(args_per_intervention[from_id]['IDS'])
    args_per_intervention[to_id]['SENTENCES'].extend(args_per_intervention[from_id]['SENTENCES'])
    return to_id, args_per_intervention

def empty_intervention(intervention):
    return {'id': intervention, 'IDS': [], 'SCHEMES': [], 'ARGS': [], 'INTERVENTION': '', 'SENTENCES': [],
     'comment': {'added':[]}}

def find_args_per_intervention(dataset, debate, flat_Ls_args, L_to_I):
    args_per_intervention = {}
    count_orphan_Ls = 0
    count_not_orphan = 0
    for i, line in enumerate(debate):
        line_id = re.sub(' ', '_', line[0]) + '_' + str(i)
        args_per_intervention[line_id] = empty_intervention(line_id)
        args_per_intervention[line_id]['INTERVENTION'] += '\n'.join([sentence[0] for sentence in line[1] if sentence[0] != 'citation'])
        for sentence in line[1]:
            if sentence[1] in flat_Ls_args:  # these are Ls
                args_per_intervention[line_id]['IDS'].append(L_to_I[sentence[1]])  # turn back to Is
                args_per_intervention[line_id]['SENTENCES'].append(sentence[0])
                count_not_orphan += 1
            else:
                count_orphan_Ls += 1
    return args_per_intervention

def relate_args_to_intervention(dataset, unique_args, debate, flat_Ls_args, L_to_I):
    args_per_intervention = find_args_per_intervention(dataset, debate, flat_Ls_args, L_to_I)

    count = 0
    waiting_list = {}
    mapping_of_fusioned_interventions = {}

    for intervention in args_per_intervention.keys():
        for pair in unique_args:
            if pair not in args_per_intervention[intervention]['ARGS']:
                first, second = pair.split('_')
                # find if both propositions in the argument are part of the same intervention
                if first in args_per_intervention[intervention][('IDS')] and second in \
                        args_per_intervention[intervention][('IDS')]:
                    if intervention not in mapping_of_fusioned_interventions.keys():
                        args_per_intervention[intervention]['ARGS'].append(pair)
                    else:
                        args_per_intervention[mapping_of_fusioned_interventions[intervention]]['ARGS'].append(pair)
                # if both propositions are not part of the same intervention, save until you find the other half
                elif first in args_per_intervention[intervention][('IDS')] or second in \
                        args_per_intervention[intervention][('IDS')]:
                    if pair not in list(waiting_list.keys()):
                        waiting_list[pair] = {'id_first': intervention,
                                              'len_first': len(args_per_intervention[intervention]['INTERVENTION']),
                                              'first_intervention': args_per_intervention[intervention],
                                              'author_first': intervention.split('_')[0]}
                    else:
                        if waiting_list[pair]['author_first'] == intervention.split('_')[0]: # if it's from the same author, put both interventions together
                            # do not merge the interventions if that's going to make them too long and later they'll have to be cut
                            if len(args_per_intervention[intervention]['INTERVENTION']) + len(args_per_intervention[waiting_list[pair]['id_first']]['INTERVENTION']) < 1300:
                                #args_per_intervention[waiting_list[pair]['id_first']]['ARGS'].append(pair)
                                #args_per_intervention[waiting_list[pair]['id_first']]['comment']['added'].append(intervention)

                                if intervention not in mapping_of_fusioned_interventions.keys():
                                    # los pasos de fusionar solo se hace si no se han fusionado todavía
                                    actual_id, args_per_intervention = fusion_interventions(args_per_intervention, waiting_list[pair]['id_first'], intervention)
                                    args_per_intervention[actual_id]['INTERVENTION'] += '\n...\n' + args_per_intervention[intervention]['INTERVENTION']
                                    mapping_of_fusioned_interventions[intervention] = actual_id
                                    args_per_intervention[intervention] = empty_intervention(intervention)
                                    args_per_intervention[actual_id]['comment']['added'].append(intervention)
                                    args_per_intervention[intervention]['comment']['went_to'] = actual_id
                                    print('intervention', intervention, 'went to', actual_id)
                                    args_per_intervention[actual_id]['ARGS'].append(pair)  # esto se hace siempre, es la actividad core
                                else:
                                    args_per_intervention[mapping_of_fusioned_interventions[intervention]]['ARGS'].append(pair) # esto se hace siempre, es la actividad core
                            else:
                                args_per_intervention[intervention]['ARGS'].append(pair)
                                print('not merging because merge is too long', intervention)


                        else:
                            # when this happens is because A is answering to B. These are interesting arguments, but they are not really coherent with the rest
                            # in these cases, the argument should be associated to the intervention of the first proposition, I put an _A_ to be able to retrieve them later
                            count += 1
                            if dataset in ['rrd', 'us2016reddit']:#, 'US2016']:#, 'moral_maze_schemes']:
                                first, second = pair.split('_')
                                if first in args_per_intervention[intervention][('IDS')]:
                                    args_per_intervention[intervention]['ARGS'].append(first+'_A_'+second)
                                    args_per_intervention[intervention]['comment']['answering'] = waiting_list[pair]['id_first'].split('_')[0]
                                    args_per_intervention[intervention]['comment']['with_message'] = args_per_intervention[waiting_list[pair]['id_first']]['INTERVENTION']
                                    print('intervention', intervention, 'answers to', waiting_list[pair]['id_first'])
                                elif first in args_per_intervention[waiting_list[pair]['id_first']]['IDS']:
                                    args_per_intervention[waiting_list[pair]['id_first']]['ARGS'].append(
                                        first + '_A_' + second)
                                    args_per_intervention[waiting_list[pair]['id_first']]['comment']['answering'] = intervention.split('_')[0]
                                    args_per_intervention[waiting_list[pair]['id_first']]['comment']['with_message'] = args_per_intervention[intervention]['INTERVENTION']
                                    print('intervention', waiting_list[pair]['id_first'], 'answers to', intervention)

    print('args in different interventions', count)
    return args_per_intervention

def try_if_cutting(instance, L_to_I, n):
    sec1, sec2 = instance['IDS'][:n], instance['IDS'][n:]
    for arg in instance['ARGS']:
        if '_A_' not in arg:  # this is so it ignores the answering pairs
            first, second = arg.split('_')
            if (first in sec1 and second in sec1) or (first in sec2 and second in sec2):
                return True
            else:
                return False
        else:
            return True

def chunk_long_pragraphs(long_paragraph, L_to_I):
    good_cutting = False
    try_by = 2
    i = -1
    initial_cut = round(len(long_paragraph['IDS']) / try_by) # cut the long text by half
    while good_cutting==False and initial_cut + i < len(long_paragraph['IDS']):
        i += 1
        good_cutting = try_if_cutting(long_paragraph, L_to_I, initial_cut + i) # try if this cuts any argument, if so, try by next line

    # TODO: parece ser que en RRD se cortan algunos argumentos. Por ejemplo en el chunking de JetJock_156
    if good_cutting == False: # if there is no way to cut, don't do it
        print('not able to chunk', long_paragraph['id']) 
        return False, False

    # create the two new entries
    cutting_at = initial_cut+i
    entry1 = {'id': long_paragraph['id']+'_1', 'IDS': long_paragraph['IDS'][:cutting_at],
              'SCHEMES': [], 'ARGS': [], 'INTERVENTION': '', 'SENTENCES': long_paragraph['SENTENCES'][:cutting_at], 'comment':{'added':[]}}
    entry2 = {'id': long_paragraph['id'] + '_2', 'IDS': long_paragraph['IDS'][cutting_at:],
              'SCHEMES': [], 'ARGS': [], 'INTERVENTION': '', 'SENTENCES': long_paragraph['SENTENCES'][cutting_at:], 'comment':{'added':[]}}

    for e in [entry1, entry2]:
        for arg in long_paragraph['ARGS']:
            if '_A_' not in arg: # this is so it ignores the answering pairs
                first, second = arg.split('_')
                if first in e['IDS']:
                    e['ARGS'].append(arg)
        e['INTERVENTION'] += '\n'.join([sentence for sentence in e['SENTENCES'] if sentence != 'citation'])

    return entry1, entry2

def shorten_paragraphs(args_per_intervention, L_to_I):
    to_delete_entries = []
    to_add_entries = []
    for key, intervention in args_per_intervention.items():
        if len(intervention['INTERVENTION']) > 1300 and len(intervention['ARGS']) > 0:

            e1, e2 = chunk_long_pragraphs(intervention, L_to_I)
            if e1:
                to_delete_entries.append(intervention)
                to_add_entries.append(e1)
                to_add_entries.append(e2)
                print('paragraph', key, 'has been chunked')

    for entry in to_delete_entries:
        args_per_intervention.pop(entry['id'])
    for entry in to_add_entries:
        args_per_intervention[entry['id']] = entry

    return args_per_intervention

def find_last_intervention_same_speaker(args_per_intervention, intervention):
    candidates_distance = []
    candidates = []
    iden = intervention['id'].split('_')
    for key, intervention2 in args_per_intervention.items():
        candidates.append(key)
        iden2 = intervention2['id'].split('_')
        if iden2[0] == iden[0]:
            if int(iden[1]) - int(iden2[1]) > 0:
                candidates_distance.append(int(iden[1]) - int(iden2[1]))
            else:
                candidates_distance.append(100)
        else:
            candidates_distance.append(100)

    return candidates[candidates_distance.index(min(candidates_distance))]

def join_short_interventions(args_per_intervention):
    to_delete = []
    for key, intervention in args_per_intervention.items():
        if len(intervention['INTERVENTION']) < 100 and len(intervention['ARGS']) > 0:
            #print(intervention)
            last = find_last_intervention_same_speaker(args_per_intervention, intervention)
            #print(args_per_intervention[last])
            args_per_intervention[last]['INTERVENTION'] += '\n...\n' + intervention['INTERVENTION']
            actual_id, args_per_intervention = fusion_interventions(args_per_intervention, last, key)
            args_per_intervention[actual_id]['comment']['added'].append(key)
            args_per_intervention[key] = empty_intervention(key)
            args_per_intervention[key]['comment']['went_to'] = actual_id
            print('(2) intervention', key, 'went to', actual_id)

    for entry in to_delete:
        args_per_intervention.pop(entry)

    return args_per_intervention

def args_per_intervention_no_Ls(data, dataset):
    args_per_intervention = {}
    already_there = []
    for i, line in enumerate(data):
        line_id = None
        if line['full_text'] not in already_there:
            line_id = dataset + '_' + str(i)
            args_per_intervention[line_id] = empty_intervention(line_id)
            args_per_intervention[line_id]['INTERVENTION'] = line['full_text']
            args_per_intervention[line_id]['SENTENCES'] = line['full_text'].split('\n')
            already_there.append(line['full_text'])
        else:
            for entry in args_per_intervention.keys():
                if args_per_intervention[entry]['INTERVENTION'] == line['full_text']:
                    line_id = args_per_intervention[entry]['id']
        args_per_intervention[line_id]['ARGS'].append(line['proposition_id'] + '_' + line['related_proposition_id'])
        #args_per_intervention[line_id]['SCHEMES'].append(line['schema'])
    return args_per_intervention

def main():
    dataset = sys.argv[1]

    if dataset == 'moral_maze_schemes':
        whole_debate = load_moral_maze('data/originals/'+dataset+'/')
    elif dataset == 'US2016':
        whole_debate = load_US2016('data/originals/'+dataset+'/')
    elif dataset == 'rrd':
        whole_debate = load_rrd('data/originals/'+dataset+'/')
    elif dataset == 'us2016reddit':
        whole_debate = load_reddit('data/originals/'+dataset+'/')
    else:
        whole_debate = None

    #print(whole_debate)
    # for line in whole_debate:
    #     print(line[0])
    #     for sentence in line[1]:
    #         if sentence[0] != 'citation':
    #             print(sentence[0])

    ## Load critical questions
    schemes_dict, mapping_dict, to_my_names = load_mapping(dataset)

    interesting_schemas = ["GenericAdHominem", "PracticalReasoning", "Sign", "Consequences", "CircumstantialAdHominem",
                           "Values", "Bias", "DangerAppeal", "CauseToEffect", "Analogy", "PositionToKnow",
                           "PopularPractice", "Alternatives", "FearAppeal", "PopularOpinion", "ExpertOpinion",
                           "VerbalClassification", "Example"]

    # LOAD DATA INTO FORMAT BY ARGUMENT
    data, I_to_L = load_dataset('data/originals/'+dataset+'/', return_I_to_L=True)
    #print(data[0]['full_text'])
    #out_by_entry = {}
    arguments = []
    for i,entry in enumerate(data):
        if to_my_names[entry['schema']] in interesting_schemas:
            #out = []
            try:
                scheme = mapping_dict[entry['schema']]
                premises = []
                for premise in schemes_dict[scheme]['form']['premises']:
                    premises.append(premise)
                premises.append(schemes_dict[scheme]['form']['conclusion'])

                if dataset == 'moral_maze_schemes': # issue with moral maze
                    entry['full_text'] = entry['full_text'].replace("http://www.theonering.net/torwp/2015/05/16/98212-one-does-not-simply-fly-into-mordor/", "ARTICLE ")

                arguments.append([entry['proposition_id']+'_'+entry['related_proposition_id'], entry['schema'],
                                  (entry['proposition'], entry['related_proposition']), premises,
                                 schemes_dict[scheme]['cq']])
            except KeyError:
                print(entry['schema'], 'not in the mapping.')
                print('')

    #print(len(arguments))
    #print(arguments[0])

    # STRUCTURE IT IN PARAGRAPH FORMAT
    count = 0
    negative_count = 0
    all_Is = []
    for k, entry in I_to_L.items():
        if entry['L']:
            count += 1
            all_Is.append(k)
        else:
            negative_count += 1
    #print(count, negative_count)

    flat_Ls_args = []
    L_to_I = {}
    for Is in all_Is:
        L_to_I[I_to_L[Is]['L']] = Is
        flat_Ls_args.append(I_to_L[Is]['L'])

    args = [arg[0] for arg in arguments]
    unique_args = list(set(args))

    if dataset in ['moral_maze_schemes', 'US2016', 'rrd', 'us2016reddit']:
        args_per_intervention = relate_args_to_intervention(dataset, unique_args, whole_debate, flat_Ls_args, L_to_I)

        args_per_intervention = shorten_paragraphs(args_per_intervention, L_to_I)
        # if needed it, I can cut again running the same function as is, but I don't need it
        # if dataset != 'us2016reddit': # commented this because it messed the data
        #     args_per_intervention = join_short_interventions(args_per_intervention)
        #     # I repeate this two times to not get any short paragraph
        #     args_per_intervention = join_short_interventions(args_per_intervention)
    else:
        args_per_intervention = args_per_intervention_no_Ls(data, dataset)


    for i, intervention in enumerate(args_per_intervention.keys()):
        args_per_intervention[intervention]['FULL_ARGS'] = []
        if intervention == 'JL_71' and dataset=='moral_maze_schemes':  # this text was an error, and it was there twice
            args_per_intervention[intervention]['INTERVENTION'] = args_per_intervention[intervention]['INTERVENTION'][
                                                                  :round(len(args_per_intervention[intervention][
                                                                                 'INTERVENTION']) / 2)]
        for arg in args_per_intervention[intervention]['ARGS']:
            for ref_args in arguments:
                if '_A_' in arg:
                    first, answer, second = arg.split('_')
                    new_arg = first+'_'+second
                else:
                    new_arg=arg
                if ref_args[0] == new_arg:
                    args_per_intervention[intervention]['SCHEMES'].append(ref_args[1])
                    args_per_intervention[intervention]['FULL_ARGS'].append({'id':arg, 'scheme':ref_args[1],
                                                                             'propositions':ref_args[2],
                                                                             'premises':ref_args[3],
                                                                            'cqs': ref_args[4]})
    with open('data/by_intervention/'+dataset+'.json', 'w') as o:
        json.dump(args_per_intervention, o, indent=5)

    print('Número de intervenciones:', len(list(args_per_intervention.keys())))
    count = 0
    for key, value in args_per_intervention.items():
        if value['FULL_ARGS'] and not key.startswith('HOLT'):
            count+= len(value['FULL_ARGS'])
    print('Número de argumentos', count, '\n')


if __name__ == "__main__":
        main()