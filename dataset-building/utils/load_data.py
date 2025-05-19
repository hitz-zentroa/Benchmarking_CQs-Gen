import os
import json
import csv
import re
from collections import Counter


def I_to_L(data, Ls, directory):
    out = {}

    # COLLECT ALL THE Is
    for node in data['nodes']:
        if node['type'] == 'I':
            ids = node['nodeID']
            out[ids] = {'edges_YA': [], 'edges_L': [], 'L': '', 'L_content':''}

    # IF Ls HAVE BEEN ANNOTATED, ASSIGN THE CORRESPONDING L TO EACH I

    if Ls: #and len(data['locutions']) != 0: # if there are Ls assign them, otherwise assign I
        for edge in data['edges']:
            if edge['toID'] in out.keys(): # find all the toID edges that are Is
                out[edge['toID']]['edges_YA'].append(edge['fromID']) # save the fromID as candidates

        for instance in out.keys(): # for all candidates
            for edge in data['edges']:
                if edge['toID'] in out[instance]['edges_YA']: # find if the candidates are in any edge as a toID
                    out[instance]['edges_L'].append(edge['fromID'])

            for node in data['nodes']:
                if node['nodeID'] in out[instance]['edges_L'] and node['nodeID'] in Ls:
                    if not out[instance]['L']:
                        out[instance]['L'] = node['nodeID']
                        out[instance]['L_content'] = node['text']
                    else:
                        #print(node['nodeID'], out[instance]['L'])
                        pass # TODO: this is an issue that happens 2 times
    else:
        for instance in out.keys(): # if there have not been Ls distinction, then the I should be the L
            out[instance] = {'edges_YA': [], 'edges_L': [], 'L': instance, 'L_content': ''}

    return out

def look_for_it_in_data(ids, data):
    for node in data['nodes']:
        if node['nodeID'] == ids:
            print(node)

def get_relevant_paragraph(proposition_id, related_proposition_id, text, I_to_L_mapping, data, context):
    relevant_paragraph = ''

    from_line = None
    L_id = I_to_L_mapping[proposition_id]['L']
    context_span = context
    for i, sent in enumerate(text):
        if sent[1] == L_id:
            from_line = i - context_span
            to_line = i + context_span

    try:
        if from_line != None and related_proposition_id != '0': # these 0s are in the moral_maze dataset and they make no sense
            r_from_line = None
            related_L_id = I_to_L_mapping[related_proposition_id]['L']
            for i, sent in enumerate(text):
                if sent[1] == related_L_id:
                    if i >= from_line and i <= to_line:
                        pass
                    else:
                        r_from_line = i - context_span
                        r_to_line = i + context_span


            for sent in text[from_line:to_line]:
                relevant_paragraph += '\n'+ sent[0]

            if r_from_line:  # if the related sentence was not included in the paragraph, add the lines that surround them
                if r_from_line <= to_line:
                    r_from_line = to_line
                else:
                    relevant_paragraph += '\n'+ '...'
                for sent in text[r_from_line:r_to_line]:
                    relevant_paragraph += '\n' +sent[0]
        else:
            pass
    except KeyError:
        #print("The following related proposition node is not an I:") # some are errors and some are citations, I corrected the errors in US2016 so just the citations are left
        #look_for_it_in_data(related_proposition_id, data)
        pass

    return relevant_paragraph


def get_from_folders(base_directory):
    directories = os.listdir(base_directory)
    out_files = []

    for directory in directories:
        files = os.listdir(base_directory + directory)
        for file in files:
            if file[-4:] == 'json':
                out_files.append(directory+'/'+file)

    return out_files


def load_dataset(directory, context=100, return_I_to_L=False):

    if directory == 'data/originals/LEGAL/':
        files = get_from_folders(directory)
    else:
        files = os.listdir(directory)
        files = sorted(files)

    if directory in ['data/originals/US2016/', 'data/originals/rrd/', 'data/origignals/us2016reddit/', 'data/originals/moral_maze_schemes/']:
        paragraph_approach = True
    else:
        paragraph_approach = False

    I_to_L_overall = {}
    list_of_propositional_relations = []
    full_text_in_file = {}
    for file in files:
        #print(file)
        with open(directory + file) as f:
            data = json.load(f)

        if directory == 'data/originals/moral_maze_schemes/':
            Ls = [node['nodeID'] for node in data['nodes'] if node['type'] == 'L' and node['nodeID'] != '8837']
        else:
            Ls = [node['nodeID'] for node in data['nodes'] if node['type'] == 'L']

        I_to_L_mapping = I_to_L(data, Ls, directory)
        I_to_L_overall.update(I_to_L_mapping)

        # get the whole text directly from L and save it for releasing it later, you can get the whole text or the relavant paragraph
        all_text = []
        if Ls:
            for node in data['nodes']:
                if node['type'] == 'L':
                    all_text.append([node['text'], node['nodeID']])
        else:
            for node in data['nodes']:
                if node['type'] == 'I':
                    all_text.append([node['text'], node['nodeID']])

        if paragraph_approach:
            full_text_in_file[file] = []
        else:
            full_text_in_file[file] = ''
        for sent in all_text:
            if not paragraph_approach:
                full_text_in_file[file] += '\n'+sent[0]
            else:
                if directory == 'data/US2016/': # TODO: a better way to do this could be found
                    if re.match('[A-Z]* :', sent[0]) and not re.match('[A-Z]*[ ]?: [A-Z]* :', sent[0]):
                        full_text_in_file[file].append(sent)
                elif directory == 'data/rrd/':
                    if not re.match('Annot: ', sent[0]):
                        full_text_in_file[file].append(sent)
                elif directory == 'data/us2016reddit/':
                    if not re.match('[A-Za-z0-9\_]*[ ]?: [A-Za-z0-9\_]* :', sent[0]):
                        full_text_in_file[file].append(sent)
                elif directory == 'data/moral_maze_schemes/':
                    if not re.match('^[A-Z]*[ ]?: [A-Z]*: ', sent[0]) and not sent[0] == 'analyses':
                        full_text_in_file[file].append(sent)
                else:
                    full_text_in_file[file].append(sent)

        for node in data['nodes']:
            if node['type'] == 'I':
                proposition_id = node['nodeID']
                text = node['text']

                # Buscar con qué propositional relation tiene un edge
                candidate_relation = ''
                schema = ''
                relation = ''
                for edge in data['edges']:
                    if edge['fromID'] == proposition_id:
                        candidate_relation = edge['toID']
                        for candidate_node in data['nodes']:
                            if candidate_node['nodeID'] == candidate_relation and candidate_node['type'] in ['RA', 'CA', 'MA']:
                                relation = candidate_node['type']
                                if Ls:
                                    try:
                                        schema = candidate_node['scheme']
                                    except KeyError:
                                        schema = ''
                                    break
                                else: # this is for essays
                                    schema = candidate_node['text']
                        break

                # Buscar a que proposition conecta ese edge
                related_proposition_id = ''
                related_proposition = ''
                if candidate_relation != '':
                    for edge in data['edges']:
                        if edge['fromID'] == candidate_relation:
                            related_proposition_id = edge['toID']
                            for candidate_node in data['nodes']:
                                if candidate_node['nodeID'] == related_proposition_id:
                                    related_proposition = candidate_node['text']
                                    break
                            break


                    # find the YA of each I
                    YAs = {'0':'', '1':''}
                    for i, prop in enumerate([proposition_id, related_proposition_id]):
                        for edge in data['edges']:
                            if edge['toID'] == prop:
                                candidate_intent = edge['fromID']
                                for candidate_node in data['nodes']:
                                    if candidate_node['nodeID'] == candidate_intent and candidate_node['type'] == 'YA':
                                        YAs[str(i)] = candidate_node['text']
                                        break
                                break

                    # get the surrounding sencentes
                    if paragraph_approach:
                        relevant_paragraph = get_relevant_paragraph(proposition_id, related_proposition_id, full_text_in_file[file],
                                                                    I_to_L_mapping, data, context)
                    else:
                        relevant_paragraph = full_text_in_file[file]

                    interesting_info = {'proposition_id': proposition_id, 'proposition': text, 'type_relation': relation,
                                'schema': schema, 'related_proposition_id': related_proposition_id,
                                'related_proposition': related_proposition, 'file_name': file,
                                'intent1': YAs['0'], 'intent2': YAs['1'], 'full_text': relevant_paragraph} # 'speaker2':speaker2,'speaker1': speaker1,
                    list_of_propositional_relations.append(interesting_info)
    if return_I_to_L:
        return list_of_propositional_relations, I_to_L_overall
    else:
        return list_of_propositional_relations

def load_my_names():
    # Load mapping from each dataset
    mapping = []
    with open('dataset_creation/mapping_scripts/schemes_mapping.csv') as m:
        r = csv.reader(m)
        # next(r)
        for line in r:
            mapping.append(line)

    mapping_dict = {}
    scheme_column = [i for i, name in enumerate(mapping[0]) if name == 'schemes_json'][0]
    added_schemes = [i for i, name in enumerate(mapping[0]) if name == 'schemes_in_Walton2001?'][0]
    my_names_column = [i for i, name in enumerate(mapping[0]) if name == 'my_labels'][0]

    for line in mapping[1:]:  # first raw in the column names
        if line[scheme_column]:
            mapping_dict[line[scheme_column]] = line[my_names_column]
        elif line[added_schemes]:
            mapping_dict[line[added_schemes]] = line[my_names_column]
    return mapping_dict

def load_mapping(dataset):

    schemes = []
    with open('scripts/utils/walton_plus.jsonl') as w:
        for line in w:
            schemes.append(json.loads(line))
    schemes_dict = {}
    for line in schemes:
        schemes_dict[line['name']] = line

    # Load mapping from each dataset
    mapping = []
    with open('scripts/utils/schemes_mapping.csv') as m:
        r = csv.reader(m)
        #next(r)
        for line in r:
            mapping.append(line)
    col_id = [i for i,name in enumerate(mapping[0]) if name == dataset][0]

    mapping_dict = {}
    to_my_names = {}
    scheme_column = [i for i, name in enumerate(mapping[0]) if name == 'schemes_json'][0]
    added_schemes = [i for i, name in enumerate(mapping[0]) if name == 'schemes_in_Walton2001?'][0]
    my_names_column = [i for i, name in enumerate(mapping[0]) if name == 'my_labels'][0]

    for line in mapping[1:]: # first raw in the column names
        to_my_names[line[col_id]] = line[my_names_column]
        if line[scheme_column]:
            mapping_dict[line[col_id]] = line[scheme_column]
        elif line[added_schemes]:
            mapping_dict[line[col_id]] = line[added_schemes]

    return schemes_dict, mapping_dict, to_my_names


def correct_questions_grammar(model, tokenizer, text):

    grammarly_edited = ''
    lines = text.split('\n')
    for line in lines[1:]:
        input_text = 'Rewrite to make this easier to understand: ' + line
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        input_ids = input_ids.to('cuda')
        outputs = model.generate(input_ids, max_length=256)
        edited_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        grammarly_edited += '\n' + edited_text

    return grammarly_edited

def id_selection_p1():
    return ["TRUMP_15", "TRUMP_99", "CLINTON_107", "TRUMP_112", "CLINTON_136", "CLINTON_1_1", "TRUMP_129_2", "TRUMP_174_1", "TRUMP_240_1", "CLINTON_244_1", "CL_8", "MT_14", "ND_19", "ND_23", "MP_24", "JW_35", "CF_58", "CF_62", "JL_69", "MT_45"]