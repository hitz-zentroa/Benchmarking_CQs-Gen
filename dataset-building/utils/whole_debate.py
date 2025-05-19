import json
import os
import re

def delete_if_consecutive_repeated(data):
    new_data = []
    previous_line= None
    for line in data:
        if previous_line:
            if not line[0] == previous_line:
                new_data.append(line)
        previous_line = line[0]
    return new_data

def load_US2016(directory):

    files = os.listdir(directory)
    files = sorted(files)
    whole_debate = []
    intervention_track = []
    previous_speaker = None

    for file in files:
        with open(directory + file) as f:
            data = json.load(f)

        # GET THE WHOLE TEXT WITH L ELEMENTS
        for node in data['nodes']:
            if node['type'] == 'L':
                text = node['text']
                speaker = re.findall('[A-Z]+[A-Za-z]* ?: ', text)
                if len(speaker) == 1 and speaker[0][:-3] in ['TRUMP', 'CLINTON', 'HOLT']:
                    text = re.sub('[A-Z]* : ', '', text)
                    if previous_speaker:
                        if previous_speaker == speaker[0]:
                            intervention_track.append([text, node['nodeID']])
                        else:
                            whole_debate.append([previous_speaker[:-3], intervention_track])
                            intervention_track = [[text, node['nodeID']]]
                        previous_speaker = speaker[0]
                    else: # para el primero
                        intervention_track.append([text, node['nodeID']])
                        previous_speaker = speaker[0]
                else:
                    intervention_track.append(['citation', node['nodeID']])
    whole_debate.append([previous_speaker[:-3], intervention_track])

    return whole_debate


def load_moral_maze(directory):

    files = os.listdir(directory)
    files = sorted(files)
    whole_debate = []
    intervention_track = []
    previous_speaker = None

    already_there = []
    for file in files:
        with open(directory + file) as f:
            data = json.load(f)

        # GET THE WHOLE TEXT WITH L ELEMENTS
        for node in data['nodes']:
            if node['nodeID'] not in already_there and node['type'] == 'L' and node['nodeID'] != '8837':
                already_there.append(node['nodeID']) # needed because the same text is there in different files
                text = node['text']
                speaker = re.findall('[A-Z]+[A-Za-z]*[ ]{1,2}: ', text)
                #print(node['nodeID'],speaker, text)
                if len(speaker) == 1:
                    text = re.sub('[A-Za-z]*[ ]{1,2}: ', '', text)
                    if previous_speaker:
                        if previous_speaker == speaker[0]:
                            intervention_track.append([text, node['nodeID']])
                        else:
                            whole_debate.append([previous_speaker[:-3], intervention_track])
                            intervention_track = [[text, node['nodeID']]]
                        previous_speaker = speaker[0]
                    else: # para el primero
                        intervention_track.append([text, node['nodeID']])
                        previous_speaker = speaker[0]
                else:
                    intervention_track.append(['citation', node['nodeID']])
                    # if text != 'analyses':
                    #     print(node)
                    #     exit()
            #if node['nodeID'] == '8837':
                #print(intervention_track)
                #print(node)
                #exit()
            # if node['nodeID'] == '30988':
            #     #print(intervention_track)
            #     print(node)
            #     exit()
    return whole_debate


def load_rrd(directory):

    files = os.listdir(directory)
    files = sorted(files)
    whole_debate = []
    intervention_track = []
    previous_interacting = None

    already_there = []
    for file in files:
        with open(directory + file) as f:
            data = json.load(f)

        # GET THE WHOLE TEXT WITH L ELEMENTS
        for node in data['nodes']:
            #print(node)
            if node['nodeID'] not in already_there and node['type'] == 'L' and node['nodeID']:
                already_there.append(node['nodeID']) # needed because the same text is there in different files
                text = node['text']

                if not text.startswith('Annot:') and not text.startswith('Barbara:'): # barbara is not a person
                    #print(text)
                    participant = re.findall('[ A-Za-z0-9\.\(\)\_]*:', text)

                    if participant:
                        speaker = re.sub('(\([0-9]*\))?:', '', participant[0])
                        speaker = re.sub('\_', '-', speaker).strip()
                        speaker = re.sub(' ', '-', speaker)

                        text = re.sub('^[ A-Za-z0-9\.\(\)\_]*: ([ A-Za-z0-9\.\(\)\_]*: )?', '', text)
                        if previous_interacting:
                            if previous_interacting == speaker:
                                intervention_track.append([text, node['nodeID']])
                            else:
                                whole_debate.append([previous_interacting, intervention_track])
                                intervention_track = [[text, node['nodeID']]]
                            previous_interacting = speaker
                        else: # para el primero
                            intervention_track.append([text, node['nodeID']])
                            previous_interacting = speaker
                    else:
                        intervention_track.append(['citation', node['nodeID']])
    #print(whole_debate[0])
    #exit()

    return whole_debate


def load_reddit(directory):

    files = os.listdir(directory)
    files = sorted(files)
    whole_debate = []
    intervention_track = []
    previous_interacting = None

    already_there = []
    for file in files:
        with open(directory + file) as f:
            data = json.load(f)

        # GET THE WHOLE TEXT WITH L ELEMENTS
        for node in data['nodes']:
            #print(node)
            if node['nodeID'] not in already_there and node['type'] == 'L' and node['nodeID']:
                #speaker=None
                already_there.append(node['nodeID']) # needed because the same text is there in different files
                text = node['text']
                if not text.startswith('Annot:') and not text.startswith('Barbara:'):  # barbara is not a person

                    participants = re.findall('[ A-Za-z0-9\-\_\.\(\)\[\]]*:', text)

                    if participants: #and interacting[0] != 'Annot':
                        speaker = re.sub('(\([0-9]*\))?:', '', participants[0])
                        text = re.sub('^[ A-Za-z0-9\-\_\.\(\)\[\]]*: ([ A-Za-z0-9\-\_\.\(\)\[\]]*: )?', '', text)
                        if previous_interacting:
                            if previous_interacting == speaker:
                                intervention_track.append([text, node['nodeID']])
                            else:
                                whole_debate.append([previous_interacting, intervention_track])
                                intervention_track = [[text, node['nodeID']]]
                            previous_interacting = speaker
                        else:  # para el primero
                            intervention_track.append([text, node['nodeID']])
                            previous_interacting = speaker
                else:
                    intervention_track.append(['citation', node['nodeID']])


    return whole_debate
