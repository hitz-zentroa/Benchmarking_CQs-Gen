import json
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import Counter
import sys
import argparse
import logging
import csv
from datetime import datetime
from evaluate import load # if you don't want to use bleurt or comet and you don't have it installed, you can comment this, gpu might be needed

logger = logging.getLogger(__name__)

def main():
    # arguments
    parser = argparse.ArgumentParser(prog='Evaluate')
    parser.add_argument('--metric', default='similarity', type=str, choices=['similarity', 'bleurt', 'chrf', 'comet'])
    parser.add_argument('--input_path', type=str, default='test.json', help='Path of the test set.')
    parser.add_argument('--threshold', type=float, default=0.6, help='Threshold to determine when the sentences are not similar. For bleurt, the threshold should probably be a negative number (e.g. -0.3).') 
    parser.add_argument('--output_path', type=str, default='results/', help='Path of the output.')
    parser.add_argument('--submission_path', type=str, default='output.json', help='Path where the generated questions have been saved.')
    parser.add_argument('--submission_name', type=str, default='', help='Name your submission if it does not have a name.')
    args = parser.parse_args()

    #logger
    logging.basicConfig(filename='logs/eval.log', level=logging.INFO)
    logger.info('THRESHOLD: '+str(args.threshold)+'\nMETRIC: '+args.metric)
    print(args.submission_path, flush=True)

    # load the similarity model
    if args.metric == 'similarity':
        model = SentenceTransformer("stsb-mpnet-base-v2") 
    elif args.metric == 'bleurt':
        model = load("bleurt", module_type="metric")
    elif args.metric == 'chrf':
        model = load("chrf")
    elif args.metric == 'comet':
        comet_metric = load('comet')

    # load the whole dataset
    with open(args.input_path) as f:
        reference=json.load(f)

    with open(args.submission_path) as f:
        new = json.load(f)

    # start the evaluation
    predicted_labels = []
    punctuations = []

    for instance in new.keys(): # for each intervention
        if instance in reference.keys():
            punctuation = 0
            reference_set = [ref['cq'] for ref in reference[instance]['cqs']]
            if new[instance]['cqs'] != 'Missing CQs':
                for i, line in enumerate(new[instance]['cqs']): # look into each question of the new cqs and find the most similar question in the references
                    winner = None
                    if args.metric == 'similarity':
                        sentence_embedding = model.encode(line['cq'])
                        reference_embedding = model.encode(reference_set)
                        sims = model.similarity(sentence_embedding, reference_embedding).tolist()[0]
                        
                    if args.metric == 'bleurt':
                        results = model.compute(predictions=[line['cq']] * len(reference_set), references=reference_set)
                        sims = results['scores']

                    if args.metric == 'chrf':
                        sims = []
                        for ref in reference_set:
                            results = model.compute(predictions=[line['cq']], references=[ref])
                            sims.append(results['score'])
                        #print(sims)

                    if args.metric == 'comet':
                        results = comet_metric.compute(predictions=[line['cq']]*len(reference_set), references=reference_set, sources=['']*len(reference_set))   #TODO: we do not have a source
                        sims = [v for v in results["scores"]]


                    winner = np.argmax(sims)
                    # make sure the similarity of the winning reference sentence is at least 0.6
                    if sims[winner] > args.threshold:
                        label = reference[instance]['cqs'][winner]['label']
                        if label == 'Useful':
                            punctuation += 1/3
                    else: 
                        label = 'not_able_to_evaluate'
                    predicted_labels.append(label)
                    new[instance]['cqs'][i]['label'] = label
            else:
                predicted_labels.extend(['missing_cqs', 'missing_cqs', 'missing_cqs']) # this should disapear with a proper prompt that makes sure there are always 3 questions

            punctuations.append(punctuation)

    # metrics
    print('Distribution of the labels:', Counter(predicted_labels))
    print('Distribution of the intervention punctuation:', Counter(punctuations))
    print('Overall punctuation', sum(punctuations)/len(punctuations))

    # save the output
    with open(args.output_path+args.metric+'_'+str(args.threshold).replace('.', '')+'_'+args.submission_name+'.json', 'w') as o:
        json.dump(new, o, indent=4)

    with open('model_train/evaluation/llm-as-judge/evals/evaluations.csv', 'a') as o:
        w = csv.writer(o)
        w.writerow([datetime.now(), args.submission_name, args.metric, sum(punctuations)/len(punctuations), Counter(predicted_labels)['Useful']/len(predicted_labels), dict(Counter(predicted_labels)), args.threshold])


if __name__ == "__main__":
    main()
