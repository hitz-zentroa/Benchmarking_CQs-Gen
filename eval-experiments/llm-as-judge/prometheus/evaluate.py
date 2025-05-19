import json
import numpy as np
from collections import Counter
import yaml
import argparse
import logging
import csv
from datetime import datetime
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM


logger = logging.getLogger(__name__)


def run_model(model, tokenizer, prompt):
    messages = [
                            {"role": "user", "content": prompt},
                        ]
    
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt", padding=True)
    #print(encodeds, flush=True)
    #return ''
    model_inputs = encodeds.to('cuda')

    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True) 
    out = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] # I'm not using it as batch, it would be more efficiant to do so
    feedback = ''
    try:
        feedback = out.split('###Feedback: ')[-1]
        output = feedback.split('[RESULT]')[-1]
    except IndexError:
        print('EVAL ERROR: '+output, flush=True)

    output = output.strip()

    if output not in ['1', '2', '3']:
        print('FROM'+output+'TO', flush=True)
        #import pdb; pdb.set_trace()

    return output, out # TODO: change this because the analysis of the feedback makes no sense, so I want to see they correspond

def get_prompts(intervention, cq, references):
    return {
        'label': f"""You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance.
        
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, some reference answers for score 1, 2 and 3, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 3. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 3)\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
1. Carefully read both the question and the text.
2. Consider how the question relates to the arguments presented in the text.
3. Assess the question's usefulness in challenging or critically examining the text's content.
4. Determine which of the three labels (Useful, Unhelpful, or Invalid) best applies to the question.

Here is the question you should evaluate:
<critical_question>
{cq}
</critical_question>

And here is the text to which the question relates:
<text>
{intervention}
</text>

###Score Rubrics:
[Is the question posed by the model critical and useful for challenging the arguments in the text?]
Score 1: Invalid--> The question is not appropriately critical in this context. This could be because:
   a) The question is unrelated to the text
   b) The question is too general and could apply to many texts
   c) The question introduces new concepts not mentioned in the text
   d) The question doesn't challenge any arguments in the text (e.g., it's a simple reading comprehension question or asks about the speaker's/reader's opinion)
   e) The question critiques an argument that the speaker wasn't actually making
Score 2: Unhelpful--> The question is critical and related to the text, but not likely to be very useful in challenging its arguments. This could be because:
   a) The answer is common sense
   b) The answer is well-known and not controversial
   c) The question is very complicated to understand or answer
   d) The text itself already answers the question
   Note: Do not use this label just because better questions could have been posed.
Score 3: Useful --> The question is both critical of and directly relevant to the arguments in the text. It challenges the text's content in a meaningful way.

###Feedback: """
    }

def main():
    # arguments
    parser = argparse.ArgumentParser(prog='LLM-as-Judge')
    parser.add_argument('--prompt', default='label', type=str, choices=['label', 'compare', 'combine'])
    parser.add_argument('--model', default='prometheus-eval/prometheus-7b-v2.0', type=str)
    parser.add_argument('--input_path', type=str, default='test.json', help='Path of the test set.')
    parser.add_argument('--output_path', type=str, default='results/', help='Path of the output.')
    parser.add_argument('--submission_path', type=str, default='output.json', help='Path where the generated questions have been saved.')
    parser.add_argument('--submission_name', type=str, default='', help='Name your submission if it does not have a name.')
    args = parser.parse_args()


    #logger
    logging.basicConfig(filename='logs/prom_eval.log', level=logging.INFO)

    # load model
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    mapping = {'1':'Invalid', '2': 'Unhelpful', '3':'Useful'}

    # load the whole dataset
    with open(args.input_path) as f:
        reference=json.load(f)

    with open(args.submission_path) as f:
        new = json.load(f)

    # start the evaluation
    predicted_labels = []
    punctuations = []
    feedback_save = []

    for instance in list(new.keys()): # for each intervention

        punctuation = 0
        if instance in reference.keys(): # do to prevent old keys
            reference_set = [str(i)+': '+ref['cq'] for i,ref in enumerate(reference[instance]['cqs'])]
        else:
            continue

        if new[instance]['cqs'] != 'Missing CQs':
            for i, line in enumerate(new[instance]['cqs']):
                prompts = get_prompts(new[instance]['intervention'], line['cq'], '\n'.join(reference_set))
                #model = ''
                #tokenizer = ''
                out,feedback = run_model(model, tokenizer, prompts[args.prompt])
                #continue
                #print(out, flush=True)

                if args.prompt in ['label', 'combine']:
                    if out in mapping.keys(): # make sure the output is a lable
                        label = mapping[out]
                    else:
                        label = 'evaluation_issue'
                        print(out, flush=True)
            
                if label == 'Useful':
                    punctuation += 1/3
                predicted_labels.append(label)
                new[instance]['cqs'][i]['label'] = label
                new[instance]['cqs'][i]['feedbak'] = feedback

        else:
            print('Missing CQs', flush=True)
            predicted_labels.extend(['missing_cqs', 'missing_cqs', 'missing_cqs']) # this should disapear with a proper prompt that makes sure there are always 3 questions
            

        punctuations.append(punctuation)

    # metrics
    print('Distribution of the labels:', Counter(predicted_labels))
    print('Distribution of the intervention punctuation:', Counter(punctuations))
    print('Overall punctuation', sum(punctuations)/len(punctuations))

    # save the output
    with open(args.output_path+ args.model[:5] +'4_'+args.prompt+'_'+args.submission_name+'.json', 'w') as o:
        json.dump(new, o, indent=4)

    with open('model_train/evaluation/llm-as-judge/evals/evaluations.csv', 'a') as o:
        w = csv.writer(o)
        w.writerow([datetime.now(), args.submission_name, args.model[:5]+'_'+args.prompt, sum(punctuations)/len(punctuations), Counter(predicted_labels)['Useful']/len(predicted_labels), dict(Counter(predicted_labels))])

if __name__ == "__main__":
    main()
