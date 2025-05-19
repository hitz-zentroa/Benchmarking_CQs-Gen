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
from transformers import AutoTokenizer, AutoModelForCausalLM#, Gemma3ForConditionalGeneration#, GenerationConfig, AutoProcessor#, Qwen2_5_VLForConditionalGeneration


logger = logging.getLogger(__name__)


def run_model(model, tokenizer, prompt):
    chat = [{"role": "user", "content": prompt}] 
    chat_formated = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    #print(chat_formated, flush=True)
    inputs = tokenizer(chat_formated, return_tensors="pt")

    inputs = inputs.to('cuda')

    generated_ids = model.generate(**inputs, max_new_tokens=512) 

    #generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)] # this does not work for Gemma

    out = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    #print(out, flush=True)

    try:
        output = out.split('model\n')[1].replace('\n', '')
    except IndexError:
        print('EVAL ERROR: '+output, flush=True)

    #import pdb; pdb.set_trace()
    output = output.strip()

    return output

def get_prompts(intervention, cq, references):
    return {
        'compare': f"""You will be given a set of reference questions, each with an identifying ID, and a newly generated question. Your task is to determine if any of the reference questions are asking for the same information as the new question.
        
Here is the set of reference questions with their IDs:

<reference_questions>
{references}
</reference_questions>

Here is the newly generated question:

<new_question>
{cq}
</new_question>

Compare the new question to each of the reference questions. Look for questions that are asking for the same information, even if they are worded differently. Consider the core meaning and intent of each question, not just the exact wording.

If you find a reference question that is asking for the same information as the new question, output only the ID of that reference question.

If none of the reference questions are asking for the same information as the new question, output exactly 'Similar reference not found.' (without quotes).

Your final output should consist of only one of the following:
1. The ID of the most similar reference question
2. The exact phrase 'Similar reference not found.'

Do not include any explanation, reasoning, or additional text in your output.""",
        'label': f"""You are a fair judge assistant tasked with evaluating if a provided question is a useful critical question for a given text. Your role is to provide clear objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance.

Here is the question you should evaluate:
<critical_question>
{cq}
</critical_question>

And here is the text to which the question relates:
<text>
{intervention}
</text>

Guidelines for evaluation:
1. Carefully read both the question and the text.
2. Consider how the question relates to the arguments presented in the text.
3. Assess the question's usefulness in challenging or critically examining the text's content.
4. Determine which of the three labels (Useful, Unhelpful, or Invalid) best applies to the question.

Label criteria:
1. Useful: The question is both critical of and directly relevant to the arguments in the text. It challenges the text's content in a meaningful way.

2. Unhelpful: The question is critical and related to the text, but not likely to be very useful in challenging its arguments. This could be because:
   a) The answer is common sense
   b) The answer is well-known and not controversial
   c) The question is very complicated to understand or answer
   d) The text itself already answers the question
   Note: Do not use this label just because better questions could have been posed.

3. Invalid: The question is not appropriately critical in this context. This could be because:
   a) The question is unrelated to the text
   b) The question is too general and could apply to many texts
   c) The question introduces new concepts not mentioned in the text
   d) The question doesn't challenge any arguments in the text (e.g., it's a simple reading comprehension question or asks about the speaker's/reader's opinion)
   e) The question critiques an argument that the speaker wasn't actually making

Your task is to output only one of the three labels: Useful, Unhelpful, or Invalid. Do not include any comments, explanations, blank spaces, or new lines. Your entire output should consist of a single word - the chosen label."""}

def main():
    # arguments
    parser = argparse.ArgumentParser(prog='LLM-as-Judge')
    parser.add_argument('--prompt', default='compare', type=str, choices=['label', 'compare', 'combine'])
    parser.add_argument('--model', default='google/gemma-2-9b-it', type=str)
    parser.add_argument('--input_path', type=str, default='test.json', help='Path of the test set.')
    parser.add_argument('--output_path', type=str, default='results/', help='Path of the output.')
    parser.add_argument('--submission_path', type=str, default='output.json', help='Path where the generated questions have been saved.')
    parser.add_argument('--submission_name', type=str, default='', help='Name your submission if it does not have a name.')
    args = parser.parse_args()


    #logger
    logging.basicConfig(filename='logs/gem_eval.log', level=logging.INFO)

    # load model
    if args.model == 'google/gemma-3-12b-it':
        model = Gemma3ForConditionalGeneration.from_pretrained(args.model, device_map="auto", attn_implementation='eager')
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", attn_implementation='eager')
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    # load the whole dataset
    with open(args.input_path) as f:
        reference=json.load(f)

    with open(args.submission_path) as f:
        new = json.load(f)

    # start the evaluation
    predicted_labels = []
    punctuations = []

    for instance in list(new.keys()): # for each intervention

        punctuation = 0
        if instance in reference.keys():
            reference_set = [str(i)+': '+ref['cq'] for i,ref in enumerate(reference[instance]['cqs'])]
        else:
            continue

        if new[instance]['cqs'] != 'Missing CQs':
            for i, line in enumerate(new[instance]['cqs']):
                prompts = get_prompts(new[instance]['intervention'], line['cq'], '\n'.join(reference_set))
                out = run_model(model, tokenizer, prompts[args.prompt])
                #print(out, flush=True)

                if args.prompt in ['label', 'combine']:
                    if out in ['Useful', 'Unhelpful', 'Invalid']: # make sure the output is a lable --> remember to have strip() in the generation
                        label = out
                    else:
                        label = 'evaluation_issue'
                        print('evaluation_issue: '+out, flush=True)

                if args.prompt == 'compare':
                    winner = out 
                    try: # here make sure the output is the id of a reference cq
                        if winner.strip() != 'Similar reference not found.':
                            label = reference[instance]['cqs'][int(winner)]['label']
                        else: 
                            label = 'not_able_to_evaluate'
                    except IndexError:
                        label = 'evaluation_issue'
                        #print(out, flush=True)
                    except ValueError:
                        label = 'evaluation_issue'
                        #print(out, flush=True)
                #print(label)

                if label == 'Useful':
                    punctuation += 1/3
                predicted_labels.append(label)
                new[instance]['cqs'][i]['label'] = label

        else:
            print('Missing CQs', flush=True)
            predicted_labels.extend(['missing_cqs', 'missing_cqs', 'missing_cqs']) # this should disapear with a proper prompt that makes sure there are always 3 questions
            

        punctuations.append(punctuation)

    # metrics
    print('Distribution of the labels:', Counter(predicted_labels))
    print('Distribution of the intervention punctuation:', Counter(punctuations))
    print('Overall punctuation', sum(punctuations)/len(punctuations))

    # save the output
    with open(args.output_path+'gemma_'+args.prompt+'_'+args.submission_name+'.json', 'w') as o:
        json.dump(new, o, indent=4)

    with open('model_train/evaluation/llm-as-judge/evals/evaluations.csv', 'a') as o:
        w = csv.writer(o)
        w.writerow([datetime.now(), args.submission_name, 'gemma_'+args.prompt, sum(punctuations)/len(punctuations), Counter(predicted_labels)['Useful']/len(predicted_labels), dict(Counter(predicted_labels))])

if __name__ == "__main__":
    main()
