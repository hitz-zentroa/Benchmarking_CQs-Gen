import json
import logging
import tqdm
import re
import argparse
import anthropic
import yaml
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff



logging.basicConfig()
logger = logging.getLogger()

def output_cqs(model_name, text, prefix, client):
    # how to retry when we reach the limit
    @retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(12))
    def completion_with_backoff(**kwargs):
        return client.messages.create(**kwargs)

    instruction = prefix.format(**{'intervention':text}) 

    message = completion_with_backoff(
                                                model=model_name,
                                                max_tokens=1000,
                                                temperature=0,
                                                messages=[{
                                                    "role": "user",
                                                    "content": instruction
                                                },
                                                {
                                                    "role": "assistant",
                                                    "content": ""
                                                }])
    
    out = message.content[0].text

    return out

def structure_output(whole_text):

    #whole_text = re.sub('</think>.*</think>', '', whole_text) # this is because deepseek tends to output this content

    cqs_list = whole_text.split('\n')
    final = []
    valid = []
    not_valid = []
    for cq in cqs_list:
        if re.match('.*\?(\")?( )?(\([a-zA-Z0-9\.\'\-,\? ]*\))?([a-zA-Z \.,\"\']*)?(\")?$', cq):
            valid.append(cq)
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
            final.append(cq[occurrence.start():])
        else:
            continue

    output = []
    if len(final) >= 3:
        for i in [0, 1, 2]:
            output.append({'id':i, 'cq':final[i]})
        return output
    else:
        logger.warning('Missing CQs in '+whole_text)
        return 'Missing CQs'

def choose_prefix(choice):
    if choice == 'short':
        return ["""Suggest 3 or more critical questions that should be raised before accepting the arguments in this text:
                
"{intervention}"
                
Give one question per line. Make sure there are at least 3 questions. Do not give any explanation regarding why the question is relevant.
"""]
    if choice == 'long':
        return ["""You are tasked with generating critical questions that are useful for diminishing the acceptability of the arguments in the following text:
                
"{intervention}"
                
Take into account a question is not a useful critical question:
- If the question is not related to the text.
- If the question is not specific (for instance, if it's a general question that could be applied to a lot of texts).
- If the question introduces new concepts not mentioned in the text (for instance, if it suggests possible answers).
- If the question is not useful to diminish the acceptability of any argument. For instance, if it's a reading-comprehension question or if it asks about the opinion of the speaker/reader.
- If its answer is not likely to invalidate any of the arguments in the text. This can be because the answer to the question is common sense, or because the text itself answers the question.
                
Output 3 critical questions.                
Give one question per line. 
Make sure there are at least 3 questions.
Do not give any other output.
Do not explain why the questions are relevant.
"""]

def main():
    # arguments
    parser = argparse.ArgumentParser(prog='Prompting')
    parser.add_argument('--models', default="claude-3-5-sonnet-20241022", nargs='+')
    parser.add_argument('--input_path', type=str, default='shared_task/data_splits/', help='Path of the test set.')
    parser.add_argument('--input_set', type=str, default='test', help='Path of the test set.')
    parser.add_argument('--output_path', type=str, default='model_train/prompting/results/', help='Path where the generated questions should be saved.')
    parser.add_argument('--prompt', type=str, default='short', help='Prompt to use.', choices=['short', 'long'])
    args = parser.parse_args()

    # connect to api
    with open('scripts/utils/config.yaml', 'r') as file:
        yaml_config = yaml.safe_load(file)

    client = anthropic.Anthropic(api_key=yaml_config['anthropic']['token'],)

    prefixes = choose_prefix(args.prompt)
    
    with open(args.input_path+args.input_set+'.json') as f:
        data=json.load(f)

    models = args.models

    out = {}
    for model_name in models:
        print(model_name, flush=True)
        #new_params = False
        logger.info(model_name)


        for key,line in tqdm.tqdm(data.items()):
            for prefix in prefixes:
                cqs = output_cqs(model_name, line['intervention'], prefix, client)
                print(cqs)
                line['cqs'] = structure_output(cqs)
                print(line['cqs'])
                out[line['intervention_id']]=line

        with open(args.output_path+model_name.replace(' ', '_')+'_'+args.prompt+'_'+args.input_set+'.json', 'w') as o:
            json.dump(out, o, indent=4)

if __name__ == "__main__":
        main()