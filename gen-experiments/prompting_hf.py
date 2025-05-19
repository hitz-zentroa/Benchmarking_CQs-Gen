import json
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoProcessor#, Qwen2_5_VLForConditionalGeneration
import logging
import tqdm
import re
import argparse
import torch


logging.basicConfig()
logger = logging.getLogger()

def output_cqs(model_name, text, prefix, model, tokenizer, new_params, remove_instruction=True):

    instruction = prefix.format(**{'intervention':text}) 

    if 'Qwen' in model_name:
        messages = [{"role": "user",
                    "content": [{"type": "text", "text": instruction,},],}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
            )
    else:
        chat = [{"role": "user", "content": instruction}] 
        chat_formated = tokenizer.apply_chat_template(chat, tokenize=False)
        inputs = tokenizer(chat_formated, return_tensors="pt")


    inputs = inputs.to('cuda')

    if new_params:
        generated_ids = model.generate(**inputs, **new_params) # use if we want to give specific params
    else:
        generated_ids = model.generate(**inputs, max_new_tokens=512) 

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)] # remove everything that is not newly generated

    out = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return out

def structure_output(whole_text):

    whole_text = re.sub('</think>.*</think>', '', whole_text) # this is because deepseek tends to output this content

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
    parser.add_argument('--models', default='meta-llama/Meta-Llama-3-8B-Instruct', nargs='+')
    parser.add_argument('--input_path', type=str, default='shared_task/data_splits/', help='Path of the test set.')
    parser.add_argument('--input_set', type=str, default='test', help='Path of the test set.')
    parser.add_argument('--output_path', type=str, default='model_train/prompting/results/', help='Path where the generated questions should be saved.')
    parser.add_argument('--prompt', type=str, default='short', help='Prompt to use.', choices=['short', 'long'])
    args = parser.parse_args()

    prefixes = choose_prefix(args.prompt)
    
    with open(args.input_path+args.input_set+'.json') as f:
        data=json.load(f)

    models = args.models

    out = {}
    for model_name in models:
        print(model_name, flush=True)
        new_params = False
        logger.info(model_name)
        generation_config = GenerationConfig.from_pretrained(model_name)
        logger.info(generation_config)
        
        
        if 'Qwen' in model_name:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, device_map="auto")
            tokenizer = AutoProcessor.from_pretrained(model_name)
        elif 'gemma' in model_name:
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", attn_implementation='eager')
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16) # , attn_implementation="flash_attention_2"
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        logger.info('Loaded '+model_name)
        for key,line in tqdm.tqdm(data.items()):
            for prefix in prefixes:
                cqs = output_cqs(model_name, line['intervention'], prefix, model, tokenizer, new_params)
                line['cqs'] = structure_output(cqs)
                out[line['intervention_id']]=line

        with open(args.output_path+model_name.split('/')[1]+'_'+args.prompt+'_'+args.input_set+'.json', 'w') as o:
            json.dump(out, o, indent=4)

if __name__ == "__main__":
        main()