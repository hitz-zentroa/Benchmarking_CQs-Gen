import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, MistralForCausalLM, LlamaForCausalLM, GenerationConfig
import logging
import csv
import sys

logging.basicConfig()
logger = logging.getLogger()

def output_cqs(model_name, text, prefix, speaker, model, tokenizer, new_params, remove_instruction=False):
    if model_name == 'HuggingFaceH4/zephyr-7b-beta':
        instruction = "f\""+prefix + "</s>\n<|user|>\n" + speaker + ': ' + text + "</s>\n<|assistant|>\""
    else:
        instruction = prefix + '\n\"' + speaker + ': ' + text + '\"\n\n'
    inputs = tokenizer(instruction, return_tensors="pt")
    inputs = inputs.to('cuda')
    if new_params:
        outputs = model.generate(**inputs, **new_params) 
    else:
        outputs = model.generate(**inputs)
    out = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    if remove_instruction:
        try:
            out = out.split('<|assistant|>')[1]
        except IndexError:
            out = out[len(instruction):]
    return out

def check_if_any_cq(line):
    #print(line.keys())
    if line['theory_cqs']:
        return True
    return False

def main():
    dataset = sys.argv[1]
    model_set = sys.argv[2]
    #model_set = sys.argv[3] # small or big

    prefixes = ["List the critical questions that should be asked regarding the arguments in the following paragraph:",
                "Suggest which critical questions should be raised before accepting the arguments in this text:"
                #"Critical questions are the set of enquiries that should be asked in order to judge if an argument is good or fallacious by unmasking the assumptions held by the premises of the argument. List the critical questions that should be asked regarding the arguments in the following paragraph:"
                ]

    with open('data/theory_cqs_by_intervention/'+dataset+'.json') as f:
        data=json.load(f)

    if model_set == 'llama-3':
        models = ['meta-llama/Meta-Llama-3-8B-Instruct']
    elif model_set == 'llama-2':
        models = ["meta-llama/Llama-2-7b-chat-hf"]
    elif model_set == 'llama-3-big':
        models = ['meta-llama/Meta-Llama-3-70B-Instruct'] 
    else:
        print('model_set should be either big or small')
        exit()

    with open('data/llm_generation/'+model_set+'/'+dataset+'.csv', 'w') as o:
        w = csv.writer(o)
        w.writerow(['iden', 'model', 'prefix', 'intervention', 'output'])
        for model_name in models:
            new_params = False
            logger.warning(model_name)
            generation_config = GenerationConfig.from_pretrained(model_name)
            logger.warning(generation_config)
            if model_name == "HuggingFaceH4/zephyr-7b-beta":
                model = MistralForCausalLM.from_pretrained(model_name, device_map="auto")
                remove_instruction = True
                new_params = {
                                        "temperature": 0.9,
                                        "top_p": 0.95,
                                        "repetition_penalty": 1.2,
                                        "top_k": 50,
                                        "max_new_tokens": 1024
                                    }
                #logger.warning(generation_config)
            elif model_name.startswith("meta-llama"):
                model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto")
                remove_instruction = True
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
                remove_instruction = False

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.warning('Loaded '+model_name)
            for key,line in data.items():
                speaker = line['id'].split('_')[0]
                if len(line['ARGS']) > 0: # esto ya verifica que hay argumentos
                    #if check_if_any_cq(line): # por qué lo hacemos dos veces?
                    for prefix in prefixes:
                        #logger.warning(prefix)
                        text = line['INTERVENTION']
                        cqs = output_cqs(model_name, text, prefix, speaker, model, tokenizer, new_params, remove_instruction)
                        #logger.warning(cqs)
                        w.writerow([line['id'], model_name, prefix, text, cqs])

if __name__ == "__main__":
        main()