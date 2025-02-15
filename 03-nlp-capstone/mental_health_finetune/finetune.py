from datasets import load_dataset
from accelerate import Accelerator
import transformers
from datetime import datetime
import torch
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model

train_dataset = load_dataset('json', data_files='mental_health_train.jsonl', split='train')
eval_dataset = load_dataset('json', data_files='mental_health_val.jsonl', split='train')


def formatting_func(example):
    text = f"### Context: {example['input']}\n ### Response: {example['output']}"
    return text


from transformers import AutoTokenizer, AutoModelForCausalLM

base_model_id = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True, torch_dtype=torch.float16, load_in_8bit=True)


print(model)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
    use_fast=False,
)
tokenizer.pad_token = tokenizer.eos_token


def generate_and_tokenize_prompt(example):
    context = f"### Context: {example['input']}"
    response = f"### Response: {example['output']}"

    # Tokenize both the context and the response
    context_encodings = tokenizer(context, truncation=True, max_length=512, padding="max_length")
    response_encodings = tokenizer(response, truncation=True, max_length=512, padding="max_length")

    # Add the response as the label (target)
    context_encodings["labels"] = response_encodings["input_ids"]
    
    return context_encodings


tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

print(tokenized_train_dataset)



def plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset):
    lengths = [len(x['input_ids']) for x in tokenized_train_dataset]
    lengths += [len(x['input_ids']) for x in tokenized_val_dataset]
    print(len(lengths))

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Length of input_ids')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lengths of input_ids')
    plt.show()

max_length = 512

def generate_and_tokenize_prompt2(prompt):
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt2)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt2)


print(tokenized_train_dataset[1]['input_ids'])


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "Wqkv",
        "fc1",
        "fc2",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)
print(model)

accelerator = Accelerator()
model = accelerator.prepare_model(model)

project = "mental_health_finetune_2"
base_model_name = "phi2"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=500,
        learning_rate=2.5e-5, 
        optim="paged_adamw_8bit",
        logging_steps=25,             
        logging_dir="./logs",        
        save_strategy="steps",      
        save_steps=25,               
        evaluation_strategy="steps", 
        eval_steps=25,               
        do_eval=True,                     
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False 
trainer.train()


