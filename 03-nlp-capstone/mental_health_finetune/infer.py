import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

base_model_id = "microsoft/phi-2"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Phi2, same as before
    device_map="auto",
    trust_remote_code=True,
    load_in_8bit=True,
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
    use_fast=False,
)
tokenizer.pad_token = tokenizer.eos_token

eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True, use_fast=False)
eval_tokenizer.pad_token = tokenizer.eos_token

from peft import PeftModel

ft_model = PeftModel.from_pretrained(base_model, "phi2-mental_health_finetune_2/checkpoint-500")

# eval_prompt = "My brother has been diagnosed with paranoid schizophrenia and has not been taking his medication. He's been using methamphetamine and alcohol and was found sleeping naked in my step mom driveway in 12 degree weather.\r\n\r\nI was adopted in by his dad (who just passed) and his mother will not Get involved because she's afraid of financial responsibility. \r\n\r\nDo I have the rights to be able to sign my brother into mentalhealth facility?"
eval_prompt = "I'm always afraid that people will reject me, so I avoid social situations. How can I build confidence and stop fearing rejection?"
model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")

ft_model.eval()
with torch.no_grad():
    print(eval_tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=256, repetition_penalty=1.11)[0], skip_special_tokens=True))
