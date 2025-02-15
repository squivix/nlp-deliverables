import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from flask import Flask, render_template, request
from summa import summarizer
from summa import keywords
from flask import render_template, request, jsonify
import torch
import re

app = Flask(__name__)

base_model_id = "microsoft/phi-2"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
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

ft_model = PeftModel.from_pretrained(base_model, "../phi2-mental_health_finetune_2/checkpoint-500")

import re

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    response = ""
    bold_ranges = []
    
    if request.method == "POST":
        feeling_input = request.form["feeling_input"]
        
        prompt = f"### Context: {feeling_input}, How can I overcome this situation?\n### Response:"

        model_input = eval_tokenizer(prompt, return_tensors="pt").to("cuda")

        ft_model.eval()
        with torch.no_grad():
            resp = ft_model.generate(**model_input, max_new_tokens=256, repetition_penalty=1.11)
            generated_response = eval_tokenizer.decode(resp[0], skip_special_tokens=True)
            
            response_start = generated_response.find("### Response:")
            if response_start != -1:
                response = generated_response[response_start + len("### Response:"):].strip()
            else:
                response = generated_response

            unwanted_words = ["Response", "Context", "Output", "Response:", "Context:", "Output:"]
            response = re.sub(r'\b(?:' + '|'.join(unwanted_words) + r')\b', '', response, flags=re.IGNORECASE).strip()

            response = re.sub(r'\d+\.\s+', '', response)

            important_phrases = summarizer.summarize(response)
            keywords_list = keywords.keywords(response).split("\n") 

            keywords_list = [word for word in keywords_list if len(word) > 3]

            for phrase in important_phrases.split("."):
                phrase = phrase.strip()
                if phrase:
                    start_idx = response.find(phrase)
                    end_idx = start_idx + len(phrase)

                    bold_ranges.append({"start": start_idx, "end": end_idx})


            response = re.sub(r'###\s*', '', response)
            updated_resposne = jsonify({
            "text": response,
            "boldRanges": bold_ranges
        })  

        return updated_resposne

    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(debug=True)

        
