from datasets import load_dataset
import json

dataset = load_dataset("Amod/mental_health_counseling_conversations")

def convert_to_jsonl(split):
    data = dataset[split]
    
    output_file = f"mental_health_{split}.jsonl"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            converted_item = {
                "input": item['Context'],
                "output": item['Response']
            }
            f.write(json.dumps(converted_item, ensure_ascii=False) + '\n')
    
    return output_file

splits = dataset.keys()
for split in splits:
    output_file = convert_to_jsonl(split)
    print(f"Created {output_file}")

def preview_file(filename, num_lines=3):
    with open(filename, 'r', encoding='utf-8') as f:
        for _ in range(num_lines):
            print(f.readline().strip())

if 'train' in splits:
    print("\nPreview of training file:")
    preview_file('mental_health_train.jsonl')