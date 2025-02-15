# Onion Or Not
This is the repo for the Onion or not classification challenge.

Reproduce by cloneing this repo, installing all dependencies and running `main.ipynb`. A previous run was saved in `main.html`.

Final F1 score using five fold cv is 82.08648%

# Approach
We used a transformer classifier with learned token and positional embeddings and a BPE tokenizer. No part is pretrained.

Here is the architecture:

```
CustomModel(
  (token_embedding): Embedding(30000, 512)
  (position_embedding): Embedding(512, 512)
  (model): Sequential(
    (0): AttentionLayer(
      (multihead_attn): MultiheadAttention(
        (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
      )
    )
    (1): ReLU()
    (2): Dropout(p=0.25, inplace=False)
  )
  (linear): MLPBinaryClassifier(
    (model): Sequential(
      (0): Linear(in_features=512, out_features=2048, bias=True)
      (1): ReLU()
      (2): Dropout(p=0.25, inplace=False)
      (3): Linear(in_features=2048, out_features=2048, bias=True)
      (4): ReLU()
      (5): Dropout(p=0.25, inplace=False)
      (6): Linear(in_features=2048, out_features=2048, bias=True)
      (7): ReLU()
      (8): Dropout(p=0.25, inplace=False)
      (9): Linear(in_features=2048, out_features=1, bias=True)
      (10): Sigmoid()
    )
  )
)
```


# Results
```
Accuracy: 0.86775
Precision: 0.8274413221370001
Recall: 0.8155555555555555
F1: 0.8208648736727469
MCC: 0.7167883718582126
PR AUC: 0.9346598703703703
```

# Group Work Contribution
```
Nayan Sharma: 0%
Omer Ahmed: 100%
Zeyi Lu: 0%
```

