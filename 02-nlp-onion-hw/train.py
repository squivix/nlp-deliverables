import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def train_one_epoch(model, train_loader, optimizer, device):
    """Train model for one epoch."""
    model.train()
    for batch_x, batch_y in train_loader:
        if isinstance(batch_x, torch.Tensor):
            batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        logits = model(batch_x)
        loss = model.loss_function(logits, batch_y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def evaluate_model(model, test_loader, device):
    """Evaluate model and return loss & metrics."""
    model.eval()
    all_preds, all_labels, all_logits = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for x_test, y_test in test_loader:
            y_test = y_test.to(device)
            if isinstance(x_test, torch.Tensor):
                x_test = x_test.to(device)
            test_logits = model(x_test)
            test_loss = model.loss_function(test_logits, y_test)
            test_preds = model.predict(test_logits)

            total_loss += test_loss.item()
            all_labels.append(y_test.squeeze().cpu().numpy())
            all_preds.append(test_preds.squeeze().cpu().numpy())
            all_logits.append(test_logits.squeeze().cpu().numpy())
    all_logits = np.concat(all_logits)
    all_preds = np.concat(all_preds)
    all_labels = np.concat(all_labels)
    # Compute evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)
    pr_auc = roc_auc_score(all_labels, all_logits)

    return {
        "loss": total_loss / len(test_loader),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "pr_auc": pr_auc
    }


def kfold_train_eval(model, dataset, device, k=5, learning_rate=0.001, weight_decay=0.0,
                     max_epochs=20, batch_size=32, early_stopper=None, verbose=True):
    metrics = ["loss", "accuracy", "precision", "recall", "f1", "mcc", "pr_auc"]
    test_metrics = {m: [] for m in metrics}
    kfold = StratifiedKFold(n_splits=k, shuffle=True)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset, dataset.labels)):
        train_dataset = Subset(dataset, train_ids)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_ids), batch_size=batch_size, shuffle=False)

        model.apply(weight_reset)
        model = model.to(device)

        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if early_stopper:
            early_stopper.reset()

        # Training loop
        if verbose:
            pbar = tqdm(total=max_epochs, desc=f"Fold {fold}")
        for epoch in range(max_epochs):
            train_one_epoch(model, train_loader, optimizer, device)
            test_results = evaluate_model(model, test_loader, device)

            if early_stopper and early_stopper(test_results["loss"]):
                break
            if verbose:
                pbar.update(1)
        if verbose:
            pbar.close()

        final_metrics = evaluate_model(model, test_loader, device)
        for m in metrics:
            test_metrics[m].append(final_metrics[m])
        if verbose:
            print(f"Fold {fold}, epochs {epoch + 1}: {", ".join([f'{k}:{v:f}' for k, v in final_metrics.items()])}")

    return {f"test_{m}": np.mean(test_metrics[m]) for m in metrics}
