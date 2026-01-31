import pandas as pd
import numpy as np
from datasets import load_dataset

# Load original BoolQ dataset (train split only)
dataset = load_dataset("boolq", split="train")
boolq_df = pd.DataFrame(dataset)

def add_key(df):
    df["key"] = df["question"] + "|||" + df["passage"]
    return df

boolq_df = add_key(boolq_df)
boolq_df.rename(columns={"answer": "actual_answer"}, inplace=True)

# Load prediction files
all_train = add_key(pd.read_csv("all_train.csv"))
modellarge_train = add_key(pd.read_csv("modellarge_train.csv"))
gpt4o1_train = add_key(pd.read_csv('gpt4o(1)_train_results.csv'))

# Base merge
merged = boolq_df[["key", "actual_answer"]].copy()

merged = merged.merge(
    all_train[
        [
            "key",
            "41_pred",
            "r1_pred",
            "4omini_pred",
            "41_mini_pred",
            "phi_pred",
            "mistral_pred",
            "llama_pred",
        ]
    ],
    on="key",
    how="left",
)

merged = merged.merge(
    modellarge_train[["key", "llama405_pred"]],
    on="key",
    how="left",
)

# Merge GPT-4o predictions from its dedicated file (prefer these over all_train's)
merged = merged.merge(gpt4o1_train[['key', '4o_pred']], on='key', how='left')

# Majority voting for custom ensembles
def majority_vote(row, cols):
    votes = row[cols].dropna()
    if len(votes) == 0:
        return np.nan
    return votes.mode()[0]

merged["ensemble1"] = merged.apply(
    lambda r: majority_vote(r, ["phi_pred", "mistral_pred", "llama_pred"]), axis=1
)
merged["ensemble2"] = merged.apply(
    lambda r: majority_vote(r, ["phi_pred", "mistral_pred", "4omini_pred"]), axis=1
)
merged["ensemble3"] = merged.apply(
    lambda r: majority_vote(
        r,
        ["phi_pred", "mistral_pred", "llama_pred", "4omini_pred", "41_mini_pred"],
    ),
    axis=1,
)
merged["ensemble4"] = merged.apply(
    lambda r: majority_vote(r, ["4omini_pred", "41_mini_pred", "phi_pred"]), axis=1
)

# Models to evaluate
models = [
    "4o_pred",
    "41_pred",
    "r1_pred",
    "4omini_pred",
    "41_mini_pred",
    "phi_pred",
    "mistral_pred",
    "llama_pred",
    "llama405_pred",
    "ensemble1",
    "ensemble2",
    "ensemble3",
    "ensemble4",
]

case2_mask = merged[models].notna().all(axis=1)

# Accuracy calculation (single metric)
results = []
for m in models:
    correct = (
        merged.loc[case2_mask, m]
        == merged.loc[case2_mask, "actual_answer"]
    ).mean()
    results.append({"model": m, "acc": correct*100})

results_df = pd.DataFrame(results)

# Rename models for final table
model_names = {
    "4o_pred": "GPT-4o",
    "41_pred": "GPT-4.1",
    "r1_pred": "DeepSeek-R1",
    "4omini_pred": "GPT-4o mini",
    "41_mini_pred": "GPT-4.1 mini",
    "phi_pred": "Phi-3 Small",
    "mistral_pred": "Mistral NeMo",
    "llama_pred": "LLaMA 8B",
    "llama405_pred": "LLaMA 405B",
    "ensemble1": "Ensemble 1",
    "ensemble2": "Ensemble 2",
    "ensemble3": "Ensemble 3",
    "ensemble4": "Ensemble 4",
}

results_df["model"] = results_df["model"].map(model_names)

results_df.to_csv("final_accuracies.csv", index=False)

# -------------------------
# Bootstrap CI (Case 2 only)
# -------------------------

bootstrap_df = merged.loc[
    case2_mask, ["actual_answer", "4o_pred", "ensemble4"]
].copy()

def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()

def bootstrap_accuracy(df, pred_col, n_bootstrap=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(df)
    accs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = df.iloc[idx]
        accs.append(accuracy(sample["actual_answer"], sample[pred_col]))
    return np.array(accs)

boot_gpt4o = bootstrap_accuracy(bootstrap_df, "4o_pred")
boot_ens4 = bootstrap_accuracy(bootstrap_df, "ensemble4")

def summarize(accs):
    return {
        "mean": accs.mean(),
        "ci_lower": np.percentile(accs, 2.5),
        "ci_upper": np.percentile(accs, 97.5),
    }

summary_gpt4o = summarize(boot_gpt4o)
summary_ens4 = summarize(boot_ens4)

print("Case 2 evaluation size:", case2_mask.sum())
print("GPT-4o:", summary_gpt4o)
print("Ensemble 4:", summary_ens4)

# -------------------------
# VALIDATION SPLIT (keep separate)
# -------------------------
print('\nProcessing validation split...')
val_dataset = load_dataset("boolq", split="validation")
boolq_val_df = pd.DataFrame(val_dataset)
boolq_val_df = add_key(boolq_val_df)
boolq_val_df.rename(columns={"answer": "actual_answer"}, inplace=True)

# Load validation prediction files
model12_valid = add_key(pd.read_csv('model12_valid.csv'))
model78_valid = add_key(pd.read_csv('model78_valid.csv'))
model3456_valid = add_key(pd.read_csv('model3456_valid.csv'))
gpt4o1_valid = add_key(pd.read_csv('gpt4o(1)_valid_results.csv'))
modellarge_valid = add_key(pd.read_csv('modellarge_valid.csv'))

# Build merged validation frame starting from BoolQ validation set
merged_val = boolq_val_df[['key', 'actual_answer']].copy()

merged_val = merged_val.merge(
    gpt4o1_valid[['key', '4o_pred']], on='key', how='left')
merged_val = merged_val.merge(
    model12_valid[['key', '4o_pred', 'r1_pred']], on='key', how='left', suffixes=(None, '_m12'))
# prefer existing column names (avoid duplicate 4o_pred)
if '4o_pred_m12' in merged_val.columns:
    merged_val['r1_pred'] = merged_val['r1_pred'].fillna(merged_val.get('r1_pred'))

merged_val = merged_val.merge(
    model78_valid[['key', '41_mini_pred', '41_pred']], on='key', how='left')
merged_val = merged_val.merge(
    model3456_valid[['key', '4omini_pred', 'phi_pred', 'mistral_pred', 'llama_pred']], on='key', how='left')
merged_val = merged_val.merge(
    modellarge_valid[['key', 'llama405_pred']], on='key', how='left')

# Create ensemble predictions (majority vote) for validation
merged_val['ensemble1'] = merged_val.apply(
    lambda r: majority_vote(r, ['phi_pred', 'mistral_pred', 'llama_pred']), axis=1)
merged_val['ensemble2'] = merged_val.apply(
    lambda r: majority_vote(r, ['phi_pred', 'mistral_pred', '4omini_pred']), axis=1)
merged_val['ensemble3'] = merged_val.apply(
    lambda r: majority_vote(r, ['phi_pred', 'mistral_pred', 'llama_pred', '4omini_pred', '41_mini_pred']), axis=1)
merged_val['ensemble4'] = merged_val.apply(
    lambda r: majority_vote(r, ['4omini_pred', '41_mini_pred', 'phi_pred']), axis=1)

# Models to evaluate on validation (same set)
models_val = [
    '4o_pred', '41_pred', 'r1_pred', '4omini_pred', '41_mini_pred',
    'phi_pred', 'mistral_pred', 'llama_pred', 'llama405_pred',
    'ensemble1', 'ensemble2', 'ensemble3', 'ensemble4'
]

# Case 2 mask for validation: only rows where all models (the same set) answered
case2_mask_val = merged_val[models_val].notna().all(axis=1)

results_val = []
for m in models_val:
    if case2_mask_val.sum() > 0:
        correct = (merged_val.loc[case2_mask_val, m] == merged_val.loc[case2_mask_val, 'actual_answer']).mean()
    else:
        correct = np.nan
    results_val.append({'model': m, 'acc': float(correct*100) if not np.isnan(correct) else np.nan})

results_val_df = pd.DataFrame(results_val)
results_val_df['model'] = results_val_df['model'].map(model_names)

# Save validation results separately
results_val_df.to_csv('final_accuracies_valid.csv', index=False)
print('Saved validation accuracies to final_accuracies_valid.csv')
print('Validation Case 2 evaluation size:', case2_mask_val.sum())
