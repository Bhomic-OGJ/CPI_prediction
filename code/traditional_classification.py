#!/usr/bin/env python3
"""
Traditional fingerprints + protein features classification script.

Usage:
  python traditional_classification.py --data ../dataset/b_cancer/original/data.txt --out_dir ../output/result/

The script will:
 - Parse the data file (SMILES, protein sequence, label)
 - Compute molecular fingerprints (RDKit if available, else a hashed n-gram fallback)
 - Compute protein features (AA composition + dipeptide frequencies)
 - Combine features, run classifiers, and save evaluation results
"""
import os
import sys
import argparse
import csv
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False


AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")


def parse_data(filepath):
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            if len(parts) >= 3:
                # assume first is SMILES, second is sequence, last is label
                smiles = parts[0]
                seq = parts[1]
                label = parts[-1]
            else:
                parts = ln.split('\t')
                if len(parts) >= 3:
                    smiles, seq, label = parts[0], parts[1], parts[2]
                else:
                    # try comma
                    parts = ln.split(',')
                    if len(parts) >= 3:
                        smiles, seq, label = parts[0], parts[1], parts[2]
                    else:
                        # give up on this line
                        continue
            try:
                y = int(float(label))
            except Exception:
                # last try: label might be last token if there are >3 tokens
                try:
                    y = int(float(parts[-1]))
                except Exception:
                    continue
            records.append((smiles, seq, y))
    df = pd.DataFrame(records, columns=["smiles", "sequence", "label"]) if records else pd.DataFrame(columns=["smiles","sequence","label"])
    return df


def mol_fingerprint_rdkit(smiles, nbits=1024, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.int8)
    try:
        from rdkit.DataStructs import ConvertToNumpyArray
        ConvertToNumpyArray(fp, arr)
    except Exception:
        onbits = list(fp.GetOnBits())
        arr[onbits] = 1
    return arr


def mol_fingerprint_fallback(smiles, nbits=1024, ngram=3):
    # simple hashed n-gram counts over the SMILES string
    s = smiles
    counts = np.zeros(nbits, dtype=np.float32)
    for k in range(1, ngram + 1):
        for i in range(len(s) - k + 1):
            token = s[i:i+k]
            h = abs(hash(token)) % nbits
            counts[h] += 1
    # normalize
    if counts.sum() > 0:
        counts = counts / np.linalg.norm(counts)
    return counts


def compute_molecular_features(smiles_list, nbits=1024):
    feats = []
    for smi in smiles_list:
        if RDKit_AVAILABLE:
            try:
                arr = mol_fingerprint_rdkit(smi, nbits=nbits)
                if arr is None:
                    arr = mol_fingerprint_fallback(smi, nbits=nbits)
            except Exception:
                arr = mol_fingerprint_fallback(smi, nbits=nbits)
        else:
            arr = mol_fingerprint_fallback(smi, nbits=nbits)
        feats.append(arr)
    return np.vstack(feats)


def aa_composition(seq):
    seq = seq.upper()
    L = len(seq)
    comp = np.zeros(len(AA_LIST), dtype=np.float32)
    if L == 0:
        return comp
    counts = Counter(seq)
    for i, aa in enumerate(AA_LIST):
        comp[i] = counts.get(aa, 0) / L
    return comp


def dipeptide_freq(seq):
    seq = seq.upper()
    idx_map = {}
    idx = 0
    for a in AA_LIST:
        for b in AA_LIST:
            idx_map[a + b] = idx
            idx += 1
    vec = np.zeros(400, dtype=np.float32)
    total = 0
    for i in range(len(seq) - 1):
        pair = seq[i:i+2]
        if pair in idx_map:
            vec[idx_map[pair]] += 1
            total += 1
    if total > 0:
        vec = vec / total
    return vec


def compute_protein_features(seq_list):
    comps = [aa_composition(s) for s in seq_list]
    dipeps = [dipeptide_freq(s) for s in seq_list]
    return np.hstack([np.vstack(comps), np.vstack(dipeps)])


def evaluate_classifiers(X, y, out_dir, random_state=42):
    os.makedirs(out_dir, exist_ok=True)
    results = []
    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=1000, solver='lbfgs'),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "SVC": SVC(probability=True, gamma='scale')
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for name, clf in classifiers.items():
        aucs = []
        accs = []
        precs = []
        recs = []
        f1s = []
        for train_idx, test_idx in skf.split(X, y):
            Xtr, Xte = X[train_idx], X[test_idx]
            ytr, yte = y[train_idx], y[test_idx]
            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(Xtr)
            Xte_s = scaler.transform(Xte)
            clf.fit(Xtr_s, ytr)
            probs = clf.predict_proba(Xte_s)[:, 1]
            preds = (probs >= 0.5).astype(int)
            aucs.append(roc_auc_score(yte, probs))
            accs.append(accuracy_score(yte, preds))
            precs.append(precision_score(yte, preds, zero_division=0))
            recs.append(recall_score(yte, preds, zero_division=0))
            f1s.append(f1_score(yte, preds, zero_division=0))

        res = {
            "classifier": name,
            "auc_mean": np.mean(aucs),
            "auc_std": np.std(aucs),
            "accuracy_mean": np.mean(accs),
            "precision_mean": np.mean(precs),
            "recall_mean": np.mean(recs),
            "f1_mean": np.mean(f1s)
        }
        results.append(res)

    out_path = os.path.join(out_dir, "traditional_classification_results.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join("..","dataset","b_cancer","original","data.txt"))
    p.add_argument("--out_dir", default=os.path.join("..","output","result"))
    p.add_argument("--nbits", type=int, default=1024)
    args = p.parse_args()

    df = parse_data(args.data)
    if df.empty:
        print("No records parsed from data file. Check format.")
        sys.exit(1)

    print(f"Parsed {len(df)} records")

    X_mol = compute_molecular_features(df['smiles'].tolist(), nbits=args.nbits)
    X_prot = compute_protein_features(df['sequence'].tolist())
    X = np.hstack([X_mol, X_prot])
    y = df['label'].values.astype(int)

    print("Feature shapes: mol", X_mol.shape, "prot", X_prot.shape, "combined", X.shape)

    results = evaluate_classifiers(X, y, args.out_dir)
    print("Results saved to", os.path.join(args.out_dir, "traditional_classification_results.csv"))
    for r in results:
        print(r)


if __name__ == '__main__':
    main()
