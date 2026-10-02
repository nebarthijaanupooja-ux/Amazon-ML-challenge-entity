import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score
from rapidfuzz import fuzz

# Determine project root directory dynamically
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists("dataset"):
    PROJECT_ROOT = os.getcwd()
else:
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))

TRAIN_DIR = os.path.join(PROJECT_ROOT, "dataset", "train")
TEST_DIR = os.path.join(PROJECT_ROOT, "dataset", "test")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def preprocess_df(df):
    df = df.copy()
    df['business_name'] = df['business_name'].fillna('').astype(str)
    df['business_address'] = df['business_address'].fillna('').astype(str)
    df['country'] = df['country'].fillna('').astype(str)
    df['clean_name'] = df['business_name'].apply(clean_text)
    return df

def extract_features(s1_names, s1_addrs, s1_countries, c_names, c_addrs, c_countries, tfidf_sims):
    n_samples = len(s1_names)
    features = np.zeros((n_samples, 5))
    
    for i in range(n_samples):
        # 1. TF-IDF Cosine Similarity
        features[i, 0] = tfidf_sims[i]
        
        # 2. Token Set Ratio (Name Similarity)
        features[i, 1] = fuzz.token_set_ratio(s1_names[i], c_names[i]) / 100.0
        
        # 3. Partial Ratio (Address Similarity)
        if s1_addrs[i] and c_addrs[i]:
            features[i, 2] = fuzz.partial_ratio(s1_addrs[i], c_addrs[i]) / 100.0
        else:
            features[i, 2] = 0.5
            
        # 4. Token Sort Ratio (Address Variation)
        if s1_addrs[i] and c_addrs[i]:
            features[i, 3] = fuzz.token_sort_ratio(s1_addrs[i], c_addrs[i]) / 100.0
        else:
            features[i, 3] = 0.5
            
        # 5. Country Exact Match Boolean
        c1, c2 = s1_countries[i].strip().lower(), c_countries[i].strip().lower()
        if c1 and c2:
            features[i, 4] = 1.0 if c1 == c2 else 0.0
        else:
            features[i, 4] = 1.0
            
    return features

print("="*60)
print("🚀 STEP 1: Loading Training Dataset & Ground Truth...")
print("="*60)

tr_s1 = preprocess_df(pd.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), sep="\t"))
tr_s2 = preprocess_df(pd.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), sep="\t"))
tr_s3 = preprocess_df(pd.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), sep="\t"))
tr_gt = pd.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), sep="\t")

# Map Ground Truth Matches
gt_dict = {}
for _, row in tr_gt.iterrows():
    s1_id = row['source1_entity_id']
    m_ids = str(row['matched_entity_ids']).split(',') if pd.notna(row['matched_entity_ids']) and row['matched_entity_ids'] != "" else []
    gt_dict[s1_id] = set(m_ids)

tr_candidates = pd.concat([tr_s2, tr_s3], ignore_index=True)
cand_lookup = tr_candidates.set_index('entity_id').to_dict('index')

print("Fitting TF-IDF Vectorizer on Training Names...")
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=1)
all_train_names = pd.concat([tr_s1['clean_name'], tr_candidates['clean_name']])
vectorizer.fit(all_train_names)

s1_tfidf_tr = vectorizer.transform(tr_s1['clean_name'])
cand_tfidf_tr = vectorizer.transform(tr_candidates['clean_name'])

print("\n" + "="*60)
print("⚙️ STEP 2: Building Training Pair Dataset & Feature Extraction...")
print("="*60)

train_rows = []
cand_entities_tr = tr_candidates['entity_id'].values

for idx, row in tr_s1.iterrows():
    s1_id = row['entity_id']
    actual_matches = gt_dict.get(s1_id, set())
    
    sims = (s1_tfidf_tr[idx] @ cand_tfidf_tr.T).toarray()[0]
    top_indices = np.argsort(sims)[-10:][::-1]
    
    for c_idx in top_indices:
        if sims[c_idx] < 0.1:
            continue
        c_id = cand_entities_tr[c_idx]
        is_match = 1 if c_id in actual_matches else 0
        
        c_info = cand_lookup[c_id]
        train_rows.append({
            's1_name': row['business_name'],
            's1_addr': row['business_address'],
            's1_country': row['country'],
            'c_name': c_info['business_name'],
            'c_addr': c_info['business_address'],
            'c_country': c_info['country'],
            'tfidf_sim': sims[c_idx],
            'label': is_match
        })

train_df = pd.DataFrame(train_rows)
print(f"Generated {len(train_df)} candidate pairs for training.")
print(f"Positive matches in training pairs: {train_df['label'].sum()}")

X_train = extract_features(
    train_df['s1_name'].values, train_df['s1_addr'].values, train_df['s1_country'].values,
    train_df['c_name'].values, train_df['c_addr'].values, train_df['c_country'].values,
    train_df['tfidf_sim'].values
)
y_train = train_df['label'].values

print("\n" + "="*60)
print("🤖 STEP 3: Training Random Forest & Optimizing F_0.5 Metric...")
print("="*60)

clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

probs = clf.predict_proba(X_train)[:, 1]

# Threshold tuning for F_0.5 score maximization
best_th, best_f05 = 0.5, 0.0
for th in np.arange(0.3, 0.95, 0.05):
    preds = (probs >= th).astype(int)
    f05 = fbeta_score(y_train, preds, beta=0.5, zero_division=0)
    if f05 > best_f05:
        best_f05 = f05
        best_th = th

print(f"✓ Model Trained Successfully!")
print(f"🎯 Optimal Probability Threshold for F_0.5 Score: {best_th:.2f}")
print(f"🏆 Training F_0.5 Score: {best_f05:.4f}")

print("\n" + "="*60)
print("⚡ STEP 4: Test Dataset Inference (Generating Output Files)...")
print("="*60)

te_s1 = preprocess_df(pd.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), sep="\t"))
te_s2 = preprocess_df(pd.read_csv(os.path.join(TEST_DIR, "test_source2.tsv"), sep="\t"))
te_s3 = preprocess_df(pd.read_csv(os.path.join(TEST_DIR, "test_source3.tsv"), sep="\t"))

te_candidates = pd.concat([te_s2, te_s3], ignore_index=True)
te_cand_entities = te_candidates['entity_id'].values

vectorizer_te = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=1)
all_test_names = pd.concat([te_s1['clean_name'], te_candidates['clean_name']])
vectorizer_te.fit(all_test_names)

s1_tfidf_te = vectorizer_te.transform(te_s1['clean_name'])
cand_tfidf_te = vectorizer_te.transform(te_candidates['clean_name'])

cand_results = []
match_results = []

batch_size = 5000
total_test = len(te_s1)

te_s1_names = te_s1['business_name'].values
te_s1_addrs = te_s1['business_address'].values
te_s1_countries = te_s1['country'].values

te_c_names = te_candidates['business_name'].values
te_c_addrs = te_candidates['business_address'].values
te_c_countries = te_candidates['country'].values

for start in range(0, total_test, batch_size):
    end = min(start + batch_size, total_test)
    print(f"Processing Test Batch {start} to {end} of {total_test}...")
    sim_matrix = (s1_tfidf_te[start:end] @ cand_tfidf_te.T).toarray()
    
    for i in range(end - start):
        g_idx = start + i
        s1_id = te_s1['entity_id'].values[g_idx]
        sims = sim_matrix[i]
        
        top_idx = np.argsort(sims)[-10:][::-1]
        top_idx = [idx for idx in top_idx if sims[idx] >= 0.2]
        
        cand_ids = [te_cand_entities[idx] for idx in top_idx]
        cand_results.append({'source1_entity_id': s1_id, 'candidate_entity_ids': ",".join(cand_ids)})
        
        if not top_idx:
            match_results.append({'source1_entity_id': s1_id, 'matched_entity_ids': ""})
            continue
            
        n_cands = len(top_idx)
        test_feats = extract_features(
            [te_s1_names[g_idx]] * n_cands,
            [te_s1_addrs[g_idx]] * n_cands,
            [te_s1_countries[g_idx]] * n_cands,
            [te_c_names[idx] for idx in top_idx],
            [te_c_addrs[idx] for idx in top_idx],
            [te_c_countries[idx] for idx in top_idx],
            [sims[idx] for idx in top_idx]
        )
        
        test_probs = clf.predict_proba(test_feats)[:, 1]
        
        matched_ids = []
        for c_i, prob in enumerate(test_probs):
            if prob >= best_th:
                matched_ids.append(te_cand_entities[top_idx[c_i]])
                
        match_results.append({'source1_entity_id': s1_id, 'matched_entity_ids': ",".join(matched_ids)})

cand_df = pd.DataFrame(cand_results)
match_df = pd.DataFrame(match_results)

cand_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
match_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")

cand_df.to_csv(cand_path, sep="\t", index=False)
match_df.to_csv(match_path, sep="\t", index=False)

print(f"\n✓ Saved: {cand_path}")
print(f"✓ Saved: {match_path}")

val_script = os.path.join(PROJECT_ROOT, "utils", "validate_submission.py")
if os.path.exists(val_script):
    print("\n🔍 Running Official Amazon ML Challenge Validator...")
    os.system(f"py -3.12 {val_script}")

print("\n🎉 ML Model Training & Test Inference Pipeline Completed Successfully!")