"""
BM25-based retrieval for the Multilingual Health QA project.

Unlike embedding-based retrieval (src/retrieval.py), BM25 uses exact
keyword/term overlap rather than semantic similarity. Since ROUGE rewards
literal word overlap with reference answers, BM25's term-matching bias
may align better with the scoring metric than semantic embeddings do.
"""

import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
import re

def clean_prediction(text):
    text = re.sub(r'^[^\x00-\x7F\u0100-\u024F\u1E00-\u1EFF]+', '', str(text))
    return text.strip()

def simple_tokenize(text):
    return str(text).split()


if __name__ == '__main__':
    print("Loading data...")
    train = pd.read_csv('data/Train.csv')
    test  = pd.read_csv('data/Test.csv')
    sample_sub = pd.read_csv('data/SampleSubmission.csv')

    test['subset'] = test['ID'].apply(lambda x: '_'.join(x.split('_')[2:4]))

    print("Running language-aware BM25 retrieval...")
    predictions = {}

    for lang in test['subset'].unique():
        print(f"Processing: {lang}")
        test_lang  = test[test['subset'] == lang]
        train_lang = train[train['subset'] == lang]

        if len(train_lang) == 0:
            train_lang = train

        tokenized_corpus = [simple_tokenize(q) for q in train_lang['input'].tolist()]
        bm25 = BM25Okapi(tokenized_corpus)

        for _, row in test_lang.iterrows():
            query_tokens = simple_tokenize(row['input'])
            scores = bm25.get_scores(query_tokens)
            best_idx = np.argmax(scores)
            predictions[row['ID']] = clean_prediction(train_lang['output'].iloc[best_idx])

    pred_df = pd.DataFrame(list(predictions.items()), columns=['ID', 'prediction'])
    submission = sample_sub.merge(pred_df, on='ID')
    submission['TargetRLF1'] = submission['prediction']
    submission['TargetR1F1'] = submission['prediction']
    submission['TargetLLM']  = submission['prediction']
    submission = submission[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]
    submission = submission.fillna('No answer available')

    submission.to_csv('outputs/submissions/submission_bm25.csv', index=False)
    print('Done! Saved submission_bm25.csv')
