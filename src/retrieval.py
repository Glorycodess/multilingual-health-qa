"""
Retrieval-based answer generation for the Multilingual Health QA project.

Given a test question, finds the most similar question(s) in the training
set using sentence embeddings, and returns the corresponding training
answer(s) as the prediction. This approach was found to substantially
outperform fine-tuning alone for this task (see Experiments 10-14).
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re


def clean_prediction(text):
    """Strip leading non-Latin/African-script characters that can appear
    from encoding artifacts, while preserving valid special characters
    (ɔ, ɛ, ɲ, ŋ, ɣ, etc.)."""
    text = re.sub(r'^[^\x00-\x7F\u0100-\u024F\u1E00-\u1EFF]+', '', str(text))
    return text.strip()


def retrieve_answers(
    train,
    test,
    embedding_model_name='paraphrase-multilingual-mpnet-base-v2',
    language_aware=True,
    top_k=1,
    batch_size=32
):
    """
    Retrieve the most similar training answer(s) for each test question.

    Args:
        train: DataFrame with columns ['input', 'output', 'subset']
        test: DataFrame with columns ['ID', 'input'] (subset will be
              derived from ID if language_aware=True)
        embedding_model_name: HuggingFace sentence-transformers model name.
            'paraphrase-multilingual-MiniLM-L12-v2' (faster, Experiments 10-12)
            'paraphrase-multilingual-mpnet-base-v2' (stronger, Experiment 13 - best result)
        language_aware: if True, only retrieve from training examples in the
            same language subset as the test question (Experiment 11+)
        top_k: number of top matches to return. top_k=1 returns the single
            best match (best-performing setting); top_k>1 concatenates
            multiple matches (tested in Experiment 12, found to underperform)
        batch_size: encoding batch size

    Returns:
        dict mapping test ID -> predicted answer string
    """
    print(f'Loading embedding model: {embedding_model_name}')
    model = SentenceTransformer(embedding_model_name)

    test = test.copy()
    if language_aware:
        test['subset'] = test['ID'].apply(lambda x: '_'.join(x.split('_')[2:4]))
        languages = test['subset'].unique()
    else:
        test['subset'] = 'all'
        languages = ['all']

    predictions = {}

    for lang in languages:
        print(f'Processing: {lang}')
        test_lang = test[test['subset'] == lang]

        if language_aware:
            train_lang = train[train['subset'] == lang]
            if len(train_lang) == 0:
                print(f'  No training data for {lang}, using full training set')
                train_lang = train
        else:
            train_lang = train

        test_emb = model.encode(test_lang['input'].tolist(), batch_size=batch_size, show_progress_bar=False)
        train_emb = model.encode(train_lang['input'].tolist(), batch_size=batch_size, show_progress_bar=False)

        sims = cosine_similarity(test_emb, train_emb)

        if top_k == 1:
            best_idx = np.argmax(sims, axis=1)
            for i, idx in enumerate(best_idx):
                test_id = test_lang['ID'].iloc[i]
                predictions[test_id] = clean_prediction(train_lang['output'].iloc[idx])
        else:
            top_k_idx = np.argsort(sims, axis=1)[:, -top_k:][:, ::-1]
            for i, indices in enumerate(top_k_idx):
                test_id = test_lang['ID'].iloc[i]
                combined = ' '.join(clean_prediction(train_lang['output'].iloc[idx]) for idx in indices)
                predictions[test_id] = combined

    return predictions


def save_submission(predictions, sample_sub, output_path):
    """Format predictions dict into Zindi submission format and save."""
    pred_df = pd.DataFrame(list(predictions.items()), columns=['ID', 'prediction'])
    submission = sample_sub.merge(pred_df, on='ID')
    submission['TargetRLF1'] = submission['prediction']
    submission['TargetR1F1'] = submission['prediction']
    submission['TargetLLM'] = submission['prediction']
    submission = submission[['ID', 'TargetRLF1', 'TargetR1F1', 'TargetLLM']]
    submission = submission.fillna('No answer available')
    submission.to_csv(output_path, index=False)
    print(f'Submission saved to {output_path}')
    return submission


if __name__ == '__main__':
    train = pd.read_csv('data/Train.csv')
    test = pd.read_csv('data/Test.csv')
    sample_sub = pd.read_csv('data/SampleSubmission.csv')

    # Best-performing configuration (Experiment 13): language-aware, mpnet, top_k=1
    predictions = retrieve_answers(
        train, test,
        embedding_model_name='paraphrase-multilingual-mpnet-base-v2',
        language_aware=True,
        top_k=1
    )
    save_submission(predictions, sample_sub, 'outputs/submissions/submission_retrieval_final.csv')
