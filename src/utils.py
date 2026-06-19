"""
Utility functions for the Multilingual Health QA project.
"""

import re
import pandas as pd


def clean_text(text):
    """
    Clean text while preserving African language special characters.
    - Strips whitespace
    - Removes extra newlines
    - Does NOT lowercase (case sensitive languages)
    """
    if pd.isna(text):
        return text
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def format_submission(test_df, predictions, output_path):
    """
    Format predictions into Zindi submission format.
    All three target columns get the same prediction value.
    """
    import pandas as pd
    submission = test_df[['ID']].copy()
    submission['TargetRLF1'] = predictions
    submission['TargetR1F1'] = predictions
    submission['TargetLLM']  = predictions
    submission.to_csv(output_path, index=False)
    print(f'Submission saved to {output_path}')
    return submission
