import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import xgboost as xgb
import joblib


def load_data():

    dataset = load_dataset("bwbayu/job_cv_supervised")

    df = dataset['train'].to_pandas().reset_index(drop=True)

    del df['Unnamed: 0']

    df = df.rename(columns={'clean_cv': 'resume_text'})
    df = df.rename(columns={'clean_jd': 'job_description_text'})

    return df


def train_model(df):

    
    X = df[['resume_text', 'job_description_text']]
    y = df.label.values

    # --- 1. Helper Functions ---
    def combine_text(X):
        return X['resume_text'] + " [SEP] " + X['job_description_text']

    def split_resume(X):
        return X['resume_text']

    def split_jd(X):
        return X['job_description_text']

    def cosine_similarity_feature(X):
        import numpy as np
        from sklearn.preprocessing import normalize

        n = X.shape[1] // 2
        resume_vec = X[:, :n]
        jd_vec     = X[:, n:]
        resume_vec = normalize(resume_vec)
        jd_vec     = normalize(jd_vec)
        cos_sim = np.array((resume_vec.multiply(jd_vec)).sum(axis=1)).reshape(-1,1)
        return cos_sim


    # --- 3. Pipeline ---

    resume_tfidf = Pipeline([
        ('select', FunctionTransformer(split_resume, validate=False)),
        ('tfidf', TfidfVectorizer(max_features=5000))
    ])

    jd_tfidf = Pipeline([
        ('select', FunctionTransformer(split_jd, validate=False)),
        ('tfidf', TfidfVectorizer(max_features=5000))
    ])

    cosine_pipeline = Pipeline([
        ('tfidfs', FeatureUnion([
            ('resume', resume_tfidf),
            ('jd', jd_tfidf)
        ])),
        ('cosine', FunctionTransformer(cosine_similarity_feature, validate=False))
    ])

    combined_tfidf_pipeline = Pipeline([
        ('combine', FunctionTransformer(combine_text, validate=False)),
        ('tfidf', TfidfVectorizer(max_features=5000))
    ])


    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('combined_tfidf', combined_tfidf_pipeline),
            ('cosine', cosine_pipeline)
        ])),
        ('xgb', xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.15,
            min_child_weight=2,
            objective='binary:logistic',
            eval_metric='auc',
            random_state=1,
            n_jobs=-1
        ))
    ])


    # --- 4. Fit pipeline ---
    pipeline.fit(X, y)

    return pipeline


# --- 6. Save pipeline ---
def save_model(pipeline, output_file):
    with open(output_file, 'wb') as f_out:
        joblib.dump(pipeline, f_out)


df = load_data()
pipeline = train_model(df)
save_model(pipeline, 'model.bin')

print('Model saved to model.bin')