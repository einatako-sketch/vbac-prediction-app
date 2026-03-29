"""
Shared utilities for VBAC Prediction App
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# Standard variable names used throughout the app
ANTENATAL_VARS = {
    "age":              {"label": "Maternal age (years)",                      "type": "numeric"},
    "bmi":              {"label": "BMI (kg/m²)",                               "type": "numeric"},
    "prev_vbac":        {"label": "Number of prior VBACs",                     "type": "numeric"},
    "cs_indication":    {"label": "Indication for prior CD (1-4)",             "type": "categorical"},
    "interval_years":   {"label": "Interpregnancy interval (years)",           "type": "numeric"},
    "gestational_week": {"label": "Gestational age at delivery (weeks)",       "type": "numeric"},
    "gdm":              {"label": "GDM (0=No, 1=Yes)",                        "type": "binary"},
    "pet":              {"label": "Preeclampsia/HDP (0=No, 1=Yes)",           "type": "binary"},
    "iugr":             {"label": "IUGR (0=No, 1=Yes)",                       "type": "binary"},
    "macrosomia":       {"label": "Macrosomia (0=No, 1=Yes)",                 "type": "binary"},
    "start_mode":       {"label": "Labor onset (1=Spontaneous, 3=Induction, 4=Augmentation)", "type": "categorical"},
    "prev_vaginal_cs":  {"label": "Prior vaginal delivery before index CD",    "type": "numeric"},
}

INTRAPARTUM_VARS = {
    "dilation_admission": {"label": "Cervical dilation on admission (cm)",     "type": "numeric"},
    "epidural":           {"label": "Epidural analgesia (0=No, 1=Yes)",        "type": "binary"},
    "oxytocin":           {"label": "Oxytocin use (0=No, 1=Yes)",              "type": "binary"},
    "max_temp":           {"label": "Maximum intrapartum temperature (°C)",    "type": "numeric"},
    "prolonged_2nd":      {"label": "Prolonged second stage (0=No, 1=Yes)",    "type": "binary"},
    "prom":               {"label": "PROM (0=No, 1=Yes)",                      "type": "binary"},
    "meconium":           {"label": "Meconium-stained fluid (0=No, 1=Yes)",    "type": "binary"},
}

CS_INDICATION_LABELS = {
    1: "Antepartum",
    2: "Intrapartum fetal distress",
    3: "First-stage dystocia",
    4: "Second-stage dystocia"
}

START_MODE_LABELS = {
    1: "Spontaneous",
    3: "Induction",
    4: "Augmentation"
}


def build_pipeline(algorithm, use_smote=True):
    smote = SMOTE(random_state=42, k_neighbors=5)
    if algorithm == "Logistic Regression":
        clf = LogisticRegression(max_iter=1000, random_state=42)
        if use_smote:
            return ImbPipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('smote', smote),
                ('scaler', StandardScaler()),
                ('clf', clf)
            ])
        else:
            from sklearn.pipeline import Pipeline
            return Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('clf', clf)
            ])
    elif algorithm == "Random Forest":
        clf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    else:
        clf = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                          max_depth=4, subsample=0.8, random_state=42)
    if use_smote:
        return ImbPipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('smote', smote),
            ('clf', clf)
        ])
    else:
        from sklearn.pipeline import Pipeline
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', clf)
        ])


def train_and_evaluate(X_train, X_test, y_train, y_test, algorithm, feature_names):
    pipe = build_pipeline(algorithm)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    J = tpr - fpr
    opt_thresh = thresholds[np.argmax(J)]
    y_pred = (y_prob >= opt_thresh).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Feature importance
    clf = pipe.named_steps['clf']
    if hasattr(clf, 'feature_importances_'):
        coef = clf.feature_importances_
        importance = pd.Series(coef, index=feature_names) if len(coef) == len(feature_names) else None
    elif hasattr(clf, 'coef_'):
        coef = np.abs(clf.coef_[0])
        importance = pd.Series(coef, index=feature_names) if len(coef) == len(feature_names) else None
    else:
        importance = None

    return {
        'pipeline': pipe,
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
        'test_auc': auc,
        'sensitivity': report['1']['recall'],
        'specificity': report['0']['recall'],
        'ppv': report['1']['precision'],
        'npv': report['0']['precision'],
        'fpr': fpr,
        'tpr': tpr,
        'opt_thresh': opt_thresh,
        'y_prob': y_prob,
        'importance': importance,
    }


def derive_prev_vaginal(df, parity_col, cs_col, vbac_col):
    """Derive prior vaginal delivery before index CD."""
    return (df[parity_col] - df[cs_col] - df[vbac_col].fillna(0)).clip(lower=0)


def compute_grobman(df_ant, y):
    """
    Compute Grobman MFMU nomogram score (without race/ethnicity).
    Published coefficients from Grobman et al. (2007, Obstet Gynecol).
    Returns AUC on the full dataset.
    """
    from sklearn.metrics import roc_auc_score

    intercept = -1.5378
    coefs = {
        "age":           -0.0313,
        "bmi":           -0.0526,
        "prev_vbac":      0.9021,
        "prev_vaginal_cs": 0.7676,
        "cs_indication":  0.0,
    }

    logit = intercept
    for var, beta in coefs.items():
        if var in df_ant.columns:
            col = df_ant[var].fillna(df_ant[var].median())
            logit = logit + beta * col

    prob = 1 / (1 + np.exp(-logit))
    valid = ~prob.isna()
    if valid.sum() < 10:
        return None
    return roc_auc_score(y[valid], prob[valid])
