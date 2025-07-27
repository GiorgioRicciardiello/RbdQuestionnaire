import pathlib

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Optional, Dict, Tuple, List
import joblib
import os
from pandas import DataFrame
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from scipy.optimize import minimize_scalar
from library.metrics_functions import find_best_threshold_for_predictions,compute_metrics, compute_confidence_interval
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer  # needed to use IterativeImputer
from tabulate import tabulate
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

# %% Helper function to store the predictions, probabilties and true values per model
def create_model_prob_df(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    indices: pd.Index,
    dataset_type: str = "full",
    fold_number: int | None = None,
) -> pd.DataFrame:
    """
    Build a “long” DataFrame with one row per sample.

    Columns will be:
      - model_name
      - dataset_type
      - fold_number
      - true_label
      - prediction
      - predicted_prob    (if y_prob is 1D)
      - OR prob_class_0 … prob_class_{C-1}  (if y_prob is 2D)

    Args:
        model_name:   name of the model (e.g. "XGBoost")
        y_true:       shape = (N,) array of true labels
        y_pred:       shape = (N,) array of predicted labels
        y_prob:       None, or shape = (N,) (binary−prob), or (N, C) for C classes
        indices:      length−N Index (e.g. X_full.index)
        dataset_type: e.g. "full", "val", etc.
        fold_number:  int or None

    Returns:
        pd.DataFrame of shape (N, …) with the appropriate columns.
    """
    df = pd.DataFrame({
        "model_name":   model_name,
        "dataset_type": dataset_type,
        "fold_number":  fold_number,
        "true_label":   y_true,
        "prediction":   y_pred,
        'indices': indices
    }, index=indices)

    if y_prob is None:
        # No probability column
        return df.reset_index(drop=True)

    # If y_prob is 1D, store it under “predicted_prob”
    if y_prob.ndim == 1:
        df["predicted_prob"] = y_prob
        return df.reset_index(drop=True)

    # Otherwise, assume y_prob.ndim == 2 (multiclass)
    n_classes = y_prob.shape[1]
    prob_cols = [f"prob_class_{c}" for c in range(n_classes)]
    prob_df = pd.DataFrame(y_prob, columns=prob_cols, index=indices)
    df = df.join(prob_df)
    return df.reset_index(drop=True)

# %% Xgboost training function
def train_xgboost(train_data: pd.DataFrame,
                  train_labels: np.ndarray,
                  val_data: pd.DataFrame,
                  val_labels: np.ndarray):

    def specificity_loss(preds, dtrain):
        """
        Custom loss to weight negative examples 3.2× harder (i.e. penalize false positives 3.2×).
        """
        weight = 0.1
        labels = dtrain.get_label()
        probs  = 1.0 / (1.0 + np.exp(-preds))   # σ(s)

        # gradient:
        #   if y=1:  ∂L/∂s = –(1 – σ(s))       (standard logistic for positives)
        #   if y=0:  ∂L/∂s =  3.2·σ(s)          (3.2× negative‐class logistic)
        grad = np.where(labels == 1,
                        -(1.0 - probs),
                        weight * probs)

        # Hessian:
        #   if y=1:   ∂²L/∂s² = σ(s)(1 – σ(s))
        #   if y=0:   ∂²L/∂s² = 3.2·σ(s)(1 – σ(s))
        hess = np.where(labels == 1,
                        probs * (1.0 - probs),
                        weight * probs * (1.0 - probs))

        return grad, hess

    def specificity_eval_metric(preds, dtrain):
        """
        Custom evaluation: compute specificity at prob>0.5.
        """
        labels = dtrain.get_label()
        probs  = 1.0 / (1.0 + np.exp(-preds))
        binary = (probs > 0.5).astype(int)
        tn = np.sum((labels == 0) & (binary == 0))
        fp = np.sum((labels == 0) & (binary == 1))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return 'specificity', specificity

    def find_best_threshold_for_predictions(y_true_train: np.ndarray,
                                            y_pred_proba_train: np.ndarray,
                                            metric: str = 'specificity') -> float:
        """
        Given true labels and predicted probabilities (not logits!), find the threshold in [0,1]
        that maximizes the chosen metric.
        """
        def metric_for_threshold(thresh):
            y_pred_thresh = (y_pred_proba_train >= thresh).astype(int)
            if metric == 'f1':
                return -f1_score(y_true_train, y_pred_thresh)
            elif metric == 'accuracy':
                return -accuracy_score(y_true_train, y_pred_thresh)
            elif metric == 'sensitivity':  # Recall
                return -recall_score(y_true_train, y_pred_thresh)
            elif metric == 'specificity':
                tn, fp, fn, tp = confusion_matrix(y_true_train, y_pred_thresh).ravel()
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                return -spec
            elif metric == 'auc':
                return -roc_auc_score(y_true_train, y_pred_thresh)
            else:
                raise ValueError("Unsupported metric: " + metric)

        result = minimize_scalar(metric_for_threshold, bounds=(0.0, 1.0), method='bounded')
        return result.x  # this is “best threshold”

    # ----------------------------
    # 1. Build DMatrices
    # ----------------------------
    dtrain = xgb.DMatrix(train_data, label=train_labels)
    dval   = xgb.DMatrix(val_data,   label=val_labels)

    # ----------------------------
    # 2. XGBoost parameters
    # ----------------------------
    # params = {
    #     'scale_pos_weight': 16,   # (optional: you may want to remove this if you already weight via custom loss)
    #     'max_depth':       12,
    #     'reg_lambda':      0.001,
    #     'gamma':           0.2,
    #     'reg_alpha':       0.1,
    #     # 'objective':   'binary:logistic',  # removed, since we use a custom objective
    #     'eval_metric':     'logloss'
    # }

    params = {
        # 'objective': 'binary:logistic',  # standard and stable
        'eval_metric': 'logloss',  # or 'auc' if you care about ranking
        'max_depth': 3,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42,
        'verbosity': 0
    }

    # ----------------------------
    # 3. Train with custom loss + custom eval
    #    Note: in newer XGB versions use `feval=` for custom eval instead of `custom_metric=`
    # ----------------------------
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        obj=specificity_loss,
        custom_metric=specificity_eval_metric,
        evals=[(dtrain, 'train'), (dval, 'valid')],
        verbose_eval=False
    )

    # ----------------------------
    # 4. Find “best threshold” based on training set specificity
    # ----------------------------
    raw_logits_train = model.predict(dtrain)
    probs_train      = 1.0 / (1.0 + np.exp(-raw_logits_train))  # sigmoid

    best_threshold = find_best_threshold_for_predictions(
        y_true_train=train_labels,
        y_pred_proba_train=probs_train,
        metric='specificity'
    )

    # ----------------------------
    # 5. Produce final predictions on validation set
    # ----------------------------
    raw_logits_val = model.predict(dval)
    probs_val      = 1.0 / (1.0 + np.exp(-raw_logits_val))
    y_pred_val     = (probs_val > best_threshold).astype(int)

    raw_logits_tr = raw_logits_train  # already computed
    y_pred_tr     = (probs_train > best_threshold).astype(int)


    return y_pred_val, probs_val, y_pred_tr, probs_train


# %% Models

models = {
    "Logistic Regression": LogisticRegression(penalty=None,
                                              solver='lbfgs',
                                              max_iter=1000,
                                              C=1),
    "Lasso": LogisticRegression(penalty='l1',
                                solver='liblinear',
                                max_iter=1000),

    "Elastic Net": LogisticRegression(penalty='elasticnet',
                                      solver='saga',
                                      l1_ratio=0.7,
                                      max_iter=1000),
    "LDA": LinearDiscriminantAnalysis(),

    "SVM": SVC(kernel="rbf",
               degree=3,
               gamma="scale",
               probability=True),
    "XGBoost": train_xgboost,

}


# %%

def classify_predictions(predictions: np.ndarray,
                         labels: np.ndarray,
                         indices: np.ndarray,
                         model_name:str,
                         dataset_type: str,
                         fold: Optional[int] = None,
                         ) -> pd.DataFrame:
    """
    Classify each observation as True Positive (TP), True Negative (TN),
    False Positive (FP), or False Negative (FN).

    Parameters:
    - predictions (np.ndarray): Predicted binary labels.
    - labels (np.ndarray): True binary labels.
    - indices (np.ndarray): Original indices of the dataset.
    - fold (int): Fold number.
    - dataset_type (str): Either 'train' or 'validation' to indicate data type.

    Returns:
    - pd.DataFrame: DataFrame with original index and classification labels.
    """
    classification = np.where((labels == 1) & (predictions == 1), 'TP',
                              np.where((labels == 0) & (predictions == 0), 'TN',
                                       np.where((labels == 0) & (predictions == 1), 'FP', 'FN')))

    return pd.DataFrame({
        'model_name': model_name,
        'index': indices,  # Preserve original index
        'true_label': labels,
        'predicted_label': predictions,
        'classification': classification,
        'fold': fold,
        'dataset_type': dataset_type  # 'train' or 'validation'
    })


def fit_sklearn_regression(model, X: pd.DataFrame, y: np.ndarray):
    """Fit a regression problem of sklearn.linear_model.LinearRegression. and get probabilities if available"""
    # 2) Fit on the full dataset
    model.fit(X, y)

    # 3) Predict on the full dataset
    y_pred_full = model.predict(X)

    # 4) If probabilities are needed (e.g., for ElasticNet decision curves):
    if hasattr(model, "predict_proba"):
        try:
            y_prob_full = model.predict_proba(X)[:, 1]
        except Exception:
            y_prob_full = None
    else:
        y_prob_full = None

    return y_prob_full, y_pred_full

def compute_full_training(models:Dict[str, object],
                         df_model:pd.DataFrame,
                         features:list[str],
                         target:str,
                        continuous_features:list[str] = None,):
    """

    :param models:
    :param df_model:
    :param features:
    :param target:
    :return:
    """

    scaler = StandardScaler()
    if continuous_features:
       for feat in continuous_features:
            df_model[feat] = scaler.fit_transform(df_model[[feat]])

    data = df_model[features]
    labels = df_model[target]

    # 2) Prepare X (features) and y (labels)
    y_full = labels.values.ravel()

    # 3) Prepare containers for metrics and classifications
    metrics_records = []
    classification_records = []
    elastic_net_coefs_list = []
    elastic_net_predictions_list = []
    model_prob_records  = []
    # 4) Loop over each model, train on full data, and predict on the same full data
    for model_name, model in models.items():
        X_full = data.copy()

        # OPTIONAL: If your model has special cases (e.g., “Threshold on ESS” or “Okun Tree”),
        # handle them first. Otherwise, do a straight fit‐predict on all data.

        if model_name == "XGBoost":
            # If your XGBoost wrapper expects (train_data, train_labels, val_data, val_labels),
            # you can simply pass the same full‐data for both train & val.
            (y_pred_full,
             y_prob_full,
             y_pred_train_full,
             y_prob_train_full) = models["XGBoost"](
                train_data=X_full,
                train_labels=y_full,
                val_data=X_full,
                val_labels=y_full)

            # We only need the “val” outputs in this scenario since train=val
            metrics = compute_metrics(y_pred=y_pred_full, y_true=y_full)
            metrics.update({'model': model_name})
            metrics_records.append(metrics)

            # Record classification on full data
            full_classification = classify_predictions(
                predictions=y_pred_full,
                labels=y_full,
                indices=X_full.index,
                fold=None,
                dataset_type='full',
                model_name=model_name
            )
            classification_records.append(full_classification)
            # collect the probabilites and predicitons of the models that have that
            df_prob = create_model_prob_df(
                model_name=model_name,
                y_true=y_full,
                y_pred=y_pred_full,
                y_prob=y_prob_full,  # 1D or 2D array
                indices=X_full.index,
                dataset_type="full",
                fold_number=None
            )
            model_prob_records.append(df_prob)

            # If you care about ElasticNet‐style net benefit curves, etc., you could still record
            # y_prob_full here. But since this branch is “XGBoost,” skip ElasticNet logic.
            continue

        # ------------------------------------------------------------
        # For all other scikit‐learn–style estimators (e.g., LogisticRegression,
        # RandomForestClassifier, ElasticNetClassifier, etc.)
        # ------------------------------------------------------------
        # 1) Normalize continuous features if needed (same logic you had for folds)
        if model_name not in ["XGBoost"]:
            y_prob_full, y_pred_full = fit_sklearn_regression(model=model,
                                                              X=X_full,
                                                              y=y_full)

        # 5) Compute metrics on the full dataset
        metrics = compute_metrics(y_pred_full, y_full)
        metrics.update({'model': model_name})
        metrics_records.append(metrics)

        # 6) Record a “classification” row for every sample in X_full
        full_classification = classify_predictions(
            predictions=y_pred_full,
            labels=y_full,
            indices=X_full.index,
            fold=None,
            dataset_type='full',
            model_name=model_name
        )
        classification_records.append(full_classification)

        # 7) Build “long” DF for this model’s probabilities (if y_prob_full is not None)
        if y_prob_full is not None:
            df_prob = create_model_prob_df(
                model_name   = model_name,
                y_true       = y_full,
                y_pred       = y_pred_full,
                y_prob       = y_prob_full,       # could be 1D
                indices      = X_full.index,
                dataset_type = "full",
                fold_number  = None
            )
            model_prob_records.append(df_prob)


        # 7) If it’s Elastic Net, collect coefficients + probability records
        if model_name == 'Elastic Net' and hasattr(model, 'coef_') and not callable(model):
            elastic_net_coefs_list.append(model.coef_.flatten())


    # (A) Concatenate all classification records
    df_classifications = pd.concat(classification_records, ignore_index=True)
    df_classifications = df_classifications.sort_values(
        by=['model_name', 'fold'], na_position='last'
    )

    # (B) Build a DataFrame of aggregated metrics (no fold dimension here)
    df_metrics = pd.DataFrame(metrics_records)

    # (C) Concatenate all “model_prob_records” into a single DataFrame
    if model_prob_records:
        df_models_prob = pd.concat(model_prob_records, ignore_index=True)
    else:
        # If no model ever produced probabilities, return an empty DataFrame
        df_models_prob = pd.DataFrame()

    # (D) If you collected ElasticNet coefficients, build that DataFrame
    if elastic_net_coefs_list:
        feature_names = X_full.columns.tolist()
        df_elastic_net_feature_importance = collect_elastic_net_coefficients_and_std(
            elastic_net_coefs_list,
            feature_names
        )
    else:
        df_elastic_net_feature_importance = pd.DataFrame()

    return (
        df_metrics,  # aggregated metrics per model
        df_classifications,  # per‐sample TP/TN/FN/FP/etc.
        df_elastic_net_feature_importance,  # elastic net feature importances
        df_models_prob  # long‐format “true_label / prediction / prob” table
    )


def compute_cross_validation(models:Dict[str, object],
                             df_model:pd.DataFrame,
                             features:list[str],
                             target:str,
                             k:int=5,
                            continuous_features:list[str] = None,
                             ) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame,DataFrame]:
    """

    For each model_name in `models`, and for each fold in `imputed_folds`:
      1. Preprocess (standardize continuous features) unless special model.
      2. Handle special models ("ESS" threshold, "Okun Tree", "XGBoost") separately.
      3. Fit generic classifiers, predict & compute metrics.
      4. Record per-fold metrics, per-sample classifications, and per-sample probabilities.

    Run the stratified k-fold cross validation of each of the expected models in the dictionary of objects
    present in this library file. Compute the metrics and records which observation is a TP, TN, FN, and FP.
    :param models:
        df_avg_metrics:              DataFrame with average metrics + confidence intervals, sorted by specificity.
        df_classifications:          Concatenated per-sample classification info (TP/TN/FP/FN details).
        df_elastic_net_feature_importance: DataFrame of Elastic Net coefficients and standard errors.
        df_elastic_net_predictions:  Concatenated predictions & probabilities from Elastic Net, per fold.
        df_models_prob:              Long-format DataFrame of all models’ per-sample probabilities, labels, predictions.
    """

    def save_imputed_folds(imputed_folds: list,
                           save_path: str) -> None:
        """
        Saves each fold's imputed data to disk using joblib.

        :param imputed_folds: List of dicts with train/val data and labels.
        :param save_path: Directory where the folds will be saved.
        """
        save_path = pathlib.Path(save_path)
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)

        for i, fold in enumerate(imputed_folds):
            joblib.dump(fold, os.path.join(str(save_path), f"fold_{i}.joblib"))
        print(f"Saved {len(imputed_folds)} folds to {save_path}")


    data = df_model[features]
    labels = df_model[target]

    skf = StratifiedKFold(n_splits=k,
                          shuffle=True,
                          random_state=42)
    fold_indices = list(skf.split(data, labels))
    # Precompute the imputed folds
    folds = []
    scaler = StandardScaler()
    for fold, (train_indices, val_indices) in enumerate(fold_indices):
        print(f'|----Creating Fold {fold}: Train {len(train_indices)}; Validation {len(val_indices)}---|')
        # Get the raw train/val splits
        train_data = data.iloc[train_indices].copy()
        val_data = data.iloc[val_indices].copy()
        train_labels = labels.iloc[train_indices].values.ravel()
        val_labels = labels.iloc[val_indices].values.ravel()

        # Store imputed data and labels for later reuse
        folds.append({
            'train_data': train_data,
            'val_data': val_data,
            'train_labels': train_labels,
            'val_labels': val_labels
        })
    # save_imputed_folds(imputed_folds=imputed_folds, save_path=r'\NarcCataplexyQuestionnaire\data')

    # Containers for all results
    metrics_records = []
    classification_records = []
    elastic_net_coefs_list = []
    elastic_net_predictions_list = []
    model_prob_records = []  # Will hold DataFrames from create_model_prob_df()

    # Iterate over each model and each fold
    for model_name, model in models.items():
        for fold_idx, fold_data in enumerate(folds, start=1):
            # Extract train/validation splits
            train_data = fold_data['train_data'].copy()
            val_data = fold_data['val_data'].copy()
            train_labels = fold_data['train_labels']
            val_labels = fold_data['val_labels']

            # ----------------------------------------
            # 1) STANDARDIZE continuous features
            # ----------------------------------------
            # standardize the continuous features within each fold
            if continuous_features:
                for feat in continuous_features:
                    train_data[feat] = scaler.fit_transform(train_data[[feat]])
                    val_data[feat] = scaler.fit_transform(val_data[[feat]])


            if model_name == "XGBoost":
                # — XGBoost train‐predict wrapper returns (pred_val, prob_val, pred_train, prob_train) —
                (y_pred_val,
                 y_prob_val,
                 y_pred_train,
                 y_prob_train) = models["XGBoost"](
                    train_data=train_data,
                    train_labels=train_labels,
                    val_data=val_data,
                    val_labels=val_labels
                )

                # (a) Metrics on validation set
                metrics = compute_metrics(y_pred_val, val_labels)
                metrics.update({'fold': fold_idx, 'model': model_name})
                metrics_records.append(metrics)

                # (b) Classification details on validation set
                val_classification = classify_predictions(
                    predictions=y_pred_val,
                    labels=val_labels,
                    indices=val_data.index,
                    fold=fold_idx,
                    dataset_type='validation',
                    model_name=model_name
                )
                classification_records.append(val_classification)

                # (c) Record per-sample probabilities & predictions (validation)
                df_prob = create_model_prob_df(
                    model_name=model_name,
                    y_true=val_labels,
                    y_pred=y_pred_val,
                    y_prob=y_prob_val,
                    indices=val_data.index,
                    dataset_type="validation",
                    fold_number=fold_idx
                )
                model_prob_records.append(df_prob)

                continue  # Skip to next fold/model


            # ----------------------------------------
            # 3) GENERIC MODEL: fit on train_data, predict on val_data
            # ----------------------------------------
            model.fit(train_data, train_labels)
            y_pred_val = model.predict(val_data)

            # Extract positive‐class probabilities if available
            if hasattr(model, "predict_proba"):
                try:
                    y_prob_val = model.predict_proba(val_data)[:, 1]
                except Exception:
                    y_prob_val = None
            else:
                y_prob_val = None

            # (a) Compute metrics on validation set
            metrics = compute_metrics(y_pred_val, val_labels)
            metrics.update({'fold': fold_idx, 'model': model_name})
            metrics_records.append(metrics)

            # (b) Per‐sample classification details
            val_classification = classify_predictions(
                predictions=y_pred_val,
                labels=val_labels,
                indices=val_data.index,
                fold=fold_idx,
                dataset_type='validation',
                model_name=model_name
            )
            classification_records.append(val_classification)

            # (c) Record per-sample probabilities & predictions (if probability array exists)
            if y_prob_val is not None:
                df_prob = create_model_prob_df(
                    model_name=model_name,
                    y_true=val_labels,
                    y_pred=y_pred_val,
                    y_prob=y_prob_val,
                    indices=val_data.index,
                    dataset_type="validation",
                    fold_number=fold_idx
                )
                model_prob_records.append(df_prob)

            # (d) If Elastic Net, store coefficients + predictions for net‐benefit curves
            # 7) If it’s Elastic Net, collect coefficients + probability records
            if model_name == 'Elastic Net' and hasattr(model, 'coef_') and not callable(model):
                elastic_net_coefs_list.append(model.coef_.flatten())

                # Collect per‐sample predictions/probabilities for later net‐benefit analysis
                fold_df = create_model_prob_df(
                    model_name=model_name,
                    y_true=val_labels,
                    y_pred=y_pred_val,
                    y_prob=y_prob_val,
                    indices=val_data.index,
                    dataset_type="validation",
                    fold_number=fold_idx
                )
                elastic_net_predictions_list.append(fold_df)

    # ----------------------------------------
    # 4) CONCATENATE RESULTS AFTER LOOP
    # ----------------------------------------

    # (A) All per-sample classification rows
    df_classifications = pd.concat(classification_records, ignore_index=True)
    df_classifications = df_classifications.sort_values(
        by=['model_name', 'fold'], na_position='last'
    )

    # (B) Aggregate per-fold metrics into DataFrame
    df_agg_metrics = pd.DataFrame(metrics_records)
    # Reorder columns: make 'model' and 'fold' appear first
    cols = ['model', 'fold'] + [c for c in df_agg_metrics.columns if c not in ('model', 'fold')]
    df_agg_metrics = df_agg_metrics[cols]

    # (C) Compute confidence intervals for sensitivity & specificity per model
    ci_dict = {}
    metric_ci = ['sensitivity', 'specificity', 'npv', 'ppv']
    for m in df_agg_metrics['model'].unique():
        ci_dict[m] = {}
        for metric in metric_ci:
            vals = df_agg_metrics.loc[df_agg_metrics['model'] == m, metric].values
            ci_dict[m][f'{metric}_ci'] = compute_confidence_interval(vals)
    df_ci = pd.DataFrame.from_dict(ci_dict, orient='index').reset_index().rename(columns={'index': 'model'})

    # (D) Compute average metrics across folds (numeric columns), drop 'fold'
    df_avg_metrics = df_agg_metrics.groupby('model').mean(numeric_only=True).reset_index()
    if 'fold' in df_avg_metrics.columns:
        df_avg_metrics = df_avg_metrics.drop(columns='fold')
    # Merge CIs into the average‐metrics table
    df_avg_metrics = df_avg_metrics.merge(df_ci, on='model', how='left')
    df_avg_metrics = df_avg_metrics.sort_values(by='specificity', ascending=False)

    # (E) Elastic Net: feature importance with standard errors
    if elastic_net_coefs_list:
        # Use last fold's train_data column order for feature names
        feature_names = folds[-1]['train_data'].columns.tolist()
        df_elastic_net_feature_importance = collect_elastic_net_coefficients_and_std(
            elastic_net_coefs_list,
            feature_names
        )
    else:
        df_elastic_net_feature_importance = pd.DataFrame()

    # (F) Elastic Net: concatenated per‐fold predictions & probabilities
    if elastic_net_predictions_list:
        df_elastic_net_predictions = pd.concat(elastic_net_predictions_list, ignore_index=True)
    else:
        df_elastic_net_predictions = pd.DataFrame()

    # (G) All models' per-sample probabilities & predictions (long format)
    if model_prob_records:
        df_models_prob = pd.concat(model_prob_records, ignore_index=True)
    else:
        df_models_prob = pd.DataFrame()

    return (
        df_avg_metrics,
        df_classifications,
        df_elastic_net_feature_importance,
        df_elastic_net_predictions,
        df_models_prob
    )



# Fixing the issue with ElasticNetCV to properly extract feature importance and standard errors

def collect_elastic_net_coefficients_and_std(coefs_list:List[np.ndarray],
                                             feature_names):
    """
    Usage for Logistic Regression model of sklearn
    Given a list of coefficient arrays (one per fold), compute the mean and
    standard error (SE) for each feature.

    Parameters:
    - coefs_list: list of arrays, each of shape (n_features,), list of arrays, each list element are the
        coefficients obtained at each training fold
    - feature_names: list of feature names

    Returns:
    - DataFrame with columns: Feature, Mean Coefficient, Standard Error
    """
    # Stack the coefficient arrays (each row corresponds to a fold)
    coefs_array = np.vstack(coefs_list)
    mean_coefs = np.mean(coefs_array, axis=0)

    # Standard error computed as sample std dev divided by sqrt(n_folds)
    std_errors = np.std(coefs_array, axis=0, ddof=1) / np.sqrt(coefs_array.shape[0])

    # Create DataFrame
    stats_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean Coefficient": mean_coefs,
        "Standard Error": std_errors
    })

    # Optionally, sort by the absolute mean coefficient value (descending)
    stats_df["Abs Mean"] = np.abs(stats_df["Mean Coefficient"])
    stats_df = stats_df.sort_values(by="Abs Mean", ascending=False).drop(columns="Abs Mean")

    return stats_df


def plot_elastic_net_model_coefficients(df_params: pd.DataFrame,
                                        title: str = None,
                                        output_path: pathlib.Path = None,
                                        ):
    """
    Generate a styled plot for the elastic net feature importance coefficients.

    Parameters:
    - df_params: DataFrame with columns 'Feature', 'Mean Coefficient', 'Standard Error'
    - output_path: Path to save the plot (optional)
    """
    import matplotlib as mpl
    # Improve overall font style
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['axes.titleweight'] = 'bold'
    mpl.rcParams['axes.labelsize'] = 14

    plt.figure(figsize=(10, 6))

    # Use a colormap to assign different colors for each feature
    cmap = plt.get_cmap("tab10")
    num_features = df_params.shape[0]
    colors = [cmap(i % 10) for i in range(num_features)]

    # Calculate x-axis limits: max value among (mean coefficient + standard error) plus margin.
    max_val = (df_params["Mean Coefficient"].abs() + df_params["Standard Error"]).max()
    plt.xlim(0, max_val * 1.1)

    # Plot horizontal bar chart with error bars
    plt.barh(df_params["Feature"],
             np.abs(df_params["Mean Coefficient"]),
             xerr=df_params["Standard Error"],
             capsize=5,
             color=colors)

    plt.xlabel("Mean Absolute Coefficient")
    plt.ylabel("Feature")
    if not title:
        title = "Elastic Net Feature Importance Across Folds"
    plt.title(title)
    plt.gca().invert_yaxis()  # Highest importance on top
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path / 'elastic_net_model_coefficients.png', dpi=300)
    plt.show()


def elastic_net_feature_importance_fixed(fitted_model, X, y):
    """
    Computes feature importance for an ElasticNet model and returns a DataFrame
    including standard errors.

    Parameters:
    - model: Trained ElasticNet model
    - X: Feature matrix (numpy array or pandas DataFrame)
    - y: Target variable (numpy array or pandas Series)

    Returns:
    - DataFrame with feature importance and standard errors
    """

    # Extract coefficients across folds
    # coefs = np.array(elastic_cv.path(X, y, l1_ratio=elastic_cv.l1_ratio_)[1])

    # Compute mean and standard deviation of coefficients across cross-validation folds
    feature_importance = np.abs(fitted_model.coef_)
    feature_names = [f"Feature {i}" for i in range(X.shape[1])]

    # mean_coefs = np.abs(coefs.mean(axis=1))
    # std_errors = coefs.std(axis=1)

    # Extract feature importance (absolute coefficients)
    feature_importance = np.abs(model.coef_)
    feature_names = [f"Feature {i}" for i in range(X.shape[1])]

    # Sort features by importance
    sorted_indices = np.argsort(feature_importance)[::-1]  # Sort descending
    sorted_importance = feature_importance[sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices]
    sorted_std_errors = std_errors[sorted_indices]

    # Create DataFrame
    importance_df = pd.DataFrame({
        "Feature": sorted_names,
        "Importance": sorted_importance,
        "Standard Error": sorted_std_errors
    })

    # Plot feature importance with error bars
    plt.figure(figsize=(10, 5))
    plt.barh(sorted_names, sorted_importance, xerr=sorted_std_errors, color='royalblue', capsize=5)
    plt.xlabel("Absolute Coefficient Value")
    plt.ylabel("Feature")
    plt.title("Feature Importance in ElasticNet Model")
    plt.gca().invert_yaxis()  # Highest importance on top
    plt.show()

    return importance_df




def visualize_table(df: pd.DataFrame,
                    group_by: List[str]) -> pd.DataFrame:
    """
    Count the unique pair combinations ina dataframe within the grouped by columns
    :param df:
    :param group_by:
    :return:
    """
    df_copy = df.copy()
    print("Distribution before modification:")
    df_plot_before = df_copy.fillna('NaN')
    grouped_counts_before = df_plot_before.groupby(group_by).size().reset_index(
        name='Counts')
    print(tabulate(grouped_counts_before, headers='keys', tablefmt='grid'))
    print(f'Remaining Rows: {df_copy.shape[0]}')
    return grouped_counts_before


def load_imputed_folds(save_path: str) -> list:
    """
    Loads all saved imputed folds from disk.

    :param save_path: Directory where folds are stored.
    :return: List of dicts for each fold.
    """
    fold_files = sorted(pathlib.Path(save_path).glob("fold_*.joblib"))
    return [joblib.load(f) for f in fold_files]


def summarize_fold_consistency(imputed_folds: list,
                               feature_names: list,
                               target_name: str) -> pd.DataFrame:
    summaries = []

    for fold_idx, fold in enumerate(imputed_folds):
        for split in ['train', 'val']:
            data = fold[f'{split}_data']
            labels = fold[f'{split}_labels']

            for feature in feature_names:
                feature_vals = data[feature]
                summaries.append({
                    'fold': fold_idx,
                    'split': split,
                    'feature': feature,
                    'num_positives': labels.sum(),
                    'mean': feature_vals.mean(),
                    'std': feature_vals.std(),
                    'target_pos_rate': labels.mean()
                })

    return pd.DataFrame(summaries)