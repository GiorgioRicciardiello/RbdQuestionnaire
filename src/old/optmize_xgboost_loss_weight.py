"""
==============================================================================
Script: Optimize fp_weight Hyperparameter for XGBoost (Training Fold Evaluation Only)
------------------------------------------------------------------------------

Description:
    This script implements k-fold cross validation to optimize the 'fp_weight'
    hyperparameter used in a custom XGBoost loss function. In this evaluation,
    model performance (specificity and sensitivity) is computed using only the
    training folds. The validation fold is not used for evaluating performance.

    The custom loss function applies a penalty on false positives, controlled
    by the 'fp_weight' parameter. By varying this parameter, the script identifies
    the value that maximizes specificity on the training data.

Usage:
    - Adjust the 'weight_candidates' list as needed.
    - Run the script to visualize how 'fp_weight' impacts both specificity and sensitivity,
      and to determine the optimal value based solely on the training fold.

Author: Giorgio Ricciardiello
Date: 2024
==============================================================================
"""


import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold
from scipy.optimize import minimize_scalar
import pathlib
from config.config import config
from typing import List, Optional
import matplotlib.pyplot as plt

def train_xgboost(train_data: pd.DataFrame,
                  train_labels: np.ndarray,
                  val_data: pd.DataFrame,
                  val_labels: np.ndarray,
                  fp_weight: float = 3.2):
    """
    Train an XGBoost model using a custom loss function that incorporates a false positive penalty.

    Parameters:
    - train_data, train_labels: Training set.
    - val_data, val_labels: Validation set.
    - fp_weight: Weight to penalize false positives.

    Returns:
    - y_pred_val: Binary predictions for the validation set.
    - y_pred_prob_val: Predicted probabilities for the validation set.
    - y_pred_train: Binary predictions for the training set.
    - y_pred_prob_train: Predicted probabilities for the training set.
    """

    def specificity_loss(preds, dtrain):
        labels = dtrain.get_label()
        preds = 1 / (1 + np.exp(-preds))  # Convert logits to probabilities
        # Use fp_weight to penalize false positives
        grad = -labels * (1 - preds) + (1 - labels) * preds * fp_weight
        hess = preds * (1 - preds) * (1 + (1 - labels))
        return grad, hess

    def specificity_eval_metric(preds, dtrain):
        labels = dtrain.get_label()
        preds = (preds > 0.5).astype(int)  # Threshold at 0.5
        tn = np.sum((labels == 0) & (preds == 0))
        fp = np.sum((labels == 0) & (preds == 1))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return 'specificity', specificity

    def find_best_threshold_for_predictions(y_true_train: np.ndarray,
                                            y_pred_train: np.ndarray,
                                            metric: str = 'specificity') -> float:
        # Function to find the threshold that maximizes the desired metric
        def metric_for_threshold(threshold):
            y_pred_thresh = (y_pred_train >= threshold).astype(int)
            if metric == 'f1':
                return -f1_score(y_true_train, y_pred_thresh)
            elif metric == 'accuracy':
                return -accuracy_score(y_true_train, y_pred_thresh)
            elif metric == 'sensitivity':
                return -recall_score(y_true_train, y_pred_thresh)
            elif metric == 'specificity':
                tn, fp, fn, tp = confusion_matrix(y_true_train, y_pred_thresh).ravel()
                specificity = tn / (tn + fp)
                return -specificity
            elif metric == 'auc':
                return -roc_auc_score(y_true_train, y_pred_thresh)
            else:
                raise ValueError("Unsupported metric.")

        result = minimize_scalar(metric_for_threshold, bounds=(0.0, 1.0), method='bounded')
        return result.x

    # XGBoost parameters remain mostly fixed
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

    dtrain = xgb.DMatrix(train_data, label=train_labels)
    dval = xgb.DMatrix(val_data, label=val_labels)

    model = xgb.train(params,
                      dtrain,
                      num_boost_round=100,
                      custom_metric=specificity_eval_metric,
                      obj=specificity_loss,
                      evals=[(dtrain, 'train'), (dval, 'valid')],
                      verbose_eval=False)

    best_threshold = find_best_threshold_for_predictions(train_labels, model.predict(dtrain), metric='specificity')

    y_pred_val = (model.predict(dval) > best_threshold).astype(int)
    y_pred_prob_val = model.predict(dval)
    y_pred_train = (model.predict(dtrain) > best_threshold).astype(int)
    y_pred_prob_train = model.predict(dtrain)

    return y_pred_val, y_pred_prob_val, y_pred_train, y_pred_prob_train


def optimize_fp_weight(train_data: pd.DataFrame,
                       train_labels: np.ndarray,
                       k: int = 5,
                       weight_candidates:List[float]=None,
                       output_path:Optional[pathlib.Path]=None):
    """
    Optimize the fp_weight hyperparameter via k‑fold cross validation. We only use the training folds to find the best
    weight.

    Parameters:
    - train_data, train_labels: The full training dataset.
    - k: Number of folds for cross validation.
    - weight_candidates: List of fp_weight values to try.

    Returns:
    - best_weight: The fp_weight value that achieved the highest mean specificity.
    - best_cv_score: The highest mean specificity observed.
    """
    if weight_candidates is None:
        weight_candidates = [1.0, 2.0, 3.2, 4.0, 5.0,6,7,8,9,10]
    best_weight = None
    best_cv_score = -np.inf
    specificity_scores = []
    sensitivity_scores = []
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for weight in weight_candidates:
        cv_specificity = []
        cv_sensitivity = []
        for train_idx, val_idx in kf.split(train_data):
            X_train, X_val = train_data.iloc[train_idx], train_data.iloc[val_idx]
            y_train, y_val = train_labels[train_idx], train_labels[val_idx]

            # Train the model using the current fp_weight candidate
            y_pred_val, _, y_pred_train, _ = train_xgboost(X_train, y_train, X_val, y_val, fp_weight=weight)

            # Compute specificity on the validation fold
            tn = np.sum((y_train == 0) & (y_pred_train == 0))
            fp = np.sum((y_train == 0) & (y_pred_train == 1))
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            cv_specificity.append(specificity)

            # Compute sensitivity on the validation fold
            tp = np.sum((y_train == 1) & (y_pred_train == 1))
            fn = np.sum((y_train == 1) & (y_pred_train == 0))
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            cv_sensitivity.append(sensitivity)

        mean_specificity = np.mean(cv_specificity)
        mean_sensitivity = np.mean(cv_sensitivity)

        specificity_scores.append(mean_specificity)
        sensitivity_scores.append(mean_sensitivity)

        print(f"fp_weight: {weight}, Mean Specificity: {mean_specificity:.4f}")

        if mean_specificity > best_cv_score:
            best_cv_score = mean_specificity
            best_weight = weight

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(weight_candidates,
             specificity_scores,
             marker='o',
             label='Specificity')
    plt.plot(weight_candidates,
             sensitivity_scores,
             marker='o',
             label='Sensitivity')
    plt.xlabel('FP Weight')
    plt.ylabel('Metric Value')
    plt.title('Specificity and Sensitivity vs. FP Weight')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.show()


    return best_weight, specificity_scores, sensitivity_scores


def train_xgboost_visualize_sens_spec_each_boosting_round(train_data: pd.DataFrame,
                  train_labels: np.ndarray,
                  val_data: pd.DataFrame,
                  val_labels: np.ndarray,
                  specificity_thresh: float = 0.8,
                  sensitivity_thresh: float = 0.6,
                num_boost_round:int=100):
    """

    :param train_data:
    :param train_labels:
    :param val_data:
    :param val_labels:
    :param specificity_thresh:
    :param sensitivity_thresh:
    :param num_boost_round:
    :return:
    """
    def specificity_loss(preds, dtrain):
        """
        Custom loss to weight negative examples 3.2× harder (i.e. penalize false positives 3.2×).
        """
        weight = 0.5
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

    def compute_specificity_sensitivity(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return specificity, sensitivity

    dtrain = xgb.DMatrix(train_data, label=train_labels)
    dval   = xgb.DMatrix(val_data, label=val_labels)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 3,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42,
        'verbosity': 0
    }

    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        obj=specificity_loss,
        evals=[(dtrain, 'train'), (dval, 'valid')],
        evals_result=evals_result,
        verbose_eval=False
    )

    # Compute specificity/sensitivity at each boosting round
    spec_vals, sens_vals = [], []
    best_round = None

    for i in range(1, num_boost_round + 1):
        pred_probs = model.predict(dval, iteration_range=(0, i))
        y_pred_bin = (pred_probs > 0.5).astype(int)
        spec, sens = compute_specificity_sensitivity(val_labels, y_pred_bin)
        spec_vals.append(spec)
        sens_vals.append(sens)
        if spec >= specificity_thresh and sens >= sensitivity_thresh:
            best_round = i if best_round is None else best_round

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_boost_round + 1), spec_vals, label='Specificity')
    plt.plot(range(1, num_boost_round + 1), sens_vals, label='Sensitivity')
    plt.axhline(specificity_thresh, color='gray', linestyle='--', label='Specificity Threshold')
    plt.axhline(sensitivity_thresh, color='gray', linestyle=':', label='Sensitivity Threshold')
    if best_round:
        plt.axvline(best_round, color='red', linestyle='--', label=f'Best Round: {best_round}')
    plt.xlabel('Boosting Round')
    plt.ylabel('Metric Value')
    plt.title('Validation Specificity & Sensitivity per Boosting Round')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return best_round, spec_vals, sens_vals

if __name__ == "__main__":
    # %% Read data
    df_data = pd.read_csv(config.get('data_path').get('pp_questionnaire'))
    base_path = config.get('results_path').get('results').joinpath('xgboost_loss_weight')
    base_path.mkdir(parents=True, exist_ok=True)

    # %% Select columns and drop columns with nans
    target = 'diagnosis'

    categorical_var = [col for col in df_data.columns if col.startswith('q')]
    categorical_var.append('gender')

    continuous_var = ['age']
    columns = list(set(categorical_var + continuous_var + [target]))

    df_data.reset_index(drop=True, inplace=True)
    df_data = df_data.reindex(sorted(df_data.columns), axis=1)
    print(f'Dataset dimension: {df_data.shape}')

    # %% data splits
    train_data = df_data[[col for col in columns if col != target]]
    train_labels = df_data[target]
    # %% define the weigh candidates
    # Log-spaced values below 1
    low_weights = np.logspace(-3, 0, num=10, endpoint=False)  # from 0.001 to just below 1.0
    # Linearly spaced values above 1
    high_weights = np.linspace(1.0, 5.0, num=5)
    # Combine and round for readability (optional)
    weight_candidates = np.round(np.concatenate([low_weights, high_weights]), 4).tolist()
    #%% Optimize the fp_weight hyperparameter

    best_weight, specificity_scores, sensitivity_scores = optimize_fp_weight(train_data,
                                                 train_labels,
                                                 k=5,
                                                 weight_candidates=weight_candidates,
                                                 output_path=base_path.joinpath('xgboost_loss_weight.png'))

    df_weights = pd.DataFrame({'specificity_scores': specificity_scores,
                               'sensitivity_scores': sensitivity_scores,
                               'low_weights': weight_candidates})
    df_weights.sort_values(by='specificity_scores', ascending=False, inplace=True)
    df_weights.to_csv(base_path.joinpath('base_path.csv'), index=False)

    # %% Visualize the booring round and the sensitivity/specifcity changes
    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(df_data,
                                        test_size=0.2,
                                        random_state=42,
                                        stratify=df_data[target])

    # Separate features and target
    train_data = train_df[[col for col in columns if col != target]]
    train_labels = train_df[target]

    val_data = val_df[[col for col in columns if col != target]]
    val_labels = val_df[target]

    # Call your training function
    num_boost_round = 18
    best_round, spec_vals, sens_vals = train_xgboost_visualize_sens_spec_each_boosting_round(
        train_data=train_data,
        train_labels=train_labels,
        val_data=val_data,
        val_labels=val_labels,
        specificity_thresh=0.8,
        sensitivity_thresh=0.6,
        num_boost_round=num_boost_round
    )
    np_boosting = np.linspace(start=1, stop=num_boost_round, num=num_boost_round)

    df_sens_spec_boosting = pd.DataFrame({'sens': sens_vals,
                                          'spec': spec_vals,
                                        'boosting': np_boosting})
    df_sens_spec_boosting.to_csv(base_path.joinpath('sens_spec_boosting.csv'), index=False)

