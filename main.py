"""
main.py

Main driver script for the iRBD classification project.
Runs the full pipeline: preprocessing → modeling → evaluation → figure generation.

"""

import subprocess

def run_script(script_name):
    print(f"\nRunning {script_name}...")
    subprocess.run(["python", script_name], check=True)

if __name__ == "__main__":
    print("===== iRBD Classification Pipeline Start =====")

    # Step 1: Preprocess the Questionnaire and Labels
    run_script("pre_process_questionnaire.py")

    # Step 2: Generate Descriptive Statistics Table
    run_script("generate_table_one.py")

    # Step 3: Train Models (Elastic Net, XGBoost, etc.) with Cross-validation
    run_script("main_full_and_cross_val.py")

    # Step 4: Optimize XGBoost Class Weight for Specificity-Sensitivity Tradeoff
    run_script("optmize_xgboost_loss_weight.py")

    # Step 5: Generate ROC Curves and Threshold-Based Evaluation
    run_script("roc_curve_plots_veto_tresh.py")

    print("\n===== All scripts executed successfully. =====")
