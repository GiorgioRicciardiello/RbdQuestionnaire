"""
Subjects that should had not been considered in the study were first removed from the pre-processed version.

Here we re-do the computation of bad nights removed for the patients that should had been considered from the start
"""
import pandas as pd
from config.config import config, config_actigraphy
from tabulate import tabulate

def format_shas_id(x):
    if isinstance(x, str) and x.startswith("SHAS"):
        x_new = x.split('_')[0]
        x_new = x_new.replace('SHAS', 'SHAS-')
        return x_new
    return x  # leave other IDs unchanged


def summarize_nights(df: pd.DataFrame,
                     id_col: str = "subject_id",
                     label_col: str = "label") -> pd.DataFrame:
    """
    Summarize number of nights per subject and compute group-level stats.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (raw or cleaned).
    id_col : str
        Column with subject identifiers (default: "subject_id").
    label_col : str
        Column with labels/groups (default: "label").

    Returns
    -------
    pd.DataFrame
        Summary per group (label) with subject count, mean, std of nights.
    """

    # Nights per subject
    nights_per_subject = df.groupby(id_col).size().reset_index(name="n_nights").copy()
    # Each subject’s label (consistent within subject_id)
    labels = df.groupby(id_col)[label_col].first().reset_index()
    subject_summary = nights_per_subject.merge(labels, on=id_col)

    result = (
        subject_summary
        .groupby("label")["n_nights"]
        .agg(["count", "mean", "std", "max", "min"])
        .reset_index()
    )
    result["mean"] = result["mean"].round(1)
    result["std"] = result["std"].round(1)

    print("\nNights per subject summary:")
    print(tabulate(result, headers="keys", tablefmt="psql", showindex=False))

    # Convert counts Series → DataFrame
    counts = df[label_col].value_counts().rename("total_nights").reset_index()
    counts.columns = [label_col, "total_nights"]

    print("\nTotal nights per group:")
    print(tabulate(counts, headers="keys", tablefmt="psql", showindex=False))

    return result
if __name__ == "__main__":

    df = pd.read_csv(r"C:\Users\giorg\OneDrive\Projects\MtSinai\During\RbdQuestionnaire\data\raw\nightly_features_raw.csv")


    # %% data and path
    df_quest = pd.read_csv(config.get('data_path').get('pp_questionnaire'))
    df_actig_clean = pd.read_csv(config.get('data_path').get('pp_actig'))

    df_actig_raw = pd.read_csv(config.get('data_path').get('raw').joinpath(r'nightly_features_raw.csv'))
    # df_actig_raw = pd.read_csv(r"C:\Users\giorg\Downloads\nightly_features_raw.csv")
    df_actig_raw["ID"] = df_actig_raw["ID"].apply(format_shas_id)

    result_dir = config.get('results_path').get('results').joinpath(f'filter_bad_nights')
    result_dir.mkdir(parents=True, exist_ok=True)
    #%% Sanity check the was have all the pp subjects in raw (VASC edit of alias _2024 to ''), just naming type in one source
    print(df_actig_raw["ID"].nunique())
    print(df_actig_clean["subject_id"].nunique())

    # which are the ones missing in raw
    raw_ids = set(df_actig_raw["ID"].unique())
    clean_ids = set(df_actig_clean["subject_id"].unique())
    missing_in_raw = clean_ids - raw_ids
    print(f"Missing in Raw: {missing_in_raw}")

    # %% remove the subjects that were added and do not have the participant requierements
    df_actig_raw = df_actig_raw[df_actig_raw['ID'].isin(list(df_actig_clean["subject_id"].unique()))]
    print(df_actig_raw["ID"].nunique())
    df_post_check = df_actig_raw.copy()
    # %%
    # %% Filter the nights
    # ---- QC Flags ----
    # Counts before filtering
    nights_per_subject = df_actig_raw.groupby("subject_id").size().reset_index(name="n_nights")
    # Add each subject's label (same for all their rows)
    # (take the first value, since it's consistent per subject)
    labels = df_actig_raw.groupby("subject_id")["label"].first().reset_index()
    # Merge counts with labels
    subject_summary = nights_per_subject.merge(labels, on="subject_id")
    # Compute mean and std per group
    result = (
        subject_summary
        .groupby("label")["n_nights"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    print(result)


    total_nights = len(df_actig_raw)
    df_actig_raw_clean = df_actig_raw.copy()
    # Apply flags
    # --- Rule 1: discard TST <3 h or >12 h ------------------------------
    df_actig_raw_clean["flag_bad_TST"] = (df_actig_raw_clean["TST"] < 3) | (df_actig_raw_clean["TST"] > 12)
    # --- Rule 2: discard low temperature (<27 °C) -----------------------
    df_actig_raw_clean["flag_low_temp"] = df_actig_raw_clean["T_avg"] < 27
    # --- Rule 3: discard non-wear >2 h between 0-6 AM -------------------
    df_actig_raw_clean["flag_nonwear_night"] = df_actig_raw_clean["nw_night"] > 4

    df_actig_raw_clean["good_night"] = ~(df_actig_raw_clean[["flag_bad_TST", "flag_low_temp", "flag_nonwear_night"]].any(axis=1))

    # Count losses per stage
    loss_bad_tst = df_actig_raw_clean["flag_bad_TST"].sum()
    loss_low_temp = df_actig_raw_clean["flag_low_temp"].sum()
    loss_nonwear = df_actig_raw_clean["flag_nonwear_night"].sum()
    loss_any = (~df_actig_raw_clean["good_night"]).sum()
    kept_final = df_actig_raw_clean["good_night"].sum()

    print(f"Total nights: {total_nights}")
    print(f"Lost at bad_TST: {loss_bad_tst}")
    print(f"Lost at low_temp: {loss_low_temp}")
    print(f"Lost at nonwear_night: {loss_nonwear}")
    print(f"Lost by any rule: {loss_any}")
    print(f"Kept after all QC: {kept_final}")
    # keep only the good nights
    df_actig_raw_clean = df_actig_raw_clean[df_actig_raw_clean["good_night"]]

    # %%  Count nights per subject
    df_actig_raw_clean.rename({'ID': 'subject_id' }, inplace=True)
    df_actig_raw.rename({'ID': 'subject_id' }, inplace=True)


    print(f'Night summary raw:')
    df_nights = summarize_nights(df=df_actig_raw, id_col='subject_id', label_col='label')

    print(f'Night summary clean:')
    summarize_nights(df=df_actig_raw_clean, id_col='subject_id', label_col='label')

    # %% Sanity check
    # count how many records per subjects
    df_counts_raw = (
        df_actig_raw["subject_id"]
        .value_counts()
        .reset_index(name="n_nights")  # count column
        .rename(columns={"index": "subject_id"})  # rename the subject column
        .sort_values(by="n_nights", ascending=False)
    )
    df_counts_raw.to_csv(result_dir.joinpath('counts_night_raw.csv'), index=False)
    # Subjects with more than 30 nights
    # Nights per subject
    nights_per_subject = df_actig_raw.groupby("subject_id").size().reset_index(name="n_nights")
    subjects_over_30 = nights_per_subject[nights_per_subject["n_nights"] > 30]
    subjects_over_30.sort_values(by='n_nights', ascending=False, inplace=True)
    subjects_over_30.to_csv(result_dir.joinpath(f'subjects_over_30_raw.csv'), index=False)
    print(tabulate(subjects_over_30, headers="keys", tablefmt="psql", showindex=False))
    # check duplicates rows between these subjects
    df_over_30 = df_actig_raw[df_actig_raw["subject_id"].isin(subjects_over_30.subject_id)]
    # Check duplicates per subject
    duplicates_per_subject = (
        df_over_30
        .groupby("subject_id")
        .apply(lambda x: x.duplicated().sum())
        .reset_index(name="n_duplicates")
    )
    print(tabulate(duplicates_per_subject, headers="keys", tablefmt="psql", showindex=False))
    if duplicates_per_subject.n_duplicates.max() > 0:
        # Show actual duplicate rows for subjects > 30 nights
        duplicates = (
            df_over_30[df_over_30.duplicated(keep=False)]
            .sort_values(["subject_id"])
        )
        print(tabulate(duplicates.head(20), headers="keys", tablefmt="psql", showindex=False))


    # In the clean version
    # count how many records per subjects
    df_counts_raw_clean = (
        df_actig_raw_clean["subject_id"]
        .value_counts()
        .reset_index(name="n_nights")  # count column
        .rename(columns={"index": "subject_id"})  # rename the subject column
        .sort_values(by="n_nights", ascending=False)
    )
    df_counts_raw_clean.to_csv(result_dir.joinpath('counts_night_clean.csv'), index=False)

    nights_per_subject = df_actig_raw_clean.groupby("subject_id").size().reset_index(name="n_nights")
    subjects_over_30 = nights_per_subject[nights_per_subject["n_nights"] > 30]
    subjects_over_30.sort_values(by='n_nights', ascending=False, inplace=True)
    subjects_over_30.to_csv(result_dir.joinpath(f'subjects_over_30_clean.csv'), index=False)


    # %% How many shas and stanford
    n_nights_shas = df_actig_clean['subject_id'].str.startswith('SHAS').sum()  # 770
    n_nights_stanford = df_actig_clean['subject_id'].str.startswith('AXRB').sum()  # 1859
    n_nighs_vas =     df_actig_clean.shape[0] - n_nights_stanford - n_nights_shas  # 317
    assert df_actig_clean.shape[0] == n_nights_stanford + n_nighs_vas + n_nights_shas


    # %% Checking how many if we join with the ADRC cohort
    df_adrc = pd.read_csv(config_actigraphy.get('raw_actigraphy_adrc_wlabels'))
    df_post_check.drop(columns=['subject_id'], inplace=True)
    df_post_check.rename(columns={'ID': 'subject_id'}, inplace=True)

    df_all = pd.concat([df_post_check, df_adrc], axis=0, ignore_index=True)

    nights_per_subject = df_all.groupby("subject_id").size().reset_index(name="n_nights")
    labels = df_all.groupby("subject_id")["label"].first().reset_index()
    # Merge counts with labels
    subject_summary = nights_per_subject.merge(labels, on="subject_id")
    # Compute mean and std per group
    result = (
        subject_summary
        .groupby("label")["n_nights"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    print(result)


    total_nights = len(df_all)
    # Apply flags
    # --- Rule 1: discard TST <3 h or >12 h ------------------------------
    df_all["flag_bad_TST"] = (df_all["TST"] < 3) | (df_all["TST"] > 12)
    # --- Rule 2: discard low temperature (<27 °C) -----------------------
    df_all["flag_low_temp"] = df_all["T_avg"] < 27
    # --- Rule 3: discard non-wear >2 h between 0-6 AM -------------------
    df_all["flag_nonwear_night"] = df_all["nw_night"] > 4

    df_all["good_night"] = ~(df_all[["flag_bad_TST", "flag_low_temp", "flag_nonwear_night"]].any(axis=1))

    # Count losses per stage
    loss_bad_tst = df_all["flag_bad_TST"].sum()
    loss_low_temp = df_all["flag_low_temp"].sum()
    loss_nonwear = df_all["flag_nonwear_night"].sum()
    loss_any = (~df_all["good_night"]).sum()
    kept_final = df_all["good_night"].sum()

    print(f"Total nights: {total_nights}")
    print(f"Lost at bad_TST: {loss_bad_tst}")
    print(f"Lost at low_temp: {loss_low_temp}")
    print(f"Lost at nonwear_night: {loss_nonwear}")
    print(f"Lost by any rule: {loss_any}")
    print(f"Kept after all QC: {kept_final}")
    # keep only the good nights
    df_all_final = df_all[df_all["good_night"]]

    # count nights
    nights_per_subject = df_all_final.groupby("subject_id").size().reset_index(name="n_nights")
    labels = df_all_final.groupby("subject_id")["label"].first().reset_index()
    # Merge counts with labels
    subject_summary = nights_per_subject.merge(labels, on="subject_id")

    result = (
        subject_summary
        .groupby("label")["n_nights"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    print(result)










