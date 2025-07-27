"""
Author: Giorgio Ricciardiello
        giocrm@stanford.edu
configurations parameters for the paths
"""
import pathlib

# import shutil

# Define root path
root_path = pathlib.Path(__file__).resolve().parents[1]
# Define raw data path
data_path = root_path.joinpath('data')
# data paths
data_raw_path = data_path.joinpath('raw')
data_pp_path = data_path.joinpath('pp')
# results path
data_res = root_path.joinpath('results')




# Construct the config dictionary with nested templates
config = {
    'root_path': root_path,
    'data_path': {
        'data': data_path,
        # raw files
        'raw': data_raw_path,
        'raw_questionnaire': data_raw_path.joinpath('shas_clinic_stanford_4Qs.xlsx'),
        'raw_act_features': data_raw_path.joinpath('actig_extracted_features_multiple_nights.csv'),
        # pre-process files
        'pp': data_pp_path,
        'pp_questionnaire': data_pp_path.joinpath('pp_questionnaire.csv'),
        'pp_act_features': data_pp_path.joinpath('pp_actig_extracted_features_multiple_nights.csv'),
    },
    # results path
    'results_path': {
        'results': data_res,
        'full_cross_val': data_res.joinpath('full_cross_val'),
    }
}





