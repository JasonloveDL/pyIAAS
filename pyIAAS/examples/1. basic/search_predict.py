import pyIAAS

# set the basic information of the searching process
config_file = 'NASConfig.json'
input_file = 'VT_summer.csv'
target_name = 'RT_Demand'
test_ratio = 0.2  # the proportion of the test dataset in the whole dataset. It can be adjusted by users themself for specific tasks
pyIAAS.set_seed(44)
# start the searching process
pyIAAS.run_search(config_file, input_file, target_name, test_ratio)

# set the basic information of a prediction task
config_file = 'NASConfig.json'
target_name = 'RT_Demand'
output_dir = 'out_dir'
prediction_file = 'ME_autumn_predict.csv'

# perform the predicting task in VT_summer_predict.csv
pyIAAS.run_predict(config_file, input_file, target_name, output_dir, prediction_file)