import csv
import numpy as np
import tensorflow as tf
import os.path
import keras
import os
import pandas as pd
from difflib import SequenceMatcher
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


#################################################### FUNCTION TO LOAD THE DATASET FROM THE FILE ####################################################################################
####################################################################################################################################################################################

def load_dataset(filename) -> pd.DataFrame:
	samples = pd.read_csv(filename,sep=',')
	print(samples)
	return samples

############################################ FUNCTION TO MAKE THE TRAINING AND TESTING DATASET BETTER FOR OPTIMIZATION #####################################################################################
############################################################################################################################################################################################################

def find_k_nearest_by_coordinates(samples_df: pd.DataFrame,k_nearest: int,longitude: float,latitude: float) -> pd.DataFrame:

	geo_coordinates = samples_df[['longitude', 'latitude']]
	target_coord = np.float32([longitude, latitude])
	distances = np.sqrt(np.sum(np.power(target_coord - geo_coordinates, 2), axis=1))
	min_distance_indices = np.argpartition(distances, k_nearest)[1: k_nearest+1]
	return samples_df.iloc[min_distance_indices]



########################################### FUNCTION TO CHANGE THE VALUES OF 0's TO NEARBY SAMPLE VALUES ###################################################################################################
############################################################################################################################################################################################################

def impute_zeros_by_nearby_samples(samples_df: pd.DataFrame,location_column: str,target_column: str,std_threshold: float or None) -> pd.DataFrame:
	
	for area in samples_df[location_column].unique():
		row_ids = samples_df[location_column] == area
		target_values = samples_df.loc[row_ids, target_column]
		if target_values.shape[0] > 1:
			non_zero_ids = target_values > 0
			if non_zero_ids.sum() > 0:
				non_zero_values = target_values[non_zero_ids]
				if std_threshold is not None and np.std(non_zero_values) > std_threshold:
					continue

				zero_ids = np.invert(non_zero_ids)
				target_values[zero_ids] = non_zero_values.mean()
				samples_df.loc[row_ids, target_column] = target_values

	return samples_df


########################################### FUNCTION TO CHANGE THE VALUES OF 0's TO NEARBY SAMPLE VALUES ###################################################################################################
############################################################################################################################################################################################################

def impute_strings_by_nearby_samples(samples_df: pd.DataFrame,location_column: str,target_column: str,strings: str,frequency_threshold: float or None) -> pd.DataFrame:
	for area in samples_df[location_column].unique():
		row_ids = samples_df[location_column] == area
		target_values = samples_df.loc[row_ids, target_column]
		#print("TARGET_VALUES ARE:", target_values)
		if target_values.shape[0] > 1:
			not_strings_ids = ~target_values.isin(strings)
			#print("NON_STRINGS_IDS : ", not_strings_ids)
			if not_strings_ids.sum() > 0:
				value_frequencies = target_values.value_counts() / target_values.shape[0]
				non_strings_values = target_values[not_strings_ids]
				most_frequent_value = non_strings_values.mode()
				
				if frequency_threshold is not None and value_frequencies[most_frequent_value].tolist()[0] < frequency_threshold:
					continue
			
				strings_ids = np.invert(not_strings_ids)
				target_values[strings_ids] = most_frequent_value
				samples_df.loc[row_ids, target_column] = target_values
	return samples_df


########################################### FUNCTION TO CHANGE THE STRING VALUES TO SIMILLAR ONES #######################################################################################################
#########################################################################################################################################################################################################

def impute_strings_by_simillar_strings(samples_df : pd.DataFrame,location_column: str,similarity_ratio: float or None, frequency_threshold: int)-> pd.DataFrame:
	Change_Array = []
	spaces_count_Array = []
	Word_Array = []
	Array_of_uniques = samples_df[location_column].unique().tolist()
	i = 0
	j = 1
	while i < len(Array_of_uniques)-1:
		while j < len(Array_of_uniques):
			s = SequenceMatcher(None,(str(Array_of_uniques[i].lower())),(str(Array_of_uniques[j].lower()))).ratio()
			if (s > similarity_ratio):
				print(f'Ratio is: {s} between {Array_of_uniques[j]} and {Array_of_uniques[i]}')
				Change_Array.append(Array_of_uniques[j])
			j += 1
		if (len(Change_Array)> 0):
			Change_Array.append(Array_of_uniques[i])
			word_counter = {}
			for word in Change_Array:
				spaces_count_Array.append(word.count(" "))
			print("Change_Array is: ", Change_Array)
			for words in samples_df[location_column]:
				if words in Change_Array:
					Word_Array.append(words)
			Word_Array = [words for segments in Word_Array for words in segments.split()]
			for word in Word_Array:
				if word in word_counter:
					word_counter[word] += 1
				else:
					word_counter[word] = 1
			print("Word_Array is: ", Word_Array)
			spaces_count_Array.sort(reverse=True)
			print("spaces_count_Array: ", spaces_count_Array)
			number_of_spaces = Counter(spaces_count_Array).most_common(1)[0][0]
			print("number_of_spaces: ", number_of_spaces)
			if number_of_spaces == 0:
				number_of_spaces = 1
			popular_words = sorted(word_counter, key = word_counter.get, reverse = True)
			common_words = popular_words[:number_of_spaces]
			common_words = " ".join(common_words)
			print("common_words: ",common_words)

			for length in Change_Array:
				samples_df[location_column] = samples_df[location_column].replace({str(length):str(common_words)})
				if (length != Change_Array[len(Change_Array)-1]):
					Array_of_uniques.remove(length)
				print(f'I changed {length} with {common_words}')
			Change_Array.clear()
			Word_Array.clear()
			spaces_count_Array.clear()
		print(f'Length of unique {location_column} is: ', len(Array_of_uniques))
		print(f'I am in {i} loop')
		i += 1
		j = i + 1
	Array_of_uniques.clear()
	for change in samples_df[location_column].unique():
		row_ids = samples_df[location_column] == change
		target_values = samples_df.loc[row_ids, location_column]
		if (target_values.count() <  frequency_threshold):
			samples_df[location_column] = samples_df[location_column].replace({str(change): "Other"})
	 
	return samples_df


################################################### FUNCTION WITH THRESHOLDS TO IMPROVE THE VALUES ##########################################################################################
############################################################################################################################################################################################

def improve_values(samples):
	similarity_ratio = 0.70
	frequency_threshold = 30
	location_string_columns = ['subvillage', 'ward']
	unknown_strings = ['unknown']
	
	###################################### SUBVILLAGE VALUE IMPROVEMENT ###############################################################################
	loc_columns = ['lga','ward','region']
	samples['subvillage'] = samples['subvillage'].fillna('unknown')
	for loc_column in loc_columns:
		samples = impute_strings_by_nearby_samples(samples_df = samples,location_column = loc_column,target_column = 'subvillage',
		                                           strings = unknown_strings,
		                                           frequency_threshold = frequency_threshold)
	
	###################################### WPT_NAME VALUE IMPROVEMENT #################################################################################
	wpt_frequency_threshold = 5
	wpt_strings = ['No Name','none','Not Known','No Wpt']
	for wpt_string in wpt_strings:
		samples['wpt_name'] = samples['wpt_name'].replace({wpt_string:'unknown'})
	for location_string_column in location_string_columns:
		samples = impute_strings_by_nearby_samples(samples_df = samples,location_column = location_string_column,target_column = 'wpt_name',
		                                           strings = unknown_strings,
		                                           frequency_threshold = wpt_frequency_threshold)
	

	##################### AMOUNT_TSH VALUE_IMPROVEMENT ##################################################################################################
	amount_tsh_std_threshold = 50

	samples = impute_zeros_by_nearby_samples(samples_df=samples,location_column='subvillage',
	                                         target_column='amount_tsh',
	                                         std_threshold=amount_tsh_std_threshold)


	###################### POPULATION VALUE_IMPROVEMENT ##################################################################################################
	population_std_threshold = 50
	
	sammples = impute_zeros_by_nearby_samples(samples_df=samples,location_column='subvillage',
	                                          target_column='population',
	                                          std_threshold=population_std_threshold)


	####################### GPS_HEIGHT VALUE IMPROVEMENT ##################################################################################################
	location_columns = ['subvillage', 'ward', 'lga', 'district_code']

	for location_column in location_columns:
		samples = impute_zeros_by_nearby_samples(samples_df=samples,location_column=location_column,
		                                         target_column='gps_height',
		                                         std_threshold=None)
	
	k_neighbors = 25
	gps_height_zero_ids = samples['gps_height'] == 0
	gps_zero_samples = samples[gps_height_zero_ids]
	gps_heights = []

	for _, sample in gps_zero_samples.iterrows():
		longitude = sample['longitude']
		latitude = sample['latitude']
		nearest_samples = find_k_nearest_by_coordinates(samples_df=samples,k_nearest=k_neighbors,longitude=longitude,latitude=latitude)
		non_zero_gps_height_values = nearest_samples.loc[nearest_samples['gps_height'] != 0, 'gps_height']
		gps_heights.append(non_zero_gps_height_values.mean())
	
	samples.loc[samples['gps_height'] == 0, 'gps_height'] = gps_heights
	print(f'gps_height == 0: {(samples["gps_height"] == 0).sum()} After K-NN method')


	###################################### OPERATION TIME- NEW VALUE FOR OUR NEURAL NETWORK ##############################################################
	samples['construction_year'] = samples['construction_year'].replace({0: 2023})
	samples['operation_time'] = pd.DatetimeIndex(samples['date_recorded']).year - samples['construction_year']

	invalid_operation_time_ids = samples['operation_time'] < 0
	samples.loc[invalid_operation_time_ids, 'operation_time'] = -1

	print(f'{invalid_operation_time_ids.sum()} invalid dates were set to -1')
	
	
	#################################### FUNDER VALUE IMPROVEMENT #########################################################################################
	funder_strings  = ['None','0','No','unknown']
	samples['funder'] = samples['funder'].fillna('unknown')
	for location_string_column in location_string_columns:
		samples = impute_strings_by_nearby_samples(samples_df = samples,location_column = location_string_column,target_column = 'funder',
		                                           strings = funder_strings,
		                                           frequency_threshold = frequency_threshold)
	
	samples = impute_strings_by_simillar_strings(samples_df = samples,location_column = 'funder',
	                                   similarity_ratio = similarity_ratio,
	                                   frequency_threshold = frequency_threshold)
	


	#################################### INSTALLER VALUE IMPROVEMENT ######################################################################################
	installer_strings  = ['-','0','No','Not known','Not kno','unknown']
	samples['installer'] = samples['installer'].fillna('unknown')

	for location_string_column in location_string_columns:
		samples = impute_strings_by_nearby_samples(samples_df = samples,location_column = location_string_column,
		                                           target_column = 'installer',
		                                           strings = installer_strings,
		                                           frequency_threshold = frequency_threshold)
	
	
	samples = impute_strings_by_simillar_strings(samples_df = samples,location_column = 'installer',
	                                   similarity_ratio = similarity_ratio,
	                                   frequency_threshold = frequency_threshold)
 


	#################################### SCHEME_NAME VALUE IMPROVEMENT #####################################################################################
	scheme_name_strings  = ['None','no scheme','not known','unknown']
	samples['scheme_name'] = samples['scheme_name'].fillna('unknown')
 
	for location_string_column in location_string_columns:
		samples = impute_strings_by_nearby_samples(samples_df = samples,location_column = location_string_column,
		                                           target_column = 'scheme_name',
		                                           strings = scheme_name_strings,
		                                           frequency_threshold = frequency_threshold)
	
	
	samples = impute_strings_by_simillar_strings(samples_df = samples,location_column = 'scheme_name',
	                                   similarity_ratio = similarity_ratio,
	                                   frequency_threshold = frequency_threshold)
 

	#################################### SCHEME_MANAGEMENT VALUE IMPROVEMENT #################################################################################
	scheme_management_strings  = ['None','unknown']
	samples['scheme_management'] = samples['scheme_management'].fillna('unknown')

	for location_string_column in location_string_columns:
		samples = impute_strings_by_nearby_samples(samples_df = samples,location_column = location_string_column,
		                                           target_column = 'scheme_management',
		                                           strings = scheme_management_strings,
		                                           frequency_threshold = frequency_threshold)



	###################################### PERMIT VALUE IMPROVEMENT ############################################################################################
	samples['permit'] = samples['permit'].fillna('unknown')
	samples['permit'] = samples['permit'].replace({'False': 0,'True': 1, 'unknown': -1})
	#THIS WILL BE PLACED IN MAKEDATASET#


	##################################### PUBLIC_MEETING VALUE IMPROVEMENT ######################################################################################
	samples['public_meeting'] = samples['public_meeting'].fillna('unknown')


	##################################### MANAGEMENT VALUE IMPROVEMENT ##########################################################################################
	for location_string_column in location_string_columns:
		samples = impute_strings_by_nearby_samples(samples_df = samples,location_column = location_string_column,
		                                           target_column = 'management',
		                                           strings = unknown_strings,
		                                           frequency_threshold = frequency_threshold)
	

	##################################### MANAGEMENT_GROUP VALUE IMPROVEMENT #####################################################################################
	for location_string_column in location_string_columns:
		samples = impute_strings_by_nearby_samples(samples_df = samples,location_column = location_string_column,
		                                           target_column = 'management_group',
		                                           strings = unknown_strings,
		                                           frequency_threshold = frequency_threshold)


	##################################### PAYMENT VALUE IMPROVEMENT ###############################################################################################
	for location_string_column in location_string_columns:
		samples = impute_strings_by_nearby_samples(samples_df = samples,location_column = location_string_column,
		                                           target_column = 'payment',
		                                           strings = unknown_strings,
		                                           frequency_threshold = frequency_threshold)


	##################################### PAYMENT_TYPE VALUE IMPROVEMENT ############################################################################################
	for location_string_column in location_string_columns:
		samples = impute_strings_by_nearby_samples(samples_df = samples,location_column = location_string_column,
		                                           target_column = 'payment_type',
		                                           strings = unknown_strings,
		                                           frequency_threshold = frequency_threshold)
	

	##################################### PAYMENT VALUE IMPROVEMENT #################################################################################################
	for location_string_column in location_string_columns:
		samples = impute_strings_by_nearby_samples(samples_df = samples,location_column = location_string_column,
		                                           target_column = 'payment',
		                                           strings = unknown_strings,
		                                           frequency_threshold = frequency_threshold)	
	

	##################################### WATER_QUALITY VALUE IMPROVEMENT ###########################################################################################
	for location_string_column in location_string_columns:
		samples = impute_strings_by_nearby_samples(samples_df = samples,location_column = location_string_column,
		                                           target_column = 'water_quality',
		                                           strings = unknown_strings,
		                                           frequency_threshold = frequency_threshold)
	

	##################################### QUALITY_GROUP VALUE IMPROVEMENT #############################################################################################
	for location_string_column in location_string_columns:
		samples = impute_strings_by_nearby_samples(samples_df = samples,location_column = location_string_column,
		                                           target_column = 'quality_group',
		                                           strings = unknown_strings,
		                                           frequency_threshold = frequency_threshold)
	

	##################################### SOURCE VALUE IMPROVEMENT #################################################################################################
	for location_string_column in location_string_columns:
		samples = impute_strings_by_nearby_samples(samples_df = samples,location_column = location_string_column,
		                                           target_column = 'source',
		                                           strings = unknown_strings,
		                                           frequency_threshold = frequency_threshold)
	

	##################################### SOURCE_TYPE VALUE IMPROVEMENT #############################################################################################
	for location_string_column in location_string_columns:
		samples = impute_strings_by_nearby_samples(samples_df = samples,location_column = location_string_column,
		                                           target_column = 'source_type',
		                                           strings = unknown_strings,
		                                           frequency_threshold = frequency_threshold)
	

	##################################### SOURCE_CLASS VALUE IMPROVEMENT ##############################################################################################
	for location_string_column in location_string_columns:
		samples = impute_strings_by_nearby_samples(samples_df = samples,location_column = location_string_column,
		                                           target_column = 'source_class',
		                                           strings = unknown_strings,
		                                           frequency_threshold = frequency_threshold)
	
	return samples


#################################################### FUNCTION TO MAKE STRINGS TO CLASSES AS NUMBERS FOR INPUTS IN OUR NEURAL NETWORK #######################################################
##############################################################################################################################################################################################

def make_dataset(samples):
	columns = ['funder','installer','wpt_name','basin','subvillage','region','lga','ward','public_meeting','recorded_by','scheme_management',
	           'scheme_name','permit','extraction_type','extraction_type_group','extraction_type_class','management','management_group',
						      'payment','payment_type','water_quality','quality_group','quantity','quantity_group','source','source_type','source_class',
									   'waterpoint_type','waterpoint_type_group']

	sample_columns = ['amount_tsh','funder','gps_height','installer','longitude','latitude','wpt_name','num_private','basin','subvillage',
	                  'region','region_code','district_code','lga','ward','population','public_meeting','recorded_by',
										         'scheme_management','scheme_name','permit','extraction_type','extraction_type_group','extraction_type_class',
														     'management','management_group','payment','payment_type','water_quality','quality_group','quantity',
																   'quantity_group','source','source_type','source_class',
																	  'waterpoint_type','waterpoint_type_group','operation_time']

	for areas in columns:
		if (areas == 'permit' or areas == 'public_meeting'):
			samples[areas] = samples[areas].replace({'False': 0,'True': 1, 'unknown': -1})
		else:
			Array_of_uniques = sorted(samples[areas].unique())
			for values in Array_of_uniques:
				samples[areas] = samples[areas].replace({values:Array_of_uniques.index(values)})
	
	samples = samples[sample_columns]
	print(samples.shape)
	return samples


###################################### FUNCTION TO NORMALIZE OUR DATA FOR OUR NEURAL NETWORK ########################################################
#####################################################################################################################################################

def normalize_data(samples):
	scaler = StandardScaler()
	normalized_data = scaler.fit_transform(samples)
	print(normalized_data)
 
	return normalized_data


##################################### FUNCTION TO LOAD THE PREDICTIONS AND THE EXPECTED OUTPUTS OF OUR NEURAL NETWORK ########################################################
##############################################################################################################################################################################

def make_predictions(targets):
	targets['status_group'] = targets['status_group'].replace({'non functional': -1,'functional needs repair': 0,'functional': 1})
	targets = targets['status_group']
	Predictions =[]
	for i in range(len(targets)):
		if targets[i] == -1:
			Predictions.append([0.0,0.0,1.0])
		elif targets[i] == 0:
			Predictions.append([0.0,1.0,0.0])
		else:
			Predictions.append([1.0,0.0,0.0])
	 
	print(Predictions)

	return Predictions


########################################### FUNCTION TO TRAIN OUR NEURAL NETWORK ####################################################################################
#####################################################################################################################################################################

def neural_network_train(Dataset,Predictions):
	Predictions = np.array(Predictions)
	#X_train, X_test, Y_train, Y_test = train_test_split(Dataset, Predictions, test_size=0.1)
	X_train = Dataset
	Y_train = Predictions

	inputs = tf.keras.Input(shape=(38,))
	x = tf.keras.layers.Dense(256, activation=tf.nn.relu)(inputs)
	y = tf.keras.layers.Dense(256, activation=tf.nn.relu)(x) 
	outputs = tf.keras.layers.Dense(3, activation=tf.nn.softmax)(y)
	model = tf.keras.Model(inputs=inputs, outputs=outputs)
 
	callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
	model.summary()
	model.compile(loss='categorical_crossentropy', optimizer= 'adam',metrics=['accuracy'])
	model.fit(X_train, Y_train, epochs=200,callbacks=[callback])
 
	train_loss, train_accuracy = model.evaluate(X_train, Y_train,
                                   verbose=0)
	print("Train loss:", train_loss)
 
	print('Train accuracy is: ',train_accuracy * 100,"%\n")

	return model

######################################### FUNCTION TO SELF TEST OUR NEURAL NETWORK #################################################################
####################################################################################################################################################

def neural_network_test(Dataset,model):

	X_test = Dataset
	Y_test = model.predict(X_test)

	return (Y_test)
	


########################################### FUNCTION TO SELF TEST OUR NEURAL NETWORK ####################################################################
#########################################################################################################################################################

def write_to_file(Array1,Array2):
	i = 0
	Write_Array = []
	write = ""
	f = open("SubmissionFormat.csv",'w')
	
	writer = csv.writer(f)
	Write_Array.append(['id','status_group'])
	writer.writerow(Write_Array[i])
	
	while (i < len(Array1)):
		
		if (Array1[i][1] > Array1[i][0]):
			if (Array1[i][1] > Array1[i][2]):
				write = "functional needs repair"
			else:
				write = "non functional"
				
		elif (Array1[i][2] > Array1[i][0]):
			write = "non functional"
			
		else:
			write = "functional"
		
		Write_Array.append([Array2[i],write])
		writer.writerow(Write_Array[i+1])
		i += 1
		

############################################### MAIN FUNCTION TO RUN THE PROGRAM #######################################################################################
########################################################################################################################################################################

def main():
	if(os.path.exists('train.csv') == False):
		print("HELLO I AM IN! NO train.csv file exist")
		samples = load_dataset('training_values.csv')
		samples = improve_values(samples)
		samples.to_csv('train.csv', index=False)

	if (os.path.exists('train_number.csv') == False):
		print("HELLO I AM IN! NO train_number.csv file exist")
		samples = load_dataset('train.csv')
		samples = make_dataset(samples)
		samples.to_csv('train_number.csv', index=False)
	
	if (os.path.exists('testing.csv') == False):
		print("HELLO I AM IN! NO testing.csv file exist")
		tests = load_dataset('test.csv')
		tests = improve_values(tests)
		tests.to_csv('testing.csv', index=False)
	
	if(os.path.exists('testing_numbers.csv') == False):
		print("HELLO I AM IN! NO testing_numbers.csv file exist")
		tests = load_dataset('testing.csv')
		tests = make_dataset(tests)
		tests.to_csv('testing_numbers.csv', index=False)
		
	train_samples = load_dataset('train_number.csv')
	train_dataset = normalize_data(train_samples)

	targets = load_dataset('training_labels.csv')
	train_predictions = make_predictions(targets)
	
	test_samples = load_dataset('testing_numbers.csv')
	test_dataset = normalize_data(test_samples)
 
	model = neural_network_train(train_dataset,train_predictions)
	
	test_predictions = neural_network_test(test_dataset,model)

	tests = load_dataset('testing.csv')
	id_Array = tests['id']
	write_to_file(test_predictions,id_Array)


main()






'''
	############################################ OPTUNA RUN ###############################################
	#######################################################################################################

	study = optuna.create_study(direction="maximize")
	study.optimize(objective, n_trials=25)
	pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
	complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]
	print("Study statistics: ")
	print("  Number of finished trials: ", len(study.trials))
	print("  Number of pruned trials: ", len(pruned_trials))
	print("  Number of complete trials: ", len(complete_trials))

	print("Best trial:")
	trial = study.best_trial

	print("  Value: ", trial.value)

	print("  Params: ")
	for key, value in trial.params.items():
		print("    {}: {}".format(key, value))

	#######################################################################################################
	#######################################################################################################
	











################################################## OPTUNA TO FIND THE BEST DEEP NEURAL NETWORK LAYERS ####################################################################################################
###################################################################################################################################################################################################	

def create_classifier(trial):
	n_layers = trial.suggest_int("n_layers", 1, 3)
	learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1000)
	optimizer = trial.suggest_categorical('optimizer', ['SGD', 'Adam'])
	hidden_units = []

	for i in range(n_layers):
		n_units = trial.suggest_int("n_units_l{}".format(i), 1, 256)
		hidden_units.append(n_units)
	
############################################################################################################################################################################################
############################################################################################################################################################################################

################################################## OPTUNA TO FIND THE BEST DEEP NEURAL NETWORK OPTIMIZER ####################################################################################################
###################################################################################################################################################################################################	

def objective(trial):
	classifier = create_classifier(trial)

	optuna_pruning_hook = optuna.integration.TensorFlowPruningHook(
			  trial=trial,
	      estimator=classifier,
        metric="accuracy",
        run_every_steps=PRUNING_INTERVAL_STEPS,
    )

	train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=TRAIN_STEPS, hooks=[optuna_pruning_hook])

	eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, start_delay_secs=0, throttle_secs=0)

	eval_results, _ = tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

	return float(eval_results["accuracy"])

'''
	
