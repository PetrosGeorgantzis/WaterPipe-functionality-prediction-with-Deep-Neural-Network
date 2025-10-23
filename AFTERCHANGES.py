from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import csv
import numpy as np
import random

Id_type = []            ############ ARRAY TO KEEP INFO FOR THE ID OF EACH WATER PUMP #############################

Dataset = []            ############ ARRAY TO KEEP INFO FOR THE DATASET WE WILL USE FOR OUR DEEP NEURAL NETWORK #####################################
Predictions = []        ############ ARRAY TO KEEP INFO FOR THE CLASS OF EACH DATASET ###########################################


Dataset_train = []      ############ ARRAY TO KEEP INFO FOR THE DATASET WE WILL USE FOR TRAINING OUR DEEP NEURAL NETWORK #####################
Dataset_test = []       ############ ARRAY TO KEEP INFO FOR THE DATASET WE WILL USE FOR TESTING OUR DEEP NEURAL NETWORK #####################

Predictions_train = []  ############ ARRAY TO KEEP INFO FOR THE RECOMMENDED OUTPUTS WE WANT FROM TRAINING IN OUR DEEP NEURAL NETWORK #####################
Predictions_test = []   ############ ARRAY TO KEEP INFO FOR THE RECOMMENDED OUTPUTS WE WANT FROM TESTING IN OUR DEEP NEURAL NETWORK #####################

funder_temp = ""	########################## VARIABLE FOR FUNDER INFO ############################################################
funder_Array = []      ############################### ARRAY FOR CLASSES OF FUNDERS ############################################

installer_temp = ""    ############################### VARIABLE FOR INSTALLER INFO #############################################
installer_Array =[]    ############################# ARRAY FOR CLASSES FOR INSTALLERS #############################################

basin_temp = ""        ############################### VARIABLE FOR BASIN INFO #############################################
basin_Array = []       ############################# ARRAY FOR CLASSES FOR BASINS #############################################

region_temp = ""       ############################### VARIABLE FOR REGION INFO #############################################
region_Array = []      ############################# ARRAY FOR CLASSES FOR REGIONS #############################################

lga_temp = ""          ############################### VARIABLE FOR LGA INFO #############################################
lga_Array = []         ############################# ARRAY FOR CLASSES FOR LGAS #############################################

ward_temp = ""         ############################### VARIABLE FOR WARD INFO #############################################
ward_Array = []        ############################# ARRAY FOR CLASSES FOR WARDS #############################################

scheme_management_temp = ""    ########################## VARIABLE FOR SCHEME_MANAGEMENT INFO ##################################
scheme_management_Array = []   ########################## ARRAY FOR CLASSES FOR SCHEME_MANAGEMENTS ##############################

scheme_name_temp = ""          ########################## VARIABLE FOR SCHEME_NAME INFO ##################################
scheme_name_Array= []   ########################## ARRAY FOR CLASSES FOR SCHEME_NAMES ##############################

extraction_type_temp = ""    ########################## VARIABLE FOR EXTRACTION_TYPE INFO ##################################
extraction_type_Array = []   ########################## ARRAY FOR CLASSES FOR EXTRACTION_TYPES ##############################

extraction_type_group_temp = ""    ########################## VARIABLE FOR EXTRACTION_TYPE_GROUP INFO ##################################
extraction_type_group_Array = []   ########################## ARRAY FOR CLASSES FOR EXTRACTION_TYPE_GROUPS ##############################

extraction_type_class_temp = ""    ########################## VARIABLE FOR EXTRACTION_TYPE_CLASS INFO ##################################
extraction_type_class_Array = []   ########################## ARRAY FOR CLASSES FOR EXTRACTION_TYPE_CLASSES ##############################

management_temp = ""         ############################### VARIABLE FOR MANAGEMENT INFO #############################################
management_Array = []        ############################# ARRAY FOR CLASSES FOR MANAGEMENTS #############################################

management_group_temp = ""         ############################### VARIABLE FOR MANAGEMENT_GROUP INFO #############################################
management_group_Array = []        ############################# ARRAY FOR CLASSES FOR MANAGEMENT_GROUPS #############################################

payment_temp = ""         ############################### VARIABLE FOR PAYMENT INFO #############################################
payment_Array = []        ############################# ARRAY FOR CLASSES FOR PAYMENTS #############################################

water_quality_temp = ""         ############################### VARIABLE FOR WATER_QUALITY INFO #############################################
water_quality_Array = []        ############################# ARRAY FOR CLASSES FOR WATER_QUALITY #############################################

water_quality_group_temp = ""         ############################### VARIABLE FOR WATER_QUALITY_GROUP INFO #############################################
water_quality_group_Array = []        ############################# ARRAY FOR CLASSES FOR WATER_QUALITY_GROUP #############################################

quantity_temp = ""         ############################### VARIABLE FOR QUANTITY INFO #############################################
quantity_Array = []        ############################# ARRAY FOR CLASSES FOR QUANTITY #############################################

source_temp = ""         ############################### VARIABLE FOR SOURCE INFO #############################################
source_Array = []        ############################# ARRAY FOR CLASSES FOR SOURCES #############################################

source_type_temp = ""         ############################### VARIABLE FOR SOURCE_TYPE INFO #############################################
source_type_Array = []        ############################# ARRAY FOR CLASSES FOR SOURCE_TYPES #############################################

source_class_temp = ""         ############################### VARIABLE FOR SOURCE_CLASS INFO #############################################
source_class_Array = []        ############################# ARRAY FOR CLASSES FOR SOURCE_CLASSES #############################################

waterpoint_type_temp = ""         ############################### VARIABLE FOR WATERPOINT_TYPE INFO #############################################
waterpoint_type_Array = []        ############################# ARRAY FOR CLASSES FOR WATERPOINT_TYPES #############################################

waterpoint_type_group_temp = ""         ############################### VARIABLE FOR WATERPOINT_TYPE_GROUP INFO #############################################
waterpoint_type_group_Array = []        ############################# ARRAY FOR CLASSES FOR WATERPOINT_TYPE_GROUPS #############################################



#################################################### FUNCTION TO LOAD THE DATASET FROM THE FILE ####################################################################################
####################################################################################################################################################################################

def LoadDataset(filename):
	with open(filename) as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			if (row[0].isdigit()):
				
				Id_type.append(row[0])		#################### KEEP INFO FOR THE ID OF THE WATER PUMP INTO THE ID ARRAY #################################
				
				############################################## KEEP INFO FOR THE DATASET WE WILL USE AS INPUT IN OUR NEURAL NETWORK #############################
				Dataset.append([float(row[1]),str(row[3]),float(row[4]),str(row[5]),float(row[6]),float(row[7]),float(row[9]),str(row[10]),str(row[12]),
				float(row[13]),float(row[14]),str(row[15]),str(row[16]),float(row[17]),str(row[18]),str(row[20]),str(row[22]),float(row[23]),
				str(row[24]),str(row[27]),str(row[28]),str(row[29]),str(row[31]),str(row[33]),str(row[35]),str(row[37]),str(row[38]),
				str(row[21])])			
				#################################################################################################################################################################
				#str(row[39]),str(row[36]),str(row[32]),str(row[25]),str(row[26]),
				#################################################### DIVINE INTO CLASSES THE FUNDER INFO ########################################################################
				funder_temp = str(row[3])
				if (funder_temp == ""):
					funder_temp = "None"
				if (funder_temp not in funder_Array):
					funder_Array.append(funder_temp)
				##################################################################################################################################################################
				
				#################################################### DIVINE INTO CLASSES THE INSTALLER INFO ########################################################################
				installer_temp = str(row[5])
				if (installer_temp == ""):
					installer_temp = "None"
				if (installer_temp not in installer_Array):
		        		installer_Array.append(installer_temp)
				##################################################################################################################################################################
				
				
		       	#################################################### DIVINE INTO CLASSES THE BASIN INFO ########################################################################
				basin_temp = str(row[10])
				if (basin_temp not in basin_Array):
					basin_Array.append(basin_temp)
				##################################################################################################################################################################
				
				#################################################### DIVINE INTO CLASSES THE REGION INFO ########################################################################
				region_temp = str(row[12])
				if (region_temp not in region_Array):
					region_Array.append(region_temp)
				##################################################################################################################################################################
				
				#################################################### DIVINE INTO CLASSES THE REGION INFO ########################################################################
				lga_temp = str(row[15])
				if (lga_temp not in lga_Array):
					lga_Array.append(lga_temp)
				##################################################################################################################################################################
				
				#################################################### DIVINE INTO CLASSES THE WARD INFO ########################################################################
				ward_temp = str(row[16])
				if (ward_temp not in ward_Array):
					ward_Array.append(ward_temp)
				##################################################################################################################################################################
				
				#################################################### DIVINE INTO CLASSES THE SCHEME_MANAGEMENT INFO ###############################################################
				scheme_management_temp = str(row[20])
				if (scheme_management_temp == ""):
					scheme_management_temp = "None"
				if (scheme_management_temp not in scheme_management_Array):
					scheme_management_Array.append(scheme_management_temp)
				##################################################################################################################################################################
				
				#################################################### DIVINE INTO CLASSES THE EXTRACTION_TYPE INFO #################################################################
				extraction_type_temp = str(row[24])
				if (extraction_type_temp not in extraction_type_Array):
					extraction_type_Array.append(extraction_type_temp)
				##################################################################################################################################################################
			
				#################################################### DIVINE INTO CLASSES THE EXTRACTION_TYPE_GROUP INFO #############################################################
				extraction_type_group_temp = str(row[25])
				if (extraction_type_group_temp not in extraction_type_group_Array):
					extraction_type_group_Array.append(extraction_type_group_temp)
				##################################################################################################################################################################
			
				#################################################### DIVINE INTO CLASSES THE EXTRACTION_TYPE_CLASS INFO #############################################################
				extraction_type_class_temp = str(row[26])
				if (extraction_type_class_temp not in extraction_type_class_Array):
					extraction_type_class_Array.append(extraction_type_class_temp)
				##################################################################################################################################################################
				
				#################################################### DIVINE INTO CLASSES THE MANAGEMENT INFO ########################################################################
				management_temp = str(row[27])
				if (management_temp not in management_Array):
					management_Array.append(management_temp)
				##################################################################################################################################################################
				
				#################################################### DIVINE INTO CLASSES THE MANAGEMENT_GROUP INFO ################################################################
				management_group_temp = str(row[28])
				if (management_group_temp not in management_group_Array):
					management_group_Array.append(management_group_temp)
				##################################################################################################################################################################
				
				#################################################### DIVINE INTO CLASSES THE PAYMENT INFO ########################################################################
				payment_temp = str(row[29])
				if (payment_temp not in payment_Array):
					payment_Array.append(payment_temp)
				##################################################################################################################################################################
				
				#################################################### DIVINE INTO CLASSES THE WATER_QUALITY INFO ####################################################################
				water_quality_temp = str(row[31])
				if (water_quality_temp not in water_quality_Array):
					water_quality_Array.append(water_quality_temp)
				##################################################################################################################################################################
				
				#################################################### DIVINE INTO CLASSES THE WATER_QUALITY_GROUP INFO #############################################################
				water_quality_group_temp = str(row[32])
				if (water_quality_group_temp not in water_quality_group_Array):
					water_quality_group_Array.append(water_quality_group_temp)
				##################################################################################################################################################################
				
				#################################################### DIVINE INTO CLASSES THE QUANTITY INFO ########################################################################
				quantity_temp = str(row[33])
				if (quantity_temp not in quantity_Array):
					quantity_Array.append(quantity_temp)
				##################################################################################################################################################################
				
				#################################################### DIVINE INTO CLASSES THE SOURCE INFO ########################################################################
				source_temp = str(row[35])
				if (source_temp not in source_Array):
					source_Array.append(source_temp)
				##################################################################################################################################################################
				
				#################################################### DIVINE INTO CLASSES THE SOURCE_TYPE INFO #######################################################################
				source_type_temp = str(row[36])
				if (source_type_temp not in source_type_Array):
					source_type_Array.append(source_type_temp)
				##################################################################################################################################################################
				
				#################################################### DIVINE INTO CLASSES THE SOURCE_CLASS INFO ######################################################################
				source_class_temp = str(row[37])
				if (source_class_temp not in source_class_Array):
					source_class_Array.append(source_class_temp)
				##################################################################################################################################################################
				
				#################################################### DIVINE INTO CLASSES THE WATERPOINT_TYPE INFO #################################################################
				waterpoint_type_temp = str(row[38])
				if (waterpoint_type_temp not in waterpoint_type_Array):
					waterpoint_type_Array.append(waterpoint_type_temp)
				##################################################################################################################################################################
				
				#################################################### DIVINE INTO CLASSES THE WATERPOINT_TYPE_GROUP INFO ############################################################
				waterpoint_type_group_temp = str(row[39])
				if (waterpoint_type_group_temp not in waterpoint_type_group_Array):
					waterpoint_type_group_Array.append(waterpoint_type_group_temp)
				##################################################################################################################################################################
				
				scheme_name_temp = str(row[21])
				if (scheme_name_temp not in scheme_name_Array):
					scheme_name_Array.append(scheme_name_temp)				

###################################################################################################################################################################################################
###################################################################################################################################################################################################		
			

#################################################### FUNCTION TO MAKE STRINGS TO CLASSES AS NUMBERS FOR INPUTS IN OUR NEURAL NETWORK #######################################################
##############################################################################################################################################################################################

def MakeDataset(Array):
	i = 0				###################################### COUNTER VARIABLE TO TRACK THE ARRAY OF THE DATASET ARRAY #########################
	length = len(Array[0])
	while (i < len(Array)):
		j = 0				###################################### COUNTER VARIABLE TO TRACK THE ELEMENT INSIDE THE ARRAY OF THE DATASET ARRAY #########################
		while (j < length):
			if (type(Array[i][j]) == str):
				if (j == 1):
					if (Array[i][j] == ""):
						Array[i][j] = "None"
					Array[i][j] = float(funder_Array.index(Array[i][j]))		################################## CHANGE FUNDER INFO TO NUMBER CLASSES ######################
					
				elif (j == 3):
					if (Array[i][j] == ""):
						Array[i][j] = "None"
					Array[i][j] = float(installer_Array.index(Array[i][j]))       ################################## CHANGE INSTALLER INFO TO NUMBER CLASSES ######################
					
				elif (j == 7):
					Array[i][j] = float(basin_Array.index(Array[i][j]))       ################################## CHANGE BASIN INFO TO NUMBER CLASSES ######################
					
				elif (j == 8):
					Array[i][j] = float(region_Array.index(Array[i][j]))       ################################## CHANGE REGION INFO TO NUMBER CLASSES ######################
					
				elif (j == 11):
					Array[i][j] = float(lga_Array.index(Array[i][j]))       ################################## CHANGE LGA INFO TO NUMBER CLASSES ######################
				
				elif (j == 12):
					Array[i][j] = float(ward_Array.index(Array[i][j]))       ################################## CHANGE WARD INFO TO NUMBER CLASSES ######################
				
				elif (j == 15):
					if (Array[i][j] == ""):
						Array[i][j] = "None"
					Array[i][j] = float(scheme_management_Array.index(Array[i][j]))       ######################## CHANGE SCHEME_MANAGEMENT INFO TO NUMBER CLASSES ############
				
				elif (j == 18):
					Array[i][j] = float(extraction_type_Array.index(Array[i][j]))       ############################ CHANGE EXTRACTION_TYPE INFO TO NUMBER CLASSES #############
				
				#elif (j == 19):
					#Array[i][j] = float(extraction_type_group_Array.index(Array[i][j]))       ################### CHANGE EXTRACTION_TYPE_GROUP INFO TO NUMBER CLASSES ###########
				
				#elif (j == 20):
					#Array[i][j] = float(extraction_type_class_Array.index(Array[i][j]))      ################### CHANGE EXTRACTION_TYPE_CLASS INFO TO NUMBER CLASSES ############
					
				elif (j == 19):
					Array[i][j] = float(management_Array.index(Array[i][j]))                ################### CHANGE MANAGEMENT INFO TO NUMBER CLASSES ###############
					
				elif (j == 20):
					Array[i][j] = float(management_group_Array.index(Array[i][j]))          ################### CHANGE MANAGEMENT_GROUP INFO TO NUMBER CLASSES ###########
					
				elif (j == 21):
					Array[i][j] = float(payment_Array.index(Array[i][j]))                ################### CHANGE PAYMENT INFO TO NUMBER CLASSES ###################
					
				elif (j == 22):
					Array[i][j] = float(water_quality_Array.index(Array[i][j]))            ################### CHANGE WATER_QUALITY INFO TO NUMBER CLASSES ##################
					
				#elif (j == 24):
					#Array[i][j] = float(water_quality_group_Array.index(Array[i][j]))       ################### CHANGE WATER_QUALITY_GROUP INFO TO NUMBER CLASSES ##############
				
				elif (j == 23):
					Array[i][j] = float(quantity_Array.index(Array[i][j]))             ################### CHANGE QQUANTITY INFO TO NUMBER CLASSES ###################
				
				elif (j == 24):
					Array[i][j] = float(source_Array.index(Array[i][j]))              ################### CHANGE SOURCE INFO TO NUMBER CLASSES ###################
				
				#elif (j == 28):
					#Array[i][j] = float(source_type_Array.index(Array[i][j]))            ################### CHANGE SOURCE_TYPE INFO TO NUMBER CLASSES ###################
					
				elif (j == 25):
					Array[i][j] = float(source_class_Array.index(Array[i][j]))            ################### CHANGE SOURCE_CLASS INFO TO NUMBER CLASSES ###################
					
				elif (j == 26):
					Array[i][j] = float(waterpoint_type_Array.index(Array[i][j]))          ################### CHANGE WATERPOINT_TYPE INFO TO NUMBER CLASSES #################
					
				#elif (j == 31):
					#Array[i][j] = float(waterpoint_type_group_Array.index(Array[i][j]))      ################# CHANGE WATERPOINT_TYPE_GROUP INFO TO NUMBER CLASSES ############
					
				elif (j == 27):
					Array[i][j] = float(scheme_name_Array.index(Array[i][j]))   ################ CHANGE WATERPOINT_TYPE_GROUP INFO TO NUMBER CLASSES ##############	
					
				else:										  ################# CHANGE BOOLEAN INFO TO NUMBER CLASSES ###################
					if (Array[i][j] == "True"):
						Array[i][j] = 1.0
					else:
						Array[i][j] = 0.0
				
			j += 1		################################## SET COUNTER TO NEXT ELEMENT OF DATASET ON I ARRAY LIST ############################
			
		i += 1			################################## SET COUNTER TO NEXT ARRAY OF DATASET ############################
	
	return(Array)
	
####################################################################################################################################################################################################				
####################################################################################################################################################################################################


########################################################### FUNCTION TO LOAD THE PREDICTIONS AND THE EXPECTED OUTPUTS OF OUR NEURAL NETWORK ########################################################
####################################################################################################################################################################################################

def LoadPredictions(filename):
	with open(filename) as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			if (row[0].isdigit()):
				if (str(row[1]) == "functional"):
					Predictions.append([1.0,0.0,0.0])
				elif (str(row[1]) == "functional needs repair"):
					Predictions.append([0.0,1.0,0.0])
				else:
					Predictions.append([0.0,0.0,1.0])
	return(Predictions)


#####################################################################################################################################################################################################
#####################################################################################################################################################################################################


	       

######################################################### FUNCTION TO RANDOM GENERATE TRAINING AND TESTING SAMPLES #################################################################################
####################################################################################################################################################################################################

def RandomGenerateSamples(length):
	i = 0
	train_counter = 0	################ VARIABLE TO HOLD THE AMMOUNT OF TRAINING SAMPLES ###################
	test_counter = 0	################ VARIABLE TO HOLD THE AMMOUNT OF TESTING SAMPLES ####################
	while (i < length):
		
		if ((train_counter < 50000) and (test_counter <9400)):
			rand = random.randint(0,1)
			if (rand == 1):
				Dataset_train.append(Dataset[i])
				Predictions_train.append(Predictions[i])
				train_counter += 1
			else:
				Dataset_test.append(Dataset[i])
				Predictions_test.append(Predictions[i])
				test_counter += 1
		else:
			if (train_counter >= 50000):
				Dataset_test.append(Dataset[i])
				Predictions_test.append(Predictions[i])
				test_counter += 1
			else:
				Dataset_train.append(Dataset[i])
				Predictions_train.append(Predictions[i])
				train_counter += 1		
		i+=1
	print("Number of trains: ", train_counter)
	print("Number of tests: ", test_counter)

###################################################################################################################################################################################################
###################################################################################################################################################################################################




######################################################################### FUNCTION TO TRAIN OUR NEURAL NETWORK ####################################################################################
###################################################################################################################################################################################################

def NeuralNetworkTrain(Array1,Array2):

	X_train = np.array(Array1)
	Y_train = np.array(Array2)
	clf = RandomForestClassifier().fit(X_train,Y_train)
	
	train_accuracy = clf.score(X_train,Y_train)
	print('Train accuracy is: ',train_accuracy * 100,"%\n")
	
	return(clf)

###################################################################################################################################################################################################
###################################################################################################################################################################################################





######################################################################### FUNCTION TO SELF TEST OUR NEURAL NETWORK ################################################################################
###################################################################################################################################################################################################

def NeuralNetworkSelfTest(Array1,Array2,clf):

	X_test = np.array(Array1)
	Y_test = np.array(Array2)
	
	result = clf.predict(X_test)

	test_accuracy = clf.score(X_test,Y_test)
	
	print('Test accuracy is: ',test_accuracy * 100,"%\n")

###################################################################################################################################################################################################
###################################################################################################################################################################################################



######################################################################### FUNCTION TO SELF TEST OUR NEURAL NETWORK ################################################################################
###################################################################################################################################################################################################

def NeuralNetworkTest(Array1,clf):

	X_test = np.array(Array1)
	
	result = clf.predict(X_test)
	
	print("Result1 is: ", result, "\n")
	
	return (result)

###################################################################################################################################################################################################
###################################################################################################################################################################################################






######################################################################### FUNCTION TO SELF TEST OUR NEURAL NETWORK ################################################################################
###################################################################################################################################################################################################

def WriteToFile(Array1,Array2):
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
		
		
###################################################################################################################################################################################################
###################################################################################################################################################################################################



	
######################################################################## FUNCTION TO CLEAR THE ARRAYS #############################################################################################
###################################################################################################################################################################################################

def EmptyArrays():
	global Dataset, Predictions, funder_Array, installer_Array, lga_Array, ward_Array, basin_Array, region_Array, scheme_management_Array, extraction_type_Array, extraction_type_group_Array
	global extraction_type_class_Array, management_Array, management_group_Array, payment_Array, water_quality_Array, water_quality_group_Array, quantity_Array, source_Array, source_type_Array
	global source_class_Array, waterpoint_type_Array, waterpoint_type_group_Array, scheme_name_Array, Id_type
	
	Dataset.clear()
	Predictions.clear()
	funder_Array.clear()
	installer_Array.clear()
	lga_Array.clear()
	ward_Array.clear()
	basin_Array.clear()
	region_Array.clear()
	scheme_management_Array.clear()
	extraction_type_Array.clear()
	extraction_type_group_Array.clear()
	extraction_type_class_Array.clear()
	management_Array.clear()
	management_group_Array.clear()
	payment_Array.clear()
	water_quality_Array.clear()
	water_quality_group_Array.clear()
	quantity_Array.clear()
	source_Array.clear()
	source_type_Array.clear()
	source_class_Array.clear()
	waterpoint_type_Array.clear()
	waterpoint_type_group_Array.clear()
	scheme_name_Array.clear()
	Id_type.clear()


##################################################################################################################################################################################################
##################################################################################################################################################################################################





########################################################################## MAIN FUNCTION TO RUN THE PROGRAM #######################################################################################
###################################################################################################################################################################################################

def main():
	global Dataset, Predictions, Id_type
	LoadDataset('training_values.csv')
	Dataset = MakeDataset(Dataset)
	Predictions = LoadPredictions('training_labels.csv')
	
	#RandomGenerateSamples(len(Dataset))
	
	#clf = NeuralNetworkTrain(Dataset_train,Predictions_train)
	#NeuralNetworkSelfTest(Dataset_test,Predictions_test,clf)
	
	clf = NeuralNetworkTrain(Dataset,Predictions)
	EmptyArrays()
	
	LoadDataset('test.csv')
	Dataset = MakeDataset(Dataset)
	result = NeuralNetworkTest(Dataset,clf)
	WriteToFile(result,Id_type)
	
	'''
	print("LENGTH OF DATASET: ",len(Dataset))
	print("LENGTH OF FUNDER ARRAYS: ", len(funder_Array))
	print("LENGTH OF INSTALLER ARRAY: ", len(installer_Array))
	print("LENGTH OF LGA: ",len(lga_Array))
	print("LENGTH OF WARD: ",len(ward_Array))
	print("LENGTH OF BASIN: ",len(basin_Array))
	print("LENGTH OF REGION: ",len(region_Array))
	print("LENGTH OF SCHEME_MANAGEMENT: ",len(scheme_management_Array))
	print("LENGTH OF EXTRACTION_TYPE: ",len(extraction_type_Array))
	print("LENGTH OF EXTRACTION_TYPE_GROUP: ",len(extraction_type_group_Array))
	print("LENGTH OF EXTRACTION_TYPE_CLASS: ",len(extraction_type_class_Array))
	print("LENGTH OF MANAGEMENT: ",len(management_Array))
	print("LENGTH OF MANAGEMENT_GROUP: ",len(management_group_Array))
	print("LENGTH OF PAYMENT: ",len(payment_Array))
	print("LENGTH OF WATER_QUALITY: ",len(water_quality_Array))
	print("LENGTH OF WATER_QUALITY_GROUP: ",len(water_quality_group_Array))
	print("LENGTH OF QUANTITY: ",len(quantity_Array))
	print("LENGTH OF SOURCE: ",len(source_Array))
	print("LENGTH OF SOURCE_TYPE: ",len(source_type_Array))
	print("LENGTH OF SOURCE_CLASS: ",len(source_class_Array))
	print("LENGTH OF WATERPOINT_TYPE: ",len(waterpoint_type_Array))
	print("LENGTH OF WATERPOINT_TYPE_GROUP: ",len(waterpoint_type_group_Array))
	print("\n")
	print("waterpoint_type: ", waterpoint_type_Array,"\n")
	print("waterpoint_type_group: ", waterpoint_type_group_Array, "\n")
	print("source_class: ", source_class_Array, "\n")
	'''
######################################################################################################################################################################################################
######################################################################################################################################################################################################

main()






















"""
print(extraction_type_Array)
print(extraction_type_group_Array)
print(extraction_type_class_Array)




for x in funder_Array:
	if (x == ""):
		print("YES1")
	if(x == "None"):
		print("NO1")
for x in installer_Array:
	if (x == ""):
		print("YES2")
	if(x == "None"):
		print("NO2")	
"""			
