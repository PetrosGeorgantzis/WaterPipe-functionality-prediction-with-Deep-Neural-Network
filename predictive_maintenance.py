from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import csv
import numpy as np
import random

Id_type = []            ############ ARRAY TO KEEP INFO FOR THE ID, STRING_ID, TYPE OF EACH MACHINE #############################

Dataset = []            ############ ARRAY TO KEEP INFO FOR THE DATASET WE WILL USE FOR OUR DEEP NEURAL NETWORK #####################################

Predictions = []        ############ ARRAY TO KEEP INFO FOR THE CLASS OF EACH DATASET ###########################################


Dataset_train = []      ############ ARRAY TO KEEP INFO FOR THE DATASET WE WILL USE FOR TRAINING OUR DEEP NEURAL NETWORK #####################

Dataset_test = []       ############ ARRAY TO KEEP INFO FOR THE DATASET WE WILL USE FOR TESTING OUR DEEP NEURAL NETWORK #####################

Predictions_train = []  ############ ARRAY TO KEEP INFO FOR THE RECOMMENDED OUTPUTS WE WANT FROM TRAINING IN OUR DEEP NEURAL NETWORK #####################

Predictions_test = []   ############ ARRAY TO KEEP INFO FOR THE RECOMMENDED OUTPUTS WE WANT FROM TESTING IN OUR DEEP NEURAL NETWORK #####################

cl = 0                 ################ CLASS FOR EACH UNIQUE OUTPUT (6 HERE) ##########################
i = 0                  ################ JUST A COUNTER FOR LOOPS ##################
train_counter = 0      ################ VARIABLE TO HOLD THE AMMOUNT OF TRAINING SAMPLES ###################
test_counter = 0       ################ VARIABLE TO HOLD THE AMMOUNT OF TESTING SAMPLES ####################
rand_numbers = []      ################ ARRAY THAT KEEPS INFO FOR THE RANDOM NUMBERS BETWEEN 0,1 (0 FOR TESTING, 1 FOR TRAINING SAMPLES) ##################

with open('predictive_maintenance.csv') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	for row in reader:
		if (row[0].isdigit()):
			
			############# KEEP INFO FOR THE ID, STRING_ID AND TYPE ################### 
			Id_type.append([int(row[0]),str(row[1]),str(row[2])])
			#########################################################################
			
			############### KEEP INFO FOR THE DATASET WE WILL USE FOR MLP #####################################
			Dataset.append([float(row[3]),float(row[4]),float(row[5]),float(row[6]),float(row[7])])
			###################################################################################################
			
			################################ EACH TYPE OF FAILURE-NON FAILURE IS A DIFFRENT CLASS #############################
			if (str(row[9]) == "No Failure"):
				cl = 0
				Predictions.append([1.0,0.0,0.0,0.0,0.0,0.0])
			if (str(row[9]) == "Random Failures"):
				cl = 1
				Predictions.append([0.0,1.0,0.0,0.0,0.0,0.0])
			if (str(row[9]) == "Heat Dissipation Failure"):
				cl = 2
				Predictions.append([0.0,0.0,1.0,0.0,0.0,0.0])
			if (str(row[9]) == "Overstrain Failure"):
				cl = 3
				Predictions.append([0.0,0.0,0.0,1.0,0.0,0.0])
			if (str(row[9]) == "Power Failure"):
				cl = 4
				Predictions.append([0.0,0.0,0.0,0.0,1.0,0.0])
			if (str(row[9]) == "Tool Wear Failure"):
				cl = 5
				Predictions.append([0.0,0.0,0.0,0.0,0.0,1.0])
		       #############################################################################################################################
		       

########################################### RANDOM GENERATE TRAINING AND TESTING SAMPLES BOTH 5000 #############################################
while (i < len(Dataset)):

	if ((train_counter < 5000) and (test_counter <5000)):
		rand = random.randint(0,1)
		rand_numbers.append(rand)
		if (rand == 1):
			Dataset_train.append(Dataset[i])
			Predictions_train.append(Predictions[i])
			train_counter += 1
		else:
			Dataset_test.append(Dataset[i])
			Predictions_test.append(Predictions[i])
			test_counter += 1
	else:
		if (train_counter >= 5000):
			Dataset_test.append(Dataset[i])
			Predictions_test.append(Predictions[i])
			test_counter += 1
		else:
			Dataset_train.append(Dataset[i])
			Predictions_train.append(Predictions[i])
			train_counter += 1		
	i+=1
################################################################################################################################################################	

X_train = np.array(Dataset_train)
Y_train = np.array(Predictions_train)
clf = RandomForestClassifier().fit(X_train,Y_train)
X_test = np.array(Dataset_test)
Y_test = np.array(Predictions_test)
result = clf.predict_proba(X_test)
result1 = clf.predict(X_test)

train_accuracy = clf.score(X_train,Y_train)
test_accuracy = clf.score(X_test,Y_test)
print(X_test)
print('Train accuracy is: ',train_accuracy * 100,"%")
print('Test accuracy is: ',test_accuracy * 100,"%")

							
