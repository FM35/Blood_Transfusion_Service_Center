import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

Blood_data = pd.read_csv('Blood_Transfusion_Service_Center/transfusion.csv')
#print(Blood_data.describe(include = 'all'))

targets = Blood_data['whether he/she donated blood in March 2007']
inputs = Blood_data[['Recency (months)', 'Frequency (times)',]]
# 'Monetary (c.c. blood)', 'Time (months)'

#Feature Scaling the inputs
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)

#Splitting data into training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(inputs_scaled,targets ,test_size = 0.2, random_state = 365)

#Making regression model
x = sm.add_constant(x_train)
reg_log = sm.Logit(y_train,x)
results_log = reg_log.fit()
print(results_log.summary())

#Creating a table for Actual values from the dataset versus predictions that are based on data the model has already see, also calculating the accuracy
cm_df = pd.DataFrame(results_log.pred_table())
cm_df.columns = ['Predicted 0', 'Predicted 1']
cm_df = cm_df.rename(index = {0:'Actual 0', 1:'Actual 1'})
cm = np.array(cm_df)
accuracy_train = round(((cm[0,0] + cm[1,1])/cm.sum())*100, 2)

print(accuracy_train)

x1 = sm.add_constant(x_test)

#Function for printing out confusion matrix for data accuracy
def confusion_matrix(data, actual_values, model):
    pred_values = model.predict(data)
    bins = np.array([0, 0.5, 1])
    cm = np.histogram2d(actual_values, pred_values, bins = bins)[0]
    accuracy = (cm[0,0] + cm[1,1])/cm.sum()
    return cm,accuracy

cm = confusion_matrix(x1, y_test,results_log)
print(cm)