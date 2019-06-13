import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# reading dataset
data = pd.read_csv(".\ADDYY.csv")

# deleting unnecessary columns
data = data.drop(["Date","Adj Close","Volume"],axis=1)
print(data)
###########################################################################
###########################################################################
###########################################################################

# LET'S CREATE AN 'ACTION' ATTRIBUTE BASED ON OUR OWN DIFFERENT DECISIONS TO BUY OR SELL STOCKS:
# -Simple base approach
# -Detecting Kicking Bullish approach
# -Detecting Hammer approach

#####################################################################
# CALCULATING FUZZY IMPUTS: Let's extract fuzzy imputs from data to detect bullish patterns:
data['Lupper'] = 100 * ((data['High']-data[['Open','Close']].max(axis=1))/data['Open'])
data['Llower'] = 100 * ((data[['Open','Close']].min(axis=1) - data['Low'])/data['Open'])
data['Lbody'] = 100 * ((data['Close'] - data['Open'])/data['Close'])
data['Trend'] = 0 # initializing trend
for i in range(1,len(data)):
    data['Trend'][i] = 100 * ((data['Close'][i] - data['Close'][i-1])/data['Close'][i])

######################################################################
# CLASSIFICATION APPROACHES:

# A SIMPLE BASE APPROACH: create a column with buy or sell operations
# based only on the difference of open and close values: when it is positive we must sell, and viceversa
data['action'] = np.where((data['Close']-data['Open'])>0, 'SELL', 'BUY')

# Let's add an approach based on detecting Kicking Bullish:
# (we buy if we detect Kicking Bullish).
for i in range(1,len(data)):
        if data['High'][i-1] == data['Open'][i-1] and \
                data['Low'][i-1] == data['Close'][i-1] and \
                data['High'][i] == data['Open'][i] and \
                data['Low'][i] == data['Close'][i] and \
                data['Low'][i] > data['High'][i-1]:
                    data['action'][i] = 'BUY'

# Let's add the last approach based on detecting Hammer:
# (we buy if we detect a Hammer).
for i in range(2,len(data)):
        if data['Trend'][i] < 0 and \
                data['Trend'][i-1] < 0 and \
                data['Trend'][i-2] < 0 and \
                data['Low'][i] < data['Low'][i-1] and \
                ((data['High'][i] == data[['Open','Close']].max(axis=1)[i]) or ((data['High'][i]-data[['Open','Close']].max(axis=1)[i]) < (data['Lbody'][i]/5))) and \
                 (data[['Open','Close']].min(axis=1)[i]-data['Low'][i]) > (2*abs(data['Open'][i]-data['Close'][i])):
            data['action'][i] = 'BUY'

###########################################################################
###########################################################################
###########################################################################

# NOW WE CREATE OUR TRAIN AND TEST DATASETS

# we get the train data sets
trainSet = data.head(int(len(data)*0.67))
X_train = trainSet.drop(["action"],axis=1)
y_train = trainSet['action']

# we get the test data sets
testSet = data.tail(int(len(data)*0.33))
X_test = testSet.drop(["action"],axis=1)
y_test = testSet['action']

###########################################################################
###########################################################################
###########################################################################

# NOW WE CLASSIFY: KNN
neigh = KNeighborsClassifier(n_neighbors=3)

# fitting the model...
neigh.fit(X_train, y_train)

# getting predictions...
pred = neigh.predict(X_test)

# results:
print(accuracy_score(y_test, pred))