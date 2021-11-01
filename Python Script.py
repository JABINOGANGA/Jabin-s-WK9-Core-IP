# Application of Naive Bayes as a Spam Detection Model
## 1. Defining the Question
### a) Specifying the Question

# > We have a set containg 57 features that are supposed to help predict if a message is a spam message or an actual message 
# > We intend to train the model using Naive Bayes to determine it's accuracy in detecting spam messages


# ### b) Defining the Metric for Success
# > The ability of the model to predict spam messages should be atleat 80% Accurate and not above 98% acccuarate this is to avoid overfitting 
# ### c) Understanding the context 
# > The datassets contains columns that contain analyses message content that has been presented in numerics to identify the patter spam messages take and actual messages to therefore allowing the machine to detect a spam message from an actual message assuming the relationships between the columns improving it's ability to detect random patterns which help on the test dataset.
# ### d) Recording the Experimental Design
# > wE intend to use Gaussian(NB)Classifier method as we will scale the data to almost assume a normal distribution then classify our data accordingly
# ### e) Data Relevance
# > The data contains 
# # Importing our libraries
# 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
# %matplotlib inline
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split
## 2. Reading the Data
# Loading the Dataset
spam = pd.read_csv('spambase Data.csv', header = None)

spam.head()
#
# renaming the columns

names = ['word_freq_make','word_freq_address','word_freq_all','word_freq_3d','word_freq_our','word_freq_over',
        'word_freq_remove','word_freq_internet','word_freq_order','word_freq_mail', 'word_freq_receive','word_freq_will',
        'word_freq_people','word_freq_report','word_freq_addresses','word_freq_free','word_freq_business','word_freq_email',
         'word_freq_you','word_freq_credit','word_freq_your','word_freq_font','word_freq_000','word_freq_money','word_freq_hp',
         'word_freq_hpl','word_freq_george','word_freq_650','word_freq_lab','word_freq_labs','word_freq_telnet','word_freq_857',
         'word_freq_data','word_freq_415','word_freq_85','word_freq_technology','word_freq_1999','word_freq_parts',
         'word_freq_pm','word_freq_direct','word_freq_cs','word_freq_meeting','word_freq_original','word_freq_project',
         'word_freq_re','word_freq_edu','word_freq_table','word_freq_conference','char_freq_;','char_freq_(','char_freq_[',
         'char_freq_!','char_freq_$','char_freq_#','capital_run_length_average','capital_run_length_longest',
         'capital_run_length_total','spam/target']
spam.columns = names

spam.head()
## 3. Checking the Data
# Determining the no. of records in our dataset
## We create a function that will enable us to check for various information about our dataset
# Determining the no. of records in our dataset the N0. of colums, the duplicates, datatypes 
# and if there are any null values and duplicates

def check(data):
    df = data.shape
    df1 = data.isnull().value_counts()
    df2 = data.duplicated().value_counts()
    df3 = data.columns
    df4 = data.info()
    df5 = data.dtypes
    


    print('Data_Shape',"\n", df,"\n")
    print('Columns',"\n", df3,"\n") 
    print('Data info',"\n", df4,"\n")
    print('Data types',"\n", df5,"\n")
    print('Checking for No. of null values',"\n", df1,"\n") 
    print('Checking for No. of duplicates',"\n", df2)
check(spam)


## 5. Tidying the Dataset
# Removing the duplicates and the null values 
spam.drop_duplicates(inplace = True )
spam.duplicated().sum()


## 6. Exploratory Analysis
# Ploting the univariate summaries and recording our observations
#

# fig, ax1 = plt.subplots()
# ax1.pie(data)\

pie = spam['spam/target'].value_counts()

pie.plot(kind = 'pie', autopct= '%1.1f%%', shadow = True, startangle = 90)
plt.axis('equal')
plt.title('Spam to actual messages')
plt.show()

#### Majority of the messages were categorized as spams 
# The correlation between different columns to identify the most correlated columns that can help detect 

spam.corr()
spam.columns
# After finding the most correlated columns we plotted a pairplot to show these relationships 


corr = spam[['char_freq_$', 'word_freq_your','word_freq_business', 'word_freq_receive', 'word_freq_000','word_freq_you', 'word_freq_remove', 'word_freq_free', 'word_freq_email','word_freq_hp','spam/target']]

sns.pairplot(corr, hue = 'spam/target')


## 7. Implementing the Solution
### Naive Bayes
# We slit our data to test and train data

# Implementing the Solution
# 
corr = spam[['char_freq_$', 'word_freq_your','word_freq_business', 'word_freq_receive', 'word_freq_000','word_freq_you', 'word_freq_remove', 'word_freq_free', 'word_freq_email','word_freq_hp','spam/target']]
X = corr.iloc[:, :-1].values
y = corr.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# > Scaling our data to reduce the mean to less than 1 and std to 1 in order to improve the models perfomance on the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Running the model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

pred = classifier.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)

# > The model perfomed well and got ana accuaracy of 84% which implies there is room for improvement
## Challenging the Solution
# Implementing the Solution by changing sample size from a ratio of 80:20 to 70:30
# 
corr = spam[['char_freq_$', 'word_freq_your','word_freq_business', 'word_freq_receive', 'word_freq_000','word_freq_you', 'word_freq_remove', 'word_freq_free', 'word_freq_email','word_freq_hp','spam/target']]
X = corr.iloc[:, :-1].values
y = corr.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.naive_bayes import GaussianNB
classifier1 = GaussianNB()
classifier1.fit(X_train, y_train)


# pred1= classifier1.predict(X_test)
# from sklearn.metrics import classification_report,confusion_matrix
# print(confusion_matrix(y_test,pred))
# print(classification_report(y_test,pred1))
# from sklearn.metrics import accuracy_score
# accuracy_score(y_test, pred)

# > The accuracy of the model slightly increases to 85.1% after increasing the sample size from 80 to 20 to 70 to 30%

## 8. Challenging the solution
# Tuning the parameters 

param_grid_nb = {
    'var_smoothing': np.logspace(0,-9, num=100)
}
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
nbModel_grid = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid_nb, verbose=1, cv=10, n_jobs=-1)
nbModel_grid.fit(X_train, y_train)
print(nbModel_grid.best_estimator_)
...

# Fitting 10 folds for each of 100 candidates, totalling 1000 fits

GaussianNB(priors=None, var_smoothing=1.0)
from sklearn.naive_bayes import GaussianNB
classifier2 = GaussianNB(var_smoothing=8.111308307896872e-05)
classifier2.fit(X_train, y_train)
pred2 = classifier2.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred2))

# Reviewing the Solution 
#

print(classification_report(y_test,pred2))
from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred2)


# > The model does not really improve from the sample change as it it sticks at 85.1 but it is more accurate than the baseline model
## Conclusion
# > Gaussian (NB) A Naive Bayes Model can be improved by changing the sample size and in adddition altering the parameters 
# ## 9. Follow up questions
# The model improved from 84.4 to 85.1 which is a slight change but the prediction was not generally good 
# > At this point, we can refine our question or collect new data, all in an iterative process to get at the truth.

