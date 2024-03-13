Welcome to the Health-care- wiki! Project - Healthcare Predict whether a patient has diabetes.

Project Steps Followed Define Project Goals/Objective Data Retrieval Data Cleansing Exploratory Data Analysis Data Modeling Result Analysis Project Goals/Objective To predict whether a patient has diabetes based on certain medical predictor variables.

Data Retrieval The dataset includes several predictors (pregnancies such as BMI, insulin level, age, and so on) and one target variable Data Analysis DataFrame.describe() is used to get the descriptive statistics. The descriptive statistics summarize the count of values in each column of the data set. We get the mean(), standard deviation, and interquartile ranges while excluding NaN values. However, the describe() method deals only with numeric values, not with any categorical values. The describe() method ignores the categorical values in a column and displays a summary for the other columns. To display the categorical values, we need to pass the parameter include="all"

proj_data.describe()

   Pregnancies     Glucose  ...         Age     Outcome
count 768.000000 768.000000 ... 768.000000 768.000000 mean 3.845052 120.894531 ... 33.240885 0.348958 std 3.369578 31.972618 ... 11.760232 0.476951 min 0.000000 0.000000 ... 21.000000 0.000000 25% 1.000000 99.000000 ... 24.000000 0.000000 50% 3.000000 117.000000 ... 29.000000 0.000000 75% 6.000000 140.250000 ... 41.000000 1.000000 max 17.000000 199.000000 ... 81.000000 1.000000

[8 rows x 9 columns]

proj_data.describe().T

                      count        mean  ...        75%     max
Pregnancies 768.0 3.845052 ... 6.00000 17.00 Glucose 768.0 120.894531 ... 140.25000 199.00 BloodPressure 768.0 69.105469 ... 80.00000 122.00 SkinThickness 768.0 20.536458 ... 32.00000 99.00 Insulin 768.0 79.799479 ... 127.25000 846.00 BMI 768.0 31.992578 ... 36.60000 67.10 DiabetesPedigreeFunction 768.0 0.471876 ... 0.62625 2.42 Age 768.0 33.240885 ... 41.00000 81.00 Outcome 768.0 0.348958 ... 1.00000 1.00

[9 rows x 8 columns] We notice that the minimum value of certain fields is zero. The zero value for columns such as Glucose, etc does not make sense. This may be due to missing values

The following fields have an invalid zero value

Glucose BloodPressure SkinThickness Insulin BMI We can handle the invalid zero values by replacing them with NaN. This step will help count them easily. We need to replace the zeros with suitable values.

We copy the dataframe to a new dataframe and then replace the zero values in the above given 5 variables with NaN.

proj_data_copy = proj_data.copy(deep = True) proj_data_copy'Glucose','BloodPressure','SkinThickness','Insulin','BMI' = proj_data_copy'Glucose','BloodPressure','SkinThickness','Insulin','BMI'.replace(0,np.NaN)

print(proj_data_copy.isnull().sum())

Pregnancies 0 Glucose 5 BloodPressure 35 SkinThickness 227 Insulin 374 BMI 11 DiabetesPedigreeFunction 0 Age 0 Outcome 0 dtype: int64 We can see that the zero values in the five columns have been updated with NaN

Before Updating the NaN values with suitable values, we need to understand the data distribution using EDA techniques

hplot = proj_data.hist(figsize = (20,20)) Plot

Now we replace the NAN values with the mean value of the variable. This helps to avoid data distortion due to invalid values

For each of the column, the NAN value is replaced with mean() value

proj_data_copy['Glucose'].fillna(proj_data_copy['Glucose'].mean(), inplace = True) proj_data_copy['BloodPressure'].fillna(proj_data_copy['BloodPressure'].mean(), inplace = True) proj_data_copy['SkinThickness'].fillna(proj_data_copy['SkinThickness'].median(), inplace = True) proj_data_copy['Insulin'].fillna(proj_data_copy['Insulin'].median(), inplace = True) proj_data_copy['BMI'].fillna(proj_data_copy['BMI'].median(), inplace = True) Now we plot the histogram of updated data and analyze the changes due to the previous step

hplot = proj_data_copy.hist(figsize = (20,20)) Plot

Pregnancies Glucose BloodPressure SkinThickness Insulin BMI DiabetesPedigree Age Outcome Import the Libraries

import numpy as np import pandas as pd import matplotlib.pyplot as plt import seaborn as sns Load the data

proj_data = pd.read_csv('../diabetes-data.csv') Analysis of Data

Print the first 5 rows of the data

proj_data.head()

Pregnancies Glucose BloodPressure ... DiabetesPedigreeFunction Age Outcome 0 6 148 72 ... 0.627 50 1 1 1 85 66 ... 0.351 31 0 2 8 183 64 ... 0.672 32 1 3 1 89 66 ... 0.167 21 0 4 0 137 40 ... 2.288 33 1

[5 rows x 9 columns] Data Type Analysis

Check the data types, columns, null value counts, memory usage etc

proj_data.info(verbose=True)

<class 'pandas.core.frame.DataFrame'> RangeIndex: 768 entries, 0 to 767 Data columns (total 9 columns):

Column Non-Null Count Dtype
0 Pregnancies 768 non-null int64
1 Glucose 768 non-null int64
2 BloodPressure 768 non-null int64
3 SkinThickness 768 non-null int64
4 Insulin 768 non-null int64
5 BMI 768 non-null float64 6 DiabetesPedigreeFunction 768 non-null float64 7 Age 768 non-null int64
8 Outcome 768 non-null int64
dtypes: float64(2), int64(7) memory usage: 54.1 KB

EDA Analyze the shape of the data

proj_data.shape

(768, 9) The shape of the data shows that there are 768 records and 9 variables

Analyze the Outcome Variable

print(proj_data.Outcome.value_counts())

0 500 1 268 Name: Outcome, dtype: int64 p=proj_data.Outcome.value_counts().plot(kind="bar")

Plot

It shows that the Outcome variable is binary and hence categorical. Value 0 represents the "Non-Diabetics" category whereas value 1 represents the "Diabetics" category. The counts of the "Non-Diabetics" category is approximately twice than that of the "Diabetics" category"

Since the outcome variable is categorical, we can use the classification methods such as KNN

Scatter Matrix of unclean data

from pandas.plotting import scatter_matrix p=scatter_matrix(proj_data,figsize=(25, 25)) Plot

The pair plot inclused histogram as well as scatter plot. The histogram is on the diagonal whereas the scatter plot is off-diagonal. The histogram shows the distribution of a single variable whereas the scatter plot shows the relationship between two variable

Pair Plot of Clean Data

p=sns.pairplot(proj_data_copy, hue = 'Outcome') Plot

Pearson's Correlation Coefficient You can use Pearson's Correlation Coefficient to analyze the correlation between the variables. The value of Pearson's Correlation Coefficient varies between -1 to +1. 1 means high correlation whereas 0 means no correlation.

We can construct a heat map to visualize the correlation matrix

Construct heat map for unclean data

plt.figure(figsize=(12,10)) p=sns.heatmap(proj_data.corr(), annot=True,cmap ='RdYlGn') Plot

Construct heat map for clean data

plt.figure(figsize=(12,10)) p=sns.heatmap(proj_data_copy.corr(), annot=True,cmap ='RdYlGn') Plot

We can observe the following

Change in heatmap due to data cleansing activities Correlation between Outcome and other input variables The highest correlation is between glucose level and outcome (0.49) The lowest correlation is between Blood Pressure and Outcome (0.17)

Scaling Data scaling is an important step. The preprocessed data may contain variables with different value ranges for various quantities such as dollars, kilograms, and sales volume. The machine learning models are more effective if the attributes have the same scale

Two important methods of scaling are:

Normalization Standardization Normalization: Normalization refers to rescaling real-valued numeric attributes into the range 0 and 1. We use sklearn.preprocessing.normalize() for Normalization. The mathematical formula is

Standardization: Standardization refers to shifting the distribution of each attribute to have a mean of zero and a standard deviation of one (unit variance). The mathematical formula is

In this case we would follow the following steps

from sklearn.preprocessing import StandardScaler Initialize the StandardScaler Fit Data to Scaler Object from sklearn.preprocessing import StandardScaler scale_X = StandardScaler()

X = scale_X.fit_transform(proj_data_copy.drop(["Outcome"],axis = 1),) X = pd.DataFrame(X,columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

X.head()

Pregnancies Glucose ... DiabetesPedigreeFunction Age 0 0.639947 0.865108 ... 0.468492 1.425995 1 -0.844885 -1.206162 ... -0.365061 -0.190672 2 1.233880 2.015813 ... 0.604397 -0.105584 3 -0.844885 -1.074652 ... -0.920763 -1.041549 4 -1.141852 0.503458 ... 5.484909 -0.020496

[5 rows x 8 columns] Note: It is always advisable to bring all the attributes at the same scale for models such as KNN. The attributes or features with greater range will overshadow or diminish the smaller attributes/feature completely. Hence it will impact the performance of the model because it will give higher weightage to attributes with higher magnitude.

Train Test Split Before fitting the data to the machine learning model, we should split the data into training data and testing data. This is an important step because we would like to train the model by fitting the training data. But to test the data, we should use the data that is new to model. Then only we would be able to calculate the performance of the model on the unseen data.

We use sklearn.model_selection.train_test_split() method for Train Test Split.

The first parameter of the train_test_split is test_size which specifies the ratio of data in the test dataset and test dataset. The value will put one-third values in the test data set and two-thirds values in the training data set.

The second parameter is random_state. Before splitting the data into training and test datasets, the data is randomly shuffled. By giving a value for the random state we ensure, the data is shuffled in a similar way every time so that you get the consistent training and test dataset.

The third parameter is stratify. Stratify parameter ensures that the proportion of values in the training and test data set will be the same as the proportion of values in the master dataset. For example, if variable is a binary categorical variable with values and . Suppose there are of zeros and of ones, stratify=y will make sure that your random split has of 0's and of 1's.

from sklearn.model_selection import train_test_split y = proj_data_copy.Outcome X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42, stratify=y)

Data Modeling For KNN, we will use sklearn.neighbors.KNeighborsClassifier method. and fit the X_train, y_train, X_test, and y_test datasets that we got from the train-test-split step

We have used a for loop to fit the data on KNN model with and store the scores

from sklearn.neighbors import KNeighborsClassifier testing_score = [] training_score = []

for i in range(1,15):

knn = KNeighborsClassifier(i)
knn.fit(X_train,y_train)

training_score.append(knn.score(X_train,y_train))
testing_score.append(knn.score(X_test,y_test))
We find out the value of for which the training accuracy was the highest

max_training_score = max(training_score) train_scores_ind = [i for i, v in enumerate(training_score) if v == max_training_score] print('Max training score {} % and k = {}'.format(max_training_score*100,list(map(lambda x: x+1, train_scores_ind))))

Max training score 100.0 % and k = [1] From the results, we can see that the training accuracy was 100% for . As we have seen from the text, KNN is highly flexible when We find out the value of for which the test accuracy was the highest

max_testing_score = max(testing_score) test_scores_ind = [i for i, v in enumerate(testing_score) if v == max_testing_score] print('Max testing score {} % and k = {}'.format(max_testing_score*100,list(map(lambda x: x+1, test_scores_ind))))

Max testing score 76.5625 % and k = [11] The testing accuracy is highest with Hence for the given data set, we can select the KNN with plt.figure(figsize=(12,5)) pplot = sns.lineplot(range(1,15),training_score,marker='*',label='Training Score') pplot = sns.lineplot(range(1,15),testing_score,marker='o',label='Testing Score') Plot

We can see that the highest testing accuracy is with . We can build our KNN model with knn = KNeighborsClassifier(11)

knn.fit(X_train,y_train) knn.score(X_test,y_test)

0.765625
