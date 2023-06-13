# Churn-Classification-Project
The project creates different classification project to determing the Churn rate at Vodafone 
Project Description
In this project, we aim to find the likelihood of a customer leaving the organization, the key indicators of churn, and the retention strategies that can be implemented to avert this problem.

Significance
The project will help identify the factors that contribute to customer churn. If we could figure out why a customer leaves and when they leave with reasonable accuracy, it would immensely help the organization strategize its retention initiatives.

Project Columns
Gender -- Whether the customer is a male or a female
SeniorCitizen -- Whether a customer is a senior citizen or not
Partner -- Whether the customer has a partner or not (Yes, No)
Dependents -- Whether the customer has dependents or not (Yes, No)
Tenure -- Number of months the customer has stayed with the company
Phone Service -- Whether the customer has a phone service or not (Yes, No)
MultipleLines -- Whether the customer has multiple lines or not
InternetService -- Customer's internet service provider (DSL, Fiber Optic, No)
OnlineSecurity -- Whether the customer has online security or not (Yes, No, No Internet)
OnlineBackup -- Whether the customer has online backup or not (Yes, No, No Internet)
DeviceProtection -- Whether the customer has device protection or not (Yes, No, No internet service)
TechSupport -- Whether the customer has tech support or not (Yes, No, No internet)
StreamingTV -- Whether the customer has streaming TV or not (Yes, No, No internet service)
StreamingMovies -- Whether the customer has streaming movies or not (Yes, No, No Internet service)
Contract -- The contract term of the customer (Month-to-Month, One year, Two year)
PaperlessBilling -- Whether the customer has paperless billing or not (Yes, No)
Payment Method -- The customer's payment method (Electronic check, mailed check, Bank transfer(automatic), Credit card(automatic))
MonthlyCharges -- The amount charged to the customer monthly
TotalCharges -- The total amount charged to the customer
Churn -- Whether the customer churned or not (Yes or No)

Questions
Do senior citizens have a higher churn rate than others?
2. Do customers with dependents have higher church rates?
3. Does age and gender contribute to the churn rate?
4. Is there a relationship between tenure and churn rate?
5. Does the contract term affect the churn rate?
6. Does the number of services signed on affect the churn rate?

Hypothesis
The contract term affects attrition
Null Hypothesis
The contract term does not affect attrition. 
Alternate Hypothesis
The contract term affects attrition.

Import the Necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

#relevant ML libraries
from imblearn.over_sampling import ADASYN
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold,StratifiedKFold,RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score
import sklearn.metrics as metrics
from sklearn.neighbors import LocalOutlierFactor





#ML models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier


warnings.filterwarnings('ignore')
# Next, we import the libraries for hypothesis testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
# Now we can load the dataframe to get a better understanding of its content
c_data = pd.read_csv('Telco-Customer-Churn.csv'
c_data.head())
Exploratory Data Analysis (EDA)
# We determine the data frame's shape and list its data types to see if there is any need for data cleaning
c_data.shape

c_data.info()
Findings: The total charges column has a data type of an object instead of a float. This must be converted.
c_data['TotalCharges'] = pd.to_numeric(c_data['TotalCharges'], errors='coerce')
Next we need to determine if the data has any null values
c_data.isnull().sum()
Findings: There are 11 missing values in the Total Charge column, which is very insignificant compared with the total number of entries. Therefore these missing values will be dropped.
# Removing missing value
c_data.dropna(inplace=True)s
Next, we look to determine if the data frame has any duplicates.
c_data.duplicated().sum()
Findings: There are no duplicated values in the data frame.
The next step in our EDA is removing id columns as they are irrelevant to the project.
# removing ID column
c_data = c_data.iloc[:,1:]
Next, we look at the number of unique values in each column
for column in c_data.columns:
  print(column, ':', c_data[column].nunique())
# Next, we load the one hot encoder to convert categorical values into numerical values in the preprocessing stage before creating machine learning modules.

Univariate Analysis
Here we use the describe() method to calculate some statistical data like percentile, mean, and std of the numerical values of the Churn DataFrame. The output reveals important. it is important to note the output will only be for numerical values. 
c_data.describe()
Relevant Findings: The mean monthly charges are 64.798208, the minimum value is 18.25, and the maximum value within this column is 118.75.
The mean tenure is 32.421786, the minimum value is 1, and the maximum value within this column is 72.
The mean total charges are 2283.300441, the minimum value is 18.80, and the maximum value within this column is 8684.80.
Distribution of Monthly Charges
We use the following code to develop a histogram of the total monthly charges
sns.histplot(x = 'MonthlyCharges', data = c_data).set(title = 'distribution of Monthly Charges')
No alt text provided for this image
Total monthly charges for the churn data frame
Results: Most customers are charged $20. Generally, the distribution of monthly charges is skewed to the right. Apart from $20, most customers are charged between $40 to $110.
Distribution of Customer Tenure
We use the subsequent code to create a histogram for the tenure column.
sns.histplot(x = 'tenure', data = c_data).set(title = 'Tunure Across Board')
No alt text provided for this image
Histogram for Tenure for the churn data frame
Most customers only last 0-15 months. However, tenure is considerably strong for customers who stay about 74 months.
Distribution of Customers who use Paperless Billing
The subsequent code looks to determine the number of customers who are subscribed to paperless billing
ax = sns.countplot(x='PaperlessBilling',data=c_data
vc = c_data['PaperlessBilling'].value_counts()
for p in ax.patches:
    # Get the x and y coordinates of the bar
    x_coord = p.get_x() + p.get_width() / 2
    y_coord = p.get_height()
    
    # Format the text you want to display as the annotation
    count = int(y_coord)  # Convert the count to an integer if needed
    text = f'{count}'  # Customize the text as desired
    
    # Add the annotation to the plot
    ax.text(x=x_coord, y=y_coord/2, s=text, ha='center', va='bottom')
plt.tight_layout()
plt.title('Count of Customers Paperless Billing')
plt.show())
output:
No alt text provided for this image
Customers who use paperless billing
Results: approximately 4168 employees use paper billing services while people 2864 do not.
Distribution of Customers Who Use the Payment Method
The subsequent code looks to determine the number of customers who are subscribed to the payment method
ax = sns.countplot(x='PaymentMethod',data=c_data)
vc = c_data['PaymentMethod'].value_counts()
for p in ax.patches:
    # Get the x and y coordinates of the bar
    x_coord = p.get_x() + p.get_width() / 2
    y_coord = p.get_height()
    
    # Format the text you want to display as the annotation
    count = int(y_coord)  # Convert the count to an integer if needed
    text = f'{count}'  # Customize the text as desired
    
    # Add the annotation to the plot
    ax.text(x=x_coord, y=y_coord/2, s=text, ha='center', va='bottom')
plt.tight_layout()
plt.title('Count of Customers  Payment Method')
plt.show()
Output:
No alt text provided for this image
Number of different customers who use different payment methods
Results: Electronic check is the most popular customer payment method, while automatic credit card is the least popular one. This implies the electronic checks offer a high level of convenience for businesses and customers. With e-checks, transactions can be initiated and completed online, eliminating the need for physical checks, paper handling, and manual processing. Customers can make payments from the comfort of their homes or offices, saving time and effort compared to traditional check writing and mailing.
Distribution of Customers Who Use Different Contracts
The subsequent code looks to determine the number of customers who are subscribed to the different contracts.
ax = sns.countplot(x='Contract',data=c_data
vc = c_data['Contract'].value_counts()
for p in ax.patches:
    # Get the x and y coordinates of the bar
    x_coord = p.get_x() + p.get_width() / 2
    y_coord = p.get_height()
    
    # Format the text you want to display as the annotation
    count = int(y_coord)  # Convert the count to an integer if needed
    text = f'{count}'  # Customize the text as desired
    
    # Add the annotation to the plot
    ax.text(x=x_coord, y=y_coord/2, s=text, ha='center', va='bottom')
plt.tight_layout()
plt.title('Count of Customers  Contract')
plt.show())
Output:
No alt text provided for this image
Customer count per contract
Results: Most customers are subscribed to month-to-month contracts (3875); surprisingly, one-year contracts have fewer subscriptions (1472) than two-year contracts (1685).
Distribution of Total Charges
This helps us determine customers' frequency distribution based on total charges.
sns.histplot(x = 'TotalCharges', data = c_data).set(title = 'Distribution of Total Charges')
Output:
No alt text provided for this image
Customer count based on total charges .
Results: customer frequency decreases with an increase in total charges. The relationship between customer frequency and total charges exhibits an intriguing pattern. As the total charges increase, there is a notable decrease in customer frequency. This phenomenon can be attributed to several factors.
For instance, customers may become more cautious about their spending habits as the total charges rise. They may prioritize their expenses and cut back on non-essential purchases or services. This shift in mindset leads to a decrease in the frequency of their interactions with the business.
Secondly, the increase in total charges might indicate that customers engage in larger, one-time transactions rather than frequent, smaller ones. They may be opting for bulk purchases or higher-priced items, which naturally reduces the frequency of their visits or transactions.
Additionally, an increase in total charges could imply that the business has implemented changes such as price adjustments, additional fees, or premium services. These alterations may result in some customers seeking alternatives or exploring more cost-effective options, reducing their engagement frequency with the business.
Moreover, as the total charges escalate, customers may perceive the products or services as less affordable or less aligned with their budgetary constraints. This perception can discourage them from engaging with the business as frequently as before.
Count of Customers who Churn or not
This section looks at the number of customers who churn from the company.
ax = sns.countplot(x='Churn',data=c_data, palette='rainbow')
vc = c_data['Churn'].value_counts()
for p in ax.patches:
    # Get the x and y coordinates of the bar
    x_coord = p.get_x() + p.get_width() / 2
    y_coord = p.get_height()
    
    # Format the text you want to display as the annotation
    count = int(y_coord)  # Convert the count to an integer if needed
    text = f'{count}'  # Customize the text as desired
    
    # Add the annotation to the plot
    ax.text(x=x_coord, y=y_coord/2, s=text, ha='center', va='bottom')
plt.tight_layout()
plt.title('Count of Customer who Churn or not')
plt.show()
Results: 5174 customers did not churn, while 1869 churn. This makes target variable imbalances. Some of the reasons for churn include:
Poor customer service: Customers who experience subpar or unsatisfactory customer service are likelier to churn. This can include long response times, unhelpful support staff, difficulty resolving issues, or a lack of personalized attention. Customers want to feel valued and supported, and when their needs are unmet, they may seek alternatives.
Lack of product or service relevance: If a product or service no longer meets the needs or expectations of customers, they may decide to churn. Changes in their requirements, advancements in technology, or the availability of better alternatives can render a product or service less valuable. Regularly assessing and adapting offerings to match evolving customer demands is crucial in reducing churn.
Competitive market: In highly competitive markets, customers have many choices. Customers may be enticed to switch if a competitor offers a superior product, service, or price. It is essential for businesses to understand their competitive landscape and continually work on differentiating themselves to retain customers.
Pricing issues: Pricing plays a significant role in customer churn. If customers perceive a product or service to be overpriced or if they find better deals elsewhere, they may choose to switch to a more affordable option. Regularly evaluating pricing strategies and ensuring they align with customer expectations and market conditions can help reduce churn.
Negative customer experiences: Negative experiences, such as product failures, billing errors, or shipping delays, can lead to customer dissatisfaction and churn. Consistently delivering high-quality products, resolving issues promptly, and proactively addressing customer concerns are essential in maintaining customer satisfaction and loyalty.
Lack of engagement or communication: Customers may be more inclined to churn when they feel disconnected from a business. Insufficient communication, infrequent updates, and a lack of engagement initiatives can contribute to this sense of detachment. Building strong relationships through regular communication, personalized interactions, and targeted marketing efforts can help retain customers.
Life changes or business shifts: Customers' needs and circumstances may change over time. Personal life events, such as relocation, job changes, or business-related factors like mergers, acquisitions, or closures, can lead to customer churn. While some changes are beyond a business's control, providing flexibility, adaptability, and support during transitions can help minimize churn.
Exploration of Customer Distribution by Gender
This helps us determine the distribution of customers based on gender.
ax = sns.countplot(x='gender',data=c_data)
vc = c_data['gender'].value_counts()
for p in ax.patches:
    # Get the x and y coordinates of the bar
    x_coord = p.get_x() + p.get_width() / 2
    y_coord = p.get_height()
    
    # Format the text you want to display as the annotation
    count = int(y_coord)  # Convert the count to an integer if needed
    text = f'{count}'  # Customize the text as desired
    
    # Add the annotation to the plot
    ax.text(x=x_coord, y=y_coord/2, s=text, ha='center', va='bottom')
plt.tight_layout()
plt.title('Count of Gender')
plt.show()
No alt text provided for this image
Customer distribution based on gender
Results: The customer distribution by gender is comparable, with 3549 males and 3483 females.
Distribution of Customers with Partner
This helps us determine the number of clients with and without partners.

ax = sns.countplot(x='Partner',data=c_data)
vc = c_data['Partner'].value_counts()
for p in ax.patches:
    # Get the x and y coordinates of the bar
    x_coord = p.get_x() + p.get_width() / 2
    y_coord = p.get_height()
    
    # Format the text you want to display as the annotation
    count = int(y_coord)  # Convert the count to an integer if needed
    text = f'{count}'  # Customize the text as desired
    
    # Add the annotation to the plot
    ax.text(x=x_coord, y=y_coord/2, s=text, ha='center', va='bottom')
plt.tight_layout()
plt.title('Count of Customers with Partner')
plt.show()
Output:
No alt text provided for this image
Count of Customers with partners
Results: The difference between customers with partners and those without partners is 243, which is approximately 3.45% of all customers.
Distribution of Customers with Dependents
This helps us determine the number of clients with and without dependents.
ax = sns.countplot(x='Dependents',data=c_data)
vc = c_data['Dependents'].value_counts()
for p in ax.patches:
    # Get the x and y coordinates of the bar
    x_coord = p.get_x() + p.get_width() / 2
    y_coord = p.get_height()
    
    # Format the text you want to display as the annotation
    count = int(y_coord)  # Convert the count to an integer if needed
    text = f'{count}'  # Customize the text as desired
    
    # Add the annotation to the plot
    ax.text(x=x_coord, y=y_coord/2, s=text, ha='center', va='bottom')
plt.tight_layout()
plt.title('Count of Customers with Dependents')
plt.show()
No alt text provided for this image
Count of customers with dependents
Results: Most of the customers do not have dependents. About 4,933 customers do not have dependents, whiles 2,110 have dependents.
Distribution of Senior Citizens
This helps us determine the number of customers categorized as senior citizens.
ax = sns.countplot(x='SeniorCitizen', data=c_data)
vc = c_data['SeniorCitizen'].value_counts()
for p in ax.patches:
    # Get the x and y coordinates of the bar
    x_coord = p.get_x() + p.get_width() / 2
    y_coord = p.get_height()
    
    # Format the text you want to display as the annotation
    count = int(y_coord)  # Convert the count to an integer if needed
    text = f'{count}'  # Customize the text as desired
    
    # Add the annotation to the plot
    ax.text(x=x_coord, y=y_coord/2, s=text, ha='center', va='bottom')
plt.tight_layout()
plt.title('Count of Senior Citizen')
plt.show()
No alt text provided for this image
Count of Senior Citizens
Results: Most customers are young and cannot be categorized as senior citizens. However, 1,142 are categorized as senior citizens.
Bivariate Analysis
Gender vs. Monthly Charges
This helps to determine the distribution of monthly charges per gender
sns.violinplot(x='gender',y='tenure',data=c_data,palette='rainbow').set(title = 'distribution of Tenure')
No alt text provided for this image

Results: The distribution of tenure per gender is comparable. This means that the tenure of each gender within the company is comparable.
Gender vs. Monthly Charges
This helps to determine the distribution of monthly chargers per gender.
sns.boxplot(x='gender',y='MonthlyCharges',data=c_data).set(title = 'distribution of Monthly Charges')
No alt text provided for this image
Distribution of monthly Charges ber gender
Results: Monthly charges distribution is comparable between genders. This means males and females spend comparatively similarly amounts on monthly charges.
Dependents vs Tenure
This helps to determine the distribution of customer tenure vs. dependents.
No alt text provided for this image
Tenure vs. Dependents
Results: Customers with dependents have a longer tenure than customers who do not have dependents. This may be attributed to a variety of factors.
Dependents vs. Monthly Charges
This helps to determine the distribution of customer Monthly dependent vs. dependents.
No alt text provided for this image
Dependents vs. Monthly Charges
Result: This means people with no dependents have higher monthly charges than customers with dependents. This may be attributed to the fact that customer without dependents have higher disposable income since they have fewer obligations.
Churn vs. Tenure
This helps to determine the distribution of churn vs. tenure.
sns.boxplot(x='Churn',y='tenure', data=c_data).set(title = 'Churn vs. Monthly Charges')
No alt text provided for this image
Churn vs. Tenure
Results: customers who churn have lower tenure than customers who do not churn. This implies the churn rate determines customer tenure.
Churn vs. Monthly Charges
This helps to determine the distribution of churn vs. monthly charges
sns.violinplot(x='Churn',y='MonthlyCharges', data=c_data, palette='rainbow').set(title = 'Churn vs. Monthly Charges')
No alt text provided for this image
Churn vs, Monthly Charges
Results: Customers who churn have higher monthly charges than customers who do not churn. The high monthly charges may underpin their high churn rate.
Pairplot
This pair plot is used to understand the best set of features to explain a relationship between two variables or to form the most separated clusters.
pair_grid = sns.pairplot(c_data)
No alt text provided for this image
pairplot on the churn dataset
Relationship between Phone Service and Churn
This helps to determine the distribution of phone services vs. churn.
sns.set_style('darkgrid')    
sns.countplot(x='PhoneService', hue='Churn', data=c_data, palette='Blues')
vc = c_data['PhoneService'].value_counts()
for p in ax.patches:
    # Get the x and y coordinates of the bar
    x_coord = p.get_x() + p.get_width() / 2
    y_coord = p.get_height()
    
    # Format the text you want to display as the annotation
    count = int(y_coord)  # Convert the count to an integer if needed
    text = f'{count}'  # Customize the text as desired
    
    # Add the annotation to the plot
    ax.text(x=x_coord, y=y_coord/2, s=text, ha='center', va='bottom')
plt.tight_layout()
plt.title('Relationship between Phone Service and Churn')
plt.show() 
No alt text provided for this image
Phone service vs, churn
Results: most customer with high churn rate have high phone service use.
Relationship between Multiple Lines and Churn
This will look at the association between customers who use multiple lines and churn.
sns.set_style('darkgrid')    
sns.countplot(x='MultipleLines', hue='Churn', data=c_data, palette='Blues')
vc = c_data['MultipleLines'].value_counts()
for p in ax.patches:
    # Get the x and y coordinates of the bar
    x_coord = p.get_x() + p.get_width() / 2
    y_coord = p.get_height()
    
    # Format the text you want to display as the annotation
    count = int(y_coord)  # Convert the count to an integer if needed
    text = f'{count}'  # Customize the text as desired
    
    # Add the annotation to the plot
    ax.text(x=x_coord, y=y_coord/2, s=text, ha='center', va='bottom')
plt.tight_layout()
plt.title('Relationship between Multiple Lines and Churn')
plt.show() 
No alt text provided for this image
multiple lines vs. churn
Results: customers with multiple line have higher churn rate than customers with no phone services.
Relationship between Online Backup and Churn
This sections explores the association between online backup and churn.
sns.set_style('darkgrid')    
sns.countplot(x='OnlineBackup', hue='Churn', data=c_data, palette='Blues')
vc = c_data['OnlineBackup'].value_counts()
for p in ax.patches:
    # Get the x and y coordinates of the bar
    x_coord = p.get_x() + p.get_width() / 2
    y_coord = p.get_height()
    
    # Format the text you want to display as the annotation
    count = int(y_coord)  # Convert the count to an integer if needed
    text = f'{count}'  # Customize the text as desired
    
    # Add the annotation to the plot
    ax.text(x=x_coord, y=y_coord/2, s=text, ha='center', va='bottom')
plt.tight_layout()
plt.title('Relationship between Online Backup and Churn')
plt.show() 
No alt text provided for this image
Online backup vs churn
Results: People with no online backups have high churn rate than customers with online backups.
Relationship between Online Security and Churn
This section looks at the association between online security and churn.
sns.set_style('darkgrid')    
sns.countplot(x='OnlineSecurity', hue='Churn', data=c_data, palette='Blues')
vc = c_data['OnlineSecurity'].value_counts()
for p in ax.patches:
    # Get the x and y coordinates of the bar
    x_coord = p.get_x() + p.get_width() / 2
    y_coord = p.get_height()
    
    # Format the text you want to display as the annotation
    count = int(y_coord)  # Convert the count to an integer if needed
    text = f'{count}'  # Customize the text as desired
    
    # Add the annotation to the plot
    ax.text(x=x_coord, y=y_coord/2, s=text, ha='center', va='bottom')
plt.tight_layout()
plt.title('Relationship between Online Security and Churn')
plt.show() 
No alt text provided for this image
Online security vs. churn
Results: customers without online security have higher churn rate than customer with online security.
Relationship between Streaming Movies and Churn
This section assesses the association between streaming movies and churn.
sns.set_style('darkgrid')    
sns.countplot(x='StreamingMovies', hue='Churn', data=c_data, palette='Blues')
vc = c_data['StreamingMovies'].value_counts()
for p in ax.patches:
    # Get the x and y coordinates of the bar
    x_coord = p.get_x() + p.get_width() / 2
    y_coord = p.get_height()
    
    # Format the text you want to display as the annotation
    count = int(y_coord)  # Convert the count to an integer if needed
    text = f'{count}'  # Customize the text as desired
    
    # Add the annotation to the plot
    ax.text(x=x_coord, y=y_coord/2, s=text, ha='center', va='bottom')
plt.tight_layout()
plt.title('Relationship between Streaming Movies and Churn')
plt.show() 
No alt text provided for this image
Streaming movies vs. churn

Results: Customers who stream movies have a lower churn rate than customers who do not stream movies.
Relationship between Streaming TV and Churn 
The section looks at the association between streaming tv and churn.
sns.set_style('darkgrid')    
sns.countplot(x='StreamingTV', hue='Churn', data=c_data, palette='Blues')
vc = c_data['StreamingTV'].value_counts()
for p in ax.patches:
    # Get the x and y coordinates of the bar
    x_coord = p.get_x() + p.get_width() / 2
    y_coord = p.get_height()
    
    # Format the text you want to display as the annotation
    count = int(y_coord)  # Convert the count to an integer if needed
    text = f'{count}'  # Customize the text as desired
    
    # Add the annotation to the plot
    ax.text(x=x_coord, y=y_coord/2, s=text, ha='center', va='bottom')
plt.tight_layout()
plt.title('Relationship between Streaming TV and Churn')
plt.show() 
No alt text provided for this image
streamiing tv vs churn
Results: Comparatively, customers who do not stream tv have higher churn rate than customers who do.
Relationship between Tech Support and Churn
This section looks at the relationship between tech support and churn.
sns.set_style('darkgrid')    
sns.countplot(x='TechSupport', hue='Churn', data=c_data, palette='Blues')
vc = c_data['TechSupport'].value_counts()
for p in ax.patches:
    # Get the x and y coordinates of the bar
    x_coord = p.get_x() + p.get_width() / 2
    y_coord = p.get_height()
    
    # Format the text you want to display as the annotation
    count = int(y_coord)  # Convert the count to an integer if needed
    text = f'{count}'  # Customize the text as desired
    
    # Add the annotation to the plot
    ax.text(x=x_coord, y=y_coord/2, s=text, ha='center', va='bottom')
plt.tight_layout()
plt.title('Relationship between TechSupport and Churn')
plt.show() 
No alt text provided for this image
Tech support vs. churn
Results: Customer who do not have tech support have significantly higher churn rate than customers with tech support.
Relationship between Device Protection and Churn
This section looks at the association between device protection and churn.
sns.set_style('darkgrid')    
sns.countplot(x='SeniorCitizen', hue='Churn', data=c_data, palette='Blues')
vc = c_data['SeniorCitizen'].value_counts()
for p in ax.patches:
    # Get the x and y coordinates of the bar
    x_coord = p.get_x() + p.get_width() / 2
    y_coord = p.get_height()
    
    # Format the text you want to display as the annotation
    count = int(y_coord)  # Convert the count to an integer if needed
    text = f'{count}'  # Customize the text as desired
    
    # Add the annotation to the plot
    ax.text(x=x_coord, y=y_coord/2, s=text, ha='center', va='bottom')
plt.tight_layout()
plt.title('Relationship between Senior Citizen and Churn')
plt.show() 
No alt text provided for this image
Senior Citizen vs. Churn
Results: Non-senior citizens have higher churn rate than senior citizens.
Relationship between Internet Service and Churn
This section looks at the relationship between churn and internet service.
sns.set_style('darkgrid'
sns.countplot(x='InternetService', hue='Churn', data=c_data, palette='Blues')
vc = c_data['InternetService'].value_counts()
for p in ax.patches:
    # Get the x and y coordinates of the bar
    x_coord = p.get_x() + p.get_width() / 2
    y_coord = p.get_height()
    
    # Format the text you want to display as the annotation
    count = int(y_coord)  # Convert the count to an integer if needed
    text = f'{count}'  # Customize the text as desired
    
    # Add the annotation to the plot
    ax.text(x=x_coord, y=y_coord/2, s=text, ha='center', va='bottom')
plt.tight_layout()
plt.title('Relationship between Internet Service and Churn')
plt.show())
No alt text provided for this image
Internet service vs. Churn
Results: customers who use fiber optics have the highest churn rate, customers who use DSL also have higher churn rate than those with no internet services.
Multivariate Analysis
The first step is to encode categorical columns.
Encoding of Categorical Columns
LE = LabelEncoder(
c_data['gender'] = LE.fit_transform(c_data['gender'])
c_data['Partner'] = LE.fit_transform(c_data['Partner'])
c_data['Dependents'] = LE.fit_transform(c_data['Dependents'])
c_data['PhoneService'] = LE.fit_transform(c_data['PhoneService'])
c_data['MultipleLines'] = LE.fit_transform(c_data['MultipleLines'])
c_data['InternetService'] = LE.fit_transform(c_data['InternetService'])
c_data['OnlineSecurity'] = LE.fit_transform(c_data['OnlineSecurity'])
c_data['OnlineBackup'] = LE.fit_transform(c_data['OnlineBackup'])
c_data['DeviceProtection'] = LE.fit_transform(c_data['DeviceProtection'])
c_data['TechSupport'] = LE.fit_transform(c_data['TechSupport'])
c_data['StreamingTV'] = LE.fit_transform(c_data['StreamingTV'])
c_data['StreamingMovies'] = LE.fit_transform(c_data['StreamingMovies'])
c_data['Contract'] = LE.fit_transform(c_data['Contract'])
c_data['PaperlessBilling'] = LE.fit_transform(c_data['PaperlessBilling'])
c_data['PaymentMethod'] = LE.fit_transform(c_data['PaymentMethod']))
Next, we plot a correlation matrix
# Compute correlation matrix
correlation_matrix = c_data.corr()


# Plot correlation matrix as a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation of Variables with Churn')
plt.show()
No alt text provided for this image
correlation matrix for churn dataset
Results: The illustration show the correlation between the different variables in the dataset.
 Hypothesis Testing
One-way ANOVA (Analysis of Variance) is a statistical test used to determine if there are significant differences among the means of three or more independent groups. It helps assess whether the variation between group means is larger than the variation within each group.
model = ols('Churn ~ Contract', data = c_data).fit()
anova_result = sm.stats.anova_lm(model, typ = 2)
print(anova_result)
Results: Based on the extremely small p-value (6.065595e-263), which is essentially zero, obtained from the ANOVA test, we can conclude that there is strong evidence to reject the null hypothesis. This indicates that there are significant differences among the group means being compared (Churn and Contract). We therefore conclude that the contract term affects attrition.
In other words, the observed variation between the group means is not due to random chance alone but rather reflects true differences in the population. The F-value, in conjunction with the small p-value, provides support for rejecting the null hypothesis and accepting the alternative hypothesis that at least one group mean is significantly different from the others.
Machine Learning
#For machine learning use bagging and boosting, these do not need dealing with imbalance data. 
#Ensemble methods: Ensemble methods, such as bagging or boosting, can be effective in handling imbalanced datasets. 
#Techniques like Random Forests or Gradient Boosting can better handle class imbalance by combining multiple weak models.
#For other machine learning deal with imbalance data by using SMOTE (Synthetic Minority Over-sampling Technique), or ADASYN (Adaptive Synthetic Sampling).
y= c_data['Churn']                         # Target Variabl
X = c_data.drop('Churn', axis =1)          # Independent Variablee

scaler = StandardScaler()
cs = scaler.fit_transform(X)
csn = pd.DataFrame(cs, columns = X.columns)
csn.head(5)

def plot_metric(confusion, name):
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in
    confusion.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
    confusion.flatten()/np.sum(confusion)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
    zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    ax = sns.heatmap(confusion, annot=labels, fmt='', cmap='viridis')
    ax.set_title(f'{name}\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');
    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])
    ## Display the visualization of the Confusion Matrix.
    plt.show()
Machine Models With Resampling
X_resampled,y_resampled = ADASYN().fit_resample(X,y
X_resampled.shape,y_resampled.shape)

print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape {}'.format(Counter(y_resampled)))

X_s = X_resampled
y_s = y_resampled


# split data into train+validation set and test set
X_trainval, X_test, y_trainval, y_test = train_test_split(X_s, y_s, random_state=0)


# split train+validation set into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)


print("Size of training set: {}\n size of validation set: {}\n size of test set:" " {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

Logistic Regression
# split data into train+validation set and test set
X_trainval, X_test, y_trainval, y_test = train_test_split(X_s, y_s, random_state=0)


# split train+validation set into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)


print("Size of training set: {}\n size of validation set: {}\n size of test set:" " {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))
No alt text provided for this image
Logistic regression
Confusion Matrix
Logistic Regression Classifier classified 740 about 38%, correctly True Negative], which is the correct prediction of customers who did not churn and failed to classify 245 about 13%, correctly [False Positive], which is the incorrect prediction of customers who did not churn.
Logistic Regression Classifier classified 786 about 40%, correctly [True Positive], which is the correct prediction of customers who churn and failed to classify 178 about 9%, correctly [False Negative], which is the incorrect prediction of customers who did churn.
Random Forest
crf = RandomForestClassifier()
# Fit 'LPknn' to the training set
crf.fit(X_train, y_train)


# Predict Output
crf_predict = crf.predict(X_valid)
pred_crf = crf.predict_proba(X_valid)[:,1]

#Train and Test Scores
crf_Tr_Score = round(crf.score(X_train, y_train)*100, 2)
crf_Tt_Score = round(crf.score(X_valid, y_valid)*100, 2)
print('train set score: {:.2f}'.format(crf_Tr_Score))
print('test set score: {:.2f}'.format(crf_Tt_Score))






print()
crf_sc = round(accuracy_score(y_valid, crf_predict)*100, 2)
print("Accuracy Score: {}%".format(crf_sc))
print()
PS_crf = round(precision_score(y_valid, crf_predict)*100, 2)
print("Precision Score: {}%".format(PS_crf))
print()
RS_crf = round(recall_score(y_valid, crf_predict)*100, 2)
print("Recall Score: {}%".format(RS_crf))
print()
FS_crf = round(f1_score(y_valid, crf_predict)*100, 2)
print("F1 Score: {}%".format(FS_crf))
print()
fpr2, tpr2, threshold = roc_curve(y_valid, pred_crf)
roc_auc = metrics.auc(fpr2, tpr2)
crfA = round(metrics.auc(fpr2, tpr2)*100, 2)
print(f'AUC score is:', crfA,'%')
print()
confusion = confusion_matrix(y_valid, crf_predict)
#print("Confusion matrix:\n{}".format(confusion))
plot_metric(confusion, "Random Forest")
No alt text provided for this image
random classifier
Confusion Matrix
Random Forest Classifier classified 827, about 42%, correctly [True Negative], which is the correct prediction of customers who did not churn and failed to classify 158, about 8%, correctly [False Positive], which is the incorrect prediction of customers who did not churn.

Random Forest Classifier classified 801, about 41%, correctly [True Positive], which is the correct prediction of customers who churn and failed to classify 163, about 8%, correctly [False Negative], which is the incorrect prediction of customers who did churn.
Support Vector Machine
csvc = SVC(gamma='auto', probability=True)
# Fit 'LPknn' to the training set
csvc.fit(X_train, y_train)


# Predict Output
csvc_predict = csvc.predict(X_valid)
pred_csvc = csvc.predict_proba(X_valid)[:,1]
#pred_crf = crf.predict_proba(X_valid)[:,1]

#Train and Test Scores
csvc_Tr_Score = round(csvc.score(X_train, y_train)*100, 2)
csvc_Tt_Score = round(csvc.score(X_valid, y_valid)*100, 2)
print('train set score: {:.2f}'.format(csvc_Tr_Score))
print('test set score: {:.2f}'.format(csvc_Tt_Score))






print()
csvc_sc = round(accuracy_score(y_valid, csvc_predict)*100, 2)
print("Accuracy Score: {}%".format(csvc_sc))
print()
PS_csvc = round(precision_score(y_valid, csvc_predict)*100, 2)
print("Precision Score: {}%".format(PS_csvc))
print()
RS_csvc = round(recall_score(y_valid, csvc_predict)*100, 2)
print("Recall Score: {}%".format(RS_csvc))
print()
FS_csvc = round(f1_score(y_valid, csvc_predict)*100, 2)
print("F1 Score: {}%".format(FS_csvc))
print()
fpr3, tpr3, threshold = roc_curve(y_valid, pred_csvc)
roc_auc = metrics.auc(fpr3, tpr3)
csvcA = round(metrics.auc(fpr3, tpr3)*100, 2)
print(f'AUC score is:', csvcA,'%')
print()
confusion = confusion_matrix(y_valid, csvc_predict)
#print("Confusion matrix:\n{}".format(confusion))
plot_metric(confusion, "Support Vector Machine")
No alt text provided for this image
Confusion Matrix
Support Vector Machine Classifier classified 810, about 42%, correctly [True Negative], which is the correct prediction of customers who did not churn and failed to classify 175, about 9%, correctly [False Positive], which is the incorrect prediction of customers who did not churn.
Support Vector Machine Classifier classified 709, about 36%, correctly [True Positive], which is the correct prediction of customers who churn and failed to classify 255, about 13%, correctly [False Negative], which is the incorrect prediction of customers who did churn.
K- Nearest Neighbors
cknn = KNeighborsClassifier(n_neighbors=7)


# Fit 'LPknn' to the training set
cknn.fit(X_train, y_train)


# Predict Output
cknn_predict = cknn.predict(X_valid)
pred_cknn = cknn.predict_proba(X_valid)[:,1]

#Train and Test Scores
cknn_Tr_Score = round(cknn.score(X_train, y_train)*100, 2)
cknn_Tt_Score = round(cknn.score(X_valid, y_valid)*100, 2)
print('train set score: {:.2f}'.format(cknn_Tr_Score))
print('test set score: {:.2f}'.format(cknn_Tt_Score))






print()
cknn_sc = round(accuracy_score(y_valid, cknn_predict)*100, 2)
print("Accuracy Score: {}%".format(cknn_sc))
print()
PS_cknn = round(precision_score(y_valid, cknn_predict)*100, 2)
print("Precision Score: {}%".format(PS_cknn))
print()
RS_cknn = round(recall_score(y_valid, cknn_predict)*100, 2)
print("Recall Score: {}%".format(RS_cknn))
print()
FS_cknn = round(f1_score(y_valid, cknn_predict)*100, 2)
print("F1 Score: {}%".format(FS_cknn))
print()
fpr4, tpr4, threshold = roc_curve(y_valid, pred_cknn)
roc_auc = metrics.auc(fpr4, tpr4)
cknnA = round(metrics.auc(fpr4, tpr4)*100, 2)
print(f'AUC score is:', cknnA,'%')
print()
confusion = confusion_matrix(y_valid, cknn_predict)
#print("Confusion matrix:\n{}".format(confusion))
plot_metric(confusion, "K- Nearest Neighbors")
No alt text provided for this image
Confusion Matrix
K- Nearest Neighbors Classifier classified 629, about 32%, correctly [True Negative], which is the correct prediction of customers who did not churn and failed to classify 356, about 18%, correctly [False Positive], which is the incorrect prediction of customers who did not churn.
K- Nearest Neighbors Classifier classified 789, about 40%, correctly [True Positive], which is the correct prediction of customers who churn and failed to classify 175, about 9%, correctly [False Negative]**, which is the incorrect prediction of customers who did churn.
Extra Trees Classifier
crc = ExtraTreesClassifier()


# Fit 'LPknn' to the training set
crc.fit(X_train, y_train)


# Predict Output
crc_predict = crc.predict(X_valid)
pred_crc = crc.predict_proba(X_valid)[:,1]

#Train and Test Scores
crc_Tr_Score = round(crc.score(X_train, y_train)*100, 2)
crc_Tt_Score = round(crc.score(X_valid, y_valid)*100, 2)
print('train set score: {:.2f}'.format(crc_Tr_Score))
print('test set score: {:.2f}'.format(crc_Tt_Score))






print()
crc_sc = round(accuracy_score(y_valid, crc_predict)*100, 2)
print("Accuracy Score: {}%".format(crc_sc))
print()
PS_crc = round(precision_score(y_valid, crc_predict)*100, 2)
print("Precision Score: {}%".format(PS_crc))
print()
RS_crc = round(recall_score(y_valid, crc_predict)*100, 2)
print("Recall Score: {}%".format(RS_crc))
print()
FS_crc = round(f1_score(y_valid, crc_predict)*100, 2)
print("F1 Score: {}%".format(FS_crc))
print()
fpr5, tpr5, threshold = roc_curve(y_valid, pred_crc)
roc_auc = metrics.auc(fpr5, tpr5)
crcA = round(metrics.auc(fpr5, tpr5)*100, 2)
print(f'AUC score is:', crcA,'%')
print()
confusion = confusion_matrix(y_valid, crc_predict)
#print("Confusion matrix:\n{}".format(confusion))
plot_metric(confusion, "Extra Trees Classifier")
No alt text provided for this image
Confusion Matrix
Extra Trees Classifier classified 826, about 42%, correctly [True Negative], which is the correct prediction of customers who did not churn and failed to classify 159, about 8%, correctly [False Positive], which is the incorrect prediction of customers who did not churn.
Extra Trees Classifier classified 783, about 40%, correctly [True Positive], which is the correct prediction of customers who churn and failed to classify 181, about 9%**, correctly [False Negative], which is the incorrect prediction of customers who did churn.
Histogram-based Gradient Boosting Classification Tree
LPdt = HistGradientBoostingClassifier()

#Fit 'LPdt' to the training set
LPdt.fit(X_train, y_train)


# Predict Output
dy_predict = LPdt.predict(X_valid)
pred_LPdt = LPdt.predict_proba(X_valid)[:,1]

#Train and Test Scores
LPdt_Tr_Score = round(LPdt.score(X_train, y_train)*100, 2)
LPdt_Tt_Score = round(LPdt.score(X_valid, y_valid)*100, 2)
print('train set score: {:.2f}'.format(LPdt_Tr_Score))
print('test set score: {:.2f}'.format(LPdt_Tt_Score))




print()
LPdt_sc = round(accuracy_score(y_valid, dy_predict)*100, 2)
print("Accuracy Score: {}%".format(LPdt_sc))
print()
PS_dt = round(precision_score(y_valid, dy_predict)*100, 2)
print("Precision Score: {}%".format(PS_dt))
print()
RS_dt = round(recall_score(y_valid, dy_predict)*100, 2)
print("Recall Score: {}%".format(RS_dt))
print()
FS_dt = round(f1_score(y_valid, dy_predict)*100, 2)
print("F1 Score: {}%".format(FS_dt))
print()
fpr6, tpr6, threshold = roc_curve(y_valid, pred_LPdt)
roc_auc = metrics.auc(fpr6, tpr6)
LPhgbA = round(metrics.auc(fpr6, tpr6)*100, 2)
print(f'AUC score is:', LPhgbA,'%')
print()
confusion = confusion_matrix(y_valid, dy_predict)
#print("Confusion matrix:\n{}".format(confusion))
print()
plot_metric(confusion, "HistGradientBoosting 
No alt text provided for this image
Confusion Matrix
Histogram-based Gradient Boosting Classifier classified 814, about 42%, correctly [True Negative], which is the correct prediction of customers who did not churn and failed to classify 171, about 9%, correctly [False Positive], which is the incorrect prediction of customers who did not churn.
Histogram-based Gradient Boosting Classifier classified 801, about 41%, correctly [True Positive], which is the correct prediction of customers who churn and failed to classify 163 about 8%, correctly [False Negative], which is the incorrect prediction of customers who did churn.
AdaBoostClassifier
LPrf = AdaBoostClassifier(n_estimators=100, random_state=0)

#Fit 'LPrf' to the training set
LPrf.fit(X_train, y_train)


# Predict Output
ry_predict = LPrf.predict(X_valid)
pred_LPrf = LPrf.predict_proba(X_valid)[:,1]

#Train and Test Scores
LPrf_Tr_Score = round(LPrf.score(X_train, y_train)*100, 2)
LPrf_Tt_Score = round(LPrf.score(X_valid, y_valid)*100, 2)
print('train set score: {:.2f}'.format(LPrf_Tr_Score))
print('test set score: {:.2f}'.format(LPrf_Tt_Score))




print()
LPrf_sc = round(accuracy_score(y_valid, ry_predict)*100, 2)
print("Accuracy Score: {}%".format(LPrf_sc))
print()
PS_rf = round(precision_score(y_valid, ry_predict)*100, 2)
print("Precision Score: {}%".format(PS_rf))
print()
RS_rf = round(recall_score(y_valid, ry_predict)*100, 2)
print("Recall Score: {}%".format(RS_rf))
print()
FS_rf = round(f1_score(y_valid, ry_predict)*100, 2)
print("F1 Score: {}%".format(FS_rf))
print()
fpr7, tpr7, threshold = roc_curve(y_valid, pred_LPrf)
roc_auc = metrics.auc(fpr7, tpr7)
LPabtA = round(metrics.auc(fpr7, tpr7)*100, 2)
print(f'AUC score is:', LPabtA,'%')
print()
confusion = confusion_matrix(y_valid, ry_predict)
#print("Confusion matrix:\n{}".format(confusion))
print()
plot_metric(confusion, "AdaBoost")
No alt text provided for this image
Confusion Matrix
AdaBoost Classifier classified 765, about 39%, correctly [True Negative], which is the correct prediction of customers who did not churn and failed to classify 220 about 11%, correctly **[False Positive], which is the incorrect prediction of customers who did not churn.
AdaBoost Classifier classified 831, about 43%, correctly [True Positive], which is the correct prediction of customers who churn, and failed to classify 133 about 7%, correctly [False Negative], which is the incorrect prediction of customers who did churn.
GradientBoostingClassifier
LPgbm = GradientBoostingClassifier()

#Fit 'LPsvm' to the training set
LPgbm.fit(X_train, y_train)


# Predict Output
gy_predict = LPgbm.predict(X_valid)
pred_LPgbm = LPgbm.predict_proba(X_valid)[:,1]



#Train and Test Scores
LPgbm_Tr_Score = round(LPgbm.score(X_train, y_train)*100, 2)
LPgbm_Tt_Score = round(LPgbm.score(X_valid, y_valid)*100, 2)
print('train set score: {:.2f}'.format(LPrf_Tr_Score))
print('test set score: {:.2f}'.format(LPgbm_Tt_Score))




print()
LPgbm_sc = round(accuracy_score(y_valid, gy_predict)*100, 2)
print("Accuracy Score: {}%".format(LPgbm_sc))
print()
PS_gbm = round(precision_score(y_valid, gy_predict)*100, 2)
print("Precision Score: {}%".format(PS_gbm))
print()
RS_gbm = round(recall_score(y_valid, gy_predict)*100, 2)
print("Recall Score: {}%".format(RS_gbm))
print()
FS_gbm = round(f1_score(y_valid, gy_predict)*100, 2)
print("F1 Score: {}%".format(FS_gbm))
print()
fpr8, tpr8, threshold = roc_curve(y_valid, pred_LPgbm)
roc_auc = metrics.auc(fpr8, tpr8)
LPgbmA = round(metrics.auc(fpr8, tpr8)*100, 2)
print(f'AUC score is:', LPgbmA,'%')
print()
confusion = confusion_matrix(y_valid, gy_predict)
#print("Confusion matrix:\n{}".format(confusion))
plot_metric(confusion, "Gradient Boosting")
No alt text provided for this image
Confusion Matrix
Gradient Boosting Classifier classified 785 about 40%, correctly [True Negative], which is the correct prediction of customers who did not churn and failed to classify 200 about 10%, correctly [False Positive], which is the incorrect prediction of customers who did not churn.
Gradient Boosting Classifier classified 821, about 42%, correctly [True Positive], which is the correct prediction of customers who churn, and failed to classify 143 about 7%, correctly [False Negative], which is the incorrect prediction of customers who did churn.
Find the best Model
Evaluation Metric Summary
LPMetric = pd.DataFrame({'Models': ["Logistic Regression"
                                    "RandomForest",
                                    "Support Vector Machine",
                                    "K- Nearest Neighbors",
                                    "Extra Trees Classifier",  
                                    "HistGradientBoosting",
                                    "AdaBoost",
                                    "Gradient Boosting"],
                             'Accuracy Score': [LPlr_sc, crf_sc, csvc_sc, cknn_sc, crc_sc, LPdt_sc, LPrf_sc, LPgbm_sc],
                             'Precision Score': [PS_lr, PS_crf, PS_csvc, PS_cknn, PS_crc, PS_dt, PS_rf, PS_gbm],
                             'Recall Score': [RS_lr, RS_crf, RS_csvc, RS_cknn, RS_crc, RS_dt, RS_rf, RS_gbm],
                             'F1 Score': [FS_lr, FS_crf, FS_csvc, FS_cknn, FS_crc, FS_dt, FS_rf, FS_gbm]})




LPMetrics = LPMetric.sort_values(by = 'Accuracy Score',ascending = False)
LPMetrics = LPMetrics.set_index('Models')
LPMetrics

plt.figure(figsize = (20,10))
plt.plot(fpr1,tpr1, label = 'Logistic Regression ' 'AUC = %0.2f' %  LPlrA)
plt.plot(fpr2,tpr2, label = 'RandomForest ' 'AUC = %0.2f' % crfA)
plt.plot(fpr3,tpr3, label = 'Support Vector Machine ' 'AUC = %0.2f' % csvcA)
plt.plot(fpr4,tpr4, label = 'K- Nearest Neighbors ' 'AUC = %0.2f' % cknnA)
plt.plot(fpr5,tpr5, label = 'Extra Trees Classifier ' 'AUC = %0.2f' % crcA)
plt.plot(fpr6,tpr6, label ='HistGradientBoosting ' 'AUC = %0.2f' % LPhgbA)
plt.plot(fpr7,tpr7, label = 'AdaBoost Classifier ' 'AUC = %0.2f' % LPabtA)
plt.plot(fpr8,tpr8, label = 'Gradient Boosting ' 'AUC = %0.2f' % LPgbmA)
plt.legend(loc = 'best')
plt.xlabel('FPR', fontsize = 15)
plt.ylabel('TPR', fontsize = 15)
plt.title('RECIEVER OPERATING CHARACTERISTICS', fontsize = 20)
plt.show()
No alt text provided for this image
Feature Importance
importance = pd.DataFrame({'feature':X.columns,'importance':np.round(crf.feature_importances_,3)})
importance = importance.sort_values('importance',ascending=False).set_index('feature')
importance.head(30)
Importance
No alt text provided for this image
Dropping insignificant features
X.drop('PhoneService', axis = 1, inplace = True)
X.drop('SeniorCitizen', axis = 1, inplace = True)
X.drop('StreamingMovies', axis = 1, inplace = True)
X.drop('StreamingTV', axis = 1, inplace = True)
X.head()

#from sklearn.ensemble import
model = RandomForestClassifier()




# define grid search
grid = {
    'n_estimators': [100, 200, 300],  # try different number of trees
    'max_depth': [None, 5, 10],       # try different maximum depth values
    'max_features': ['sqrt', 'log2'], # try different options for the number of features to consider
    'criterion': ['gini', 'entropy']  # try different quality measures
}
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


LPdt = RandomForestClassifier(**grid_result.best_params_)

#Fit 'LPdt' to the training set
LPdt.fit(X_trainval, y_trainval)


# Predict Output
dy_predict = LPdt.predict(X_test)
pred_LPdt = LPdt.predict_proba(X_test)[:,1]

#Train and Test Scores
LPdt_Tr_Score = round(LPdt.score(X_trainval, y_trainval)*100, 2)
LPdt_Tt_Score = round(LPdt.score(X_test, y_test)*100, 2)
print('train set score: {:.2f}'.format(LPdt_Tr_Score))
print('test set score: {:.2f}'.format(LPdt_Tt_Score)) 
Results
Result: train set score: 99.85
test set score: 84.26
