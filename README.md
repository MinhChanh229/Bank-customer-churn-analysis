# Bank-customer-churn-analysis

## Overview:
- Back ground:

Banks today are facing an increasing customer churn rate. Since the cost of acquiring new customers is often much higher than the cost of retaining existing ones, building an early warning system for customer churn can help banks reduce losses and optimize long-term profitability.

- Goal of the Project:

Based on the data, descriptive nalytics to understand the behavior of customers who have chunred  we will find out what causes customers to leave and then come up with appropriate strategies to help the bank.

- Source:

Kaggle: https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn/data

Customer-Churn-Records.csv(837.42 kB)

18 columns and 10.000 rows

- Project Objective:

The primary objective of this notebook is to ETL and build machine learning model to predict combine with SQL to analysis more about outliers.

Buid dashboard to track customer churn rate.
  
## 👣 The First Steps
### 📥 Data import

First, let's import the needed libraries: Pandas, Matplotlib & Seaborn.
```code
# import pandas, matplotlip, seaborn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

### Extract data:

Then load data from csv file.
```code
# load data form csv
df = pd.read_csv('/content/Customer-Churn-Records.csv')
df.head()
```
 <img width="1749" height="235" alt="image" src="https://github.com/user-attachments/assets/9c48a027-f54c-4315-9282-cea3cbcd14b6" />

```code
df.shape()
```
(10000, 18)

```code
# check random
df.sample(10)
```
<img width="1766" height="416" alt="image" src="https://github.com/user-attachments/assets/9e579618-b1ff-4d1d-b3b3-6b0a82dd1846" />

 Great now we can see basic information about this file.

### Transform data:
- Data exploration:

Show the columns of the dataframe and their types.
  ```code
  df.info()
  ```
  <img width="484" height="508" alt="image" src="https://github.com/user-attachments/assets/d0dba80d-cdc7-4d2d-9f8e-e3e03a9738b3" />

Prevailing 4 objects type (surname, geography, gender, card type)
  
- Check missing value:
```code
print(df.isnull().sum())
```
<img width="266" height="395" alt="image" src="https://github.com/user-attachments/assets/552c50c1-0b5d-4584-9c00-1d3828edc097" />

  No data missing and no duplicate.
  When there are no missing values in the dataset, it usually indicates:
  1. Good data collection process
  All fields are fully entered, minimizing errors or omissions.

  2. Higher data quality
  Reduced risk of bias due to missing information.
  Analysis and modeling become more reliable.

  3. Less complex preprocessing required
  No need for imputation techniques or record deletion.

  4. Mandatory data entry by the system
  The system may be designed to require all fields to be filled in before saving the data.

- Fix data type:
```code
df['Surname'] = df['Surname'].astype('string')
df['Geography'] = df['Geography'].astype('string')
df['Gender'] = df['Gender'].astype('string')
df['Card Type'] = df['Card Type'].astype('string')
# check again
df.info()
```
<img width="498" height="504" alt="image" src="https://github.com/user-attachments/assets/2d06ed0a-63ca-4e6f-b454-7da7cc0b3694" />

  Convert data all object type to string type.

- Basic descriptive statistics:

Show a descriptive statistics of the numeric columns.
```code
df.describe()
```
<img width="1762" height="347" alt="image" src="https://github.com/user-attachments/assets/1b5bb0f3-9e45-49c7-891d-79b09d05e8c5" />

  Age: ranges from 18 to 92 years old, with a median of 37
  
  Balance: many customers have a balance of 0 (median = 97,198.54, but 25th percentile = 0).
  
  NumOfProducts: most customers use 1–2 products.
  
  HasCrCard and IsActiveMember are binary variables (0 or 1).
  
  Satisfaction Score: ranges from 1 to 5, with an average of around 3.
  
  Point Earned: ranges from 119 to 1,000, with a relatively wide distribution.

- Customer obvious analysis:

Boxplots to compare distributions by churn: Age, Balance, EstimatedSalary, CreditScore
```code
# Boxplots to compare distributions by churn
for col in ['Age', 'Balance', 'EstimatedSalary', 'CreditScore']:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Exited', y=col, data=df)
    plt.title(f'{col} vs. Exited')
    plt.show()
```
<img width="602" height="459" alt="image" src="https://github.com/user-attachments/assets/8667d363-5105-4868-a45a-8308f76528d6" />
<img width="639" height="448" alt="image" src="https://github.com/user-attachments/assets/6dc5bb98-e9f0-4ac2-8750-8f747e475445" />
<img width="694" height="481" alt="image" src="https://github.com/user-attachments/assets/707423b0-ac96-451f-a318-7de66c88bb27" />
<img width="653" height="436" alt="image" src="https://github.com/user-attachments/assets/9c877282-2401-4bc0-9f63-0fb1246d6692" />

### Remark:
- Boxplot age vs exited: Customers who exited (Exited = 1) tend to be older, with a higher median age compared to customers who stayed (Exited = 0). Additionally, the age range for exited customers is wider, indicating age may be a contributing factor to churn.

- Boxplot balance vs exited: Customers who exited (Exited = 1) generally have a slightly higher median balance compared to those who stayed (Exited = 0). However, both groups show a wide spread of balances, indicating that balance alone may not be a strong predictor of churn.

- Boxplot estimatedSalary vs exited: The distribution of estimated salaries is very similar for both customers who exited (Exited = 1) and those who stayed (Exited = 0). This suggests that estimated salary may not have a significant impact on customer churn in this dataset.

- Boxplot creditscore vs exited: The credit score distributions between the exit group (Exited = 1) and the stay group (Exited = 0) are quite similar, but the exit group has more low credit scores (~350–400) which appear as outliers. The medians of the two groups are almost equal (~650 points), suggesting that CreditScore may not be a strong determinant of customer churn. However, the bottom of the boxplot for the Exited = 1 group is lower, meaning that more customers leaving are in the very low credit score group.

- Outliers analysis in boxplot age:

<img width="1039" height="199" alt="image" src="https://github.com/user-attachments/assets/5757f816-79a3-458e-add5-a29b6539d641" />

<img width="1036" height="198" alt="image" src="https://github.com/user-attachments/assets/8e0c0f9e-7382-4dd3-8bb4-b4b965edad90" />



Violin to see more about Balance vs Churn
```code
# Violin to see more about Balance vs Churn
sns.violinplot(x='Exited', y='Balance', data=df)
plt.title('Balance vs Churn')
plt.xlabel('Exited')
plt.ylabel('Balance')
plt.show()
```
<img width="703" height="529" alt="image" src="https://github.com/user-attachments/assets/7f1d0614-33f4-4f63-8780-713a82b0c298" />

### Remark:
#### Analyze the relationship between Credit Score, Balance and Churn
1. Balance vs Churn:
- The balance distribution between  the two group is quite similar, concentrated in two main levels:
--  Balance ~ 0: Large proportion of both groups, probably less active customers.
--  Balance ~ 100000 - 150000: Is the popular balance group.
- The median of the group that left is sightly higher than the group that stayed. Customers with higher balances may be looking for another bank with better interest rates/offers.

2. Credit score and Churn:
- (Exited = 1) tend to have lower Credit Score than the group that stays.
- This suggests that customers with weak credit histories are more likely to leave banks, possibly because they face financial difficulties or do not receive good credit offers.

3. Combining insight:
- Low Credit score is a big risk factor for churn.
- High balance combined with good credit score does not necessarily retain customers, due to competition from other banks.
- Balance = 0 group may be "hibernating" customers or customers who opened an account but did not use it, requiring a reactivation campaign.

4. Suggested action:
- Retaining customers with low credit scores: offering tailored credit products and personal financial support.
- Maintain high balance customers: create interest rate incentives, VIP service packages to avoid losing customers to competitors.
- Activate balance = 0 group: send incentive offers to encourage trading or depositing funds into the account.

- Heatmap
```code
#heatmap
# Select only numeric columns (integers and floats)
num_cols = df.select_dtypes(include=['int64', 'float64'])

# Print the list of numerical columns
#print("Numeric columns selected for correlation analysis:")
#print(num_cols.columns.tolist())

# Calculate Pearson correlation matrix
corr_matrix = num_cols.corr()

# Set visualization style
sns.set(style="white")

# Plot heatmap
plt.figure(figsize=(20, 10))
sns.heatmap(corr_matrix,
            annot=True,
            fmt=".2f",  # format decimal places
            cmap="coolwarm",
            linewidths=0.5,
            cbar_kws={'label': 'Correlation Coefficient'})

plt.title("Correlation Matrix of Numerical Features", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Filter correlation matrix for strong correlations (> 0.7 or < -0.7)
strong_corr = corr_matrix[((corr_matrix > 0.7) | (corr_matrix < -0.7)) & (corr_matrix != 1.0)]
```
<img width="1389" height="749" alt="image" src="https://github.com/user-attachments/assets/3282854d-0987-46af-9142-048296b71cc8" />


  




