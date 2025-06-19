# student-performance-analysis

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter('ignore')

df = pd.read_csv(r"C:\Users\sahur\Downloads\DATASET OF DATA SCIENCE SRK SIR\StudentPerformanceFactors.csv")

# Data understanding
df
df.shape
df.size
df.info()
df.columns
df.dtypes

continuous = ['Hours_Studied', 'Attendance', 'Sleep_Hours','Previous_Scores'
,'Exam_Score']
d_count = ['Tutoring_Sessions','Physical_Activity']
d_categorical = ['Parental_Involvement','Access_to_Resources', 'Extracurricular_Activities',
                 'Motivation_Level', 'Internet_Access','Family_Income', 'Teacher_Quality',
                 'School_Type','Peer_Influence','Learning_Disabilities','Parental_Education_Level',
                 'Distance_from_Home', 'Gender']

# step-3: Exploratoty Data Analysis

df[continuous].describe()
df[d_count].describe()

# Unique and Value_counts
df['Parental_Involvement'].unique()
df['Parental_Involvement'].value_counts()
df['Access_to_Resources'].unique()
df['Access_to_Resources'].value_counts()
df['Extracurricular_Activities'].unique()
df['Extracurricular_Activities'].value_counts()
df['Motivation_Level'].unique()
df['Motivation_Level'].value_counts()
df['Internet_Access'].unique()
df['Internet_Access'].value_counts()
df['Family_Income'].unique()
df['Family_Income'].value_counts()
df['Teacher_Quality'].unique()
df['Teacher_Quality'].value_counts()
df['School_Type'].unique()
df['School_Type'].value_counts()
df['Peer_Influence'].unique()
df['Peer_Influence'].value_counts()
df['Learning_Disabilities'].unique()
df['Learning_Disabilities'].value_counts()
df['Parental_Education_Level'].unique()
df['Parental_Education_Level'].value_counts()
df['Distance_from_Home'].unique()
df['Distance_from_Home'].value_counts()
df['Gender'].unique()
df['Gender'].value_counts()

# Plots

# continuous_variables 
plt.figure(figsize = (15,23))

plt.suptitle('Histogram of discrete_categorical variables')
# Histogram of all continuous variables
plt.subplot(2,3,1)
plt.hist(df['Hours_Studied'],edgecolor = 'black')
plt.title('Distribution of Hours Studied')
plt.xlabel('Hours Studied')
plt.ylabel('Frequecy')

plt.subplot(2,3,2)
plt.hist(df['Attendance'],edgecolor = 'black')
plt.title('Distribution of Attendance of students')
plt.xlabel('Attendance')
plt.ylabel('Frequency')

plt.subplot(2,3,3)
plt.hist(df['Sleep_Hours'],edgecolor = 'black')
plt.title('Distribution of sleep_hours of students')
plt.xlabel('sleep_hours')
plt.ylabel('Frequency')

plt.subplot(2,3,4)
plt.hist(df['Previous_Scores'],edgecolor = 'black')
plt.title('Distribution of score of students in previous exams')
plt.xlabel('Previous_scores')
plt.ylabel('frequency')

plt.subplot(2,3,5)
plt.hist(df['Exam_Score'],edgecolor = 'black')
plt.title('Disribution of exam Scores after study_hours')
plt.xlabel('Exam_score')
plt.ylabel('frequency')
plt.show()

# Box-plots 
plt.figure(figsize = (12,15))
plt.suptitle('box-plot of continuous variables')

# Boxplot of all continous variables
plt.subplot(2,3,1)
plt.boxplot(df['Hours_Studied'])
plt.title('Hours Studied')

plt.subplot(2,3,2)
sns.boxplot(df['Attendance'])
plt.title('Attendance of students')

plt.subplot(2,3,3)
plt.boxplot(df['Sleep_Hours'])
plt.title('sleep_hours of students')

plt.subplot(2,3,4)
plt.boxplot(df['Previous_Scores'])
plt.title('Scores of students in previous exams')

plt.subplot(2,3,4)
sns.boxplot(df['Exam_Score'])
plt.title('Exam Scores after study_hours')
plt.show()



# by using for loop plot the box plot
 
# Box-plots 
plt.figure(figsize = (12,15))
plt.suptitle('box-plot of continuous variables')
j=1
for i in continuous:
# Boxplot of all continous variables
    
    plt.subplot(2,3,j)
    plt.boxplot(df[i])
    plt.title(i)
    j+=1

# plt.subplot(2,3,2)
# sns.boxplot(df['Attendance'])
# plt.title('Attendance of students')

# plt.subplot(2,3,3)
# plt.boxplot(df['Sleep_Hours'])
# plt.title('sleep_hours of students')

# plt.subplot(2,3,4)
# plt.boxplot(df['Previous_Scores'])
# plt.title('Scores of students in previous exams')

# plt.subplot(2,3,4)
# sns.boxplot(df['Exam_Score'])
# plt.title('Exam Scores after study_hours')
# plt.show()



# discrete_count variables
d_count= ['Tutoring_Sessions','Physical_Activity']

plt.figure(figsize= (10,7))

plt.suptitle('count -plot of descrete_count variables')
plt.subplot(1,2,1)
patches =plt.bar(df['Tutoring_Sessions'].value_counts().index,df['Tutoring_Sessions'].value_counts())
plt.bar_label(patches)
plt.title('count of tutoring sessions per month')
plt.xlabel('Tutoring sessions ')
plt.ylabel('frequency')


plt.subplot(1,2,2)
patches =plt.bar(df['Physical_Activity'].value_counts().index,df['Physical_Activity'].value_counts())
plt.bar_label(patches)
plt.title('count of physical_activity per week')
plt.xlabel('Physical_Activity')
plt.ylabel('frequency')
plt.show()



# pie - chart 
df['Physical_Activity'].unique()
array([3, 4, 2, 1, 5, 0, 6], dtype=int64)
plt.figure(figsize=(10,20))
plt.subplot(1,2,1)
plt.pie(df['Tutoring_Sessions'].value_counts(),
        labels = df['Tutoring_Sessions'].unique(),
        autopct ="%0.1f%%",
       explode=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
plt.title('Pie-chart of tutoring sessions per month')

plt.subplot(1,2,2)
plt.pie(df['Physical_Activity'].value_counts(),
       labels = df['Physical_Activity'].unique(),
       autopct = '%0.1f%%',
       explode = [0.1,0.1,0.1,0.1,0.1,0.1,0.1])
plt.title('pie-chart of physical_activity per week')
plt.show()




plt.figure(figsize = (12,10))
plt.subplot(2,3,1)
patches=plt.bar(df['Parental_Involvement'].value_counts().index,
                df['Parental_Involvement'].value_counts())
plt.bar_label(patches)
plt.title('count of parental_Involvement')
plt.xlabel('parental_Involvement')
plt.ylabel('frequency')

plt.subplot(2,3,2)
patches= plt.bar(df['Access_to_Resources'].value_counts().index,
                df['Access_to_Resources'].value_counts())
plt.bar_label(patches)
plt.title('count of Access_to_Resources')
plt.xlabel('Access_to_resources')
plt.ylabel('Frequency')

plt.subplot(2,3,3)
patches=plt.bar(df['Extracurricular_Activities'].value_counts().index,
               df['Extracurricular_Activities'].value_counts())
plt.bar_label(patches)
plt.title('count of Extracurricular_Activities')
plt.xlabel('Extracurricular_Activities')
plt.ylabel('frequency')

plt.subplot(2,3,4)
patches = plt.bar(df['Motivation_Level'].value_counts().index
                 ,df['Motivation_Level'].value_counts())
plt.bar_label(patches)
plt.title('count of Motivation_level')
plt.xlabel('Motivation_lavel')
plt.ylabel('frequency')

plt.subplot(2,3,5)
patches=plt.bar(df['Internet_Access'].value_counts().index,
               df['Internet_Access'].value_counts())
plt.bar_label(patches)
plt.title('count of Internet_Access')
plt.xlabel('Internet_Access')
plt.ylabel('frequency')

plt.subplot(2,3,6)
patches=plt.bar(df['Family_Income'].value_counts().index,
               df['Family_Income'].value_counts())
plt.bar_label(patches)
plt.title('count of Family_Income')
plt.xlabel('Family_Income')
plt.ylabel('frequency')

plt.show()

# pie - chart 
df['Family_Income'].unique()

plt.figure(figsize=(10,12))

plt.suptitle('pie-chart of discrete_categorical variables')
plt.subplot(3,2,1)
plt.pie(df['Parental_Involvement'].value_counts(),
       labels = df['Parental_Involvement'].unique(),
       autopct = '%0.1f%%',
       explode = [0.1,0.1,0.1])
plt.title('pie-chart of parental_Involvement')

plt.subplot(3,2,2)
plt.pie(df['Access_to_Resources'].value_counts(),
       labels = df['Access_to_Resources'].unique(),
       autopct = '%0.1f%%',
       explode = [0.1,0.1,0.1])
plt.title('pie-chart of Access_to_Resources')

plt.subplot(3,2,3)
plt.pie(df['Extracurricular_Activities'].value_counts(),
       labels = df['Extracurricular_Activities'].unique(),
       autopct = '%0.1f%%',
       explode = [0.1,0.1])
plt.title('Extracurricular_Activities')

plt.subplot(3,2,4)
plt.pie(df['Motivation_Level'].value_counts(),
       labels = df['Motivation_Level'].unique(),
       autopct = '%0.1f%%',
       explode = [0.1,0.1,0.1])
plt.title('pie-chart of Motivation_Level')

plt.subplot(3,2,5)
plt.pie(df['Internet_Access'].value_counts(),
       labels = df['Internet_Access'].unique(),
       autopct = '%0.1f%%',
       explode = [0.1,0.1])
plt.title('pie-chart of Internet_Access')

plt.subplot(3,2,6)
plt.pie(df['Family_Income'].value_counts(),
       labels = df['Family_Income'].unique(),
       autopct = '%0.1f%%',
       explode = [0.1,0.1,0.1])
plt.title('pie-chart of Family_Income')

plt.show()



d_categorical

plt.figure(figsize=(16,16))

plt.suptitle('count plot of discrete_categorical variables')
plt.subplot(3,3,1)
patches = plt.bar(df['Teacher_Quality'].value_counts().index,
                 df['Teacher_Quality'].value_counts())
plt.bar_label(patches)
plt.title('count of Teacher_Quality')
plt.xlabel('Teacher_Quality')
plt.ylabel('frequency')

plt.subplot(3,3,2)
patches = plt.bar(df['School_Type'].value_counts().index,
                 df['School_Type'].value_counts())
plt.bar_label(patches)
plt.title('count of School_Type')
plt.xlabel('School_Type')
plt.ylabel('frequency')

plt.subplot(3,3,3)
patches = plt.bar(df['Peer_Influence'].value_counts().index,
                 df['Peer_Influence'].value_counts())
plt.bar_label(patches)
plt.title('count of Peer_Influence')
plt.xlabel('Peer_Influence')
plt.ylabel('frequency')

plt.subplot(3,3,4)
patches = plt.bar(df['Learning_Disabilities'].value_counts().index,
                 df['Learning_Disabilities'].value_counts())
plt.bar_label(patches)
plt.title('count of Learning_Disabilities')
plt.xlabel('Learning_Disabilities')
plt.ylabel('frequency')

plt.subplot(3,3,5)
patches = plt.bar(df['Parental_Education_Level'].value_counts().index,
                 df['Parental_Education_Level'].value_counts())
plt.bar_label(patches)
plt.title('count of Parental_Education_Level')
plt.xlabel('Parental_Education_Level')
plt.ylabel('frequency')

plt.subplot(3,3,6)
patches = plt.bar(df['Distance_from_Home'].value_counts().index,
                 df['Distance_from_Home'].value_counts())
plt.bar_label(patches)
plt.title('count of Distance_from_Home')
plt.xlabel('Distance_from_Home')
plt.ylabel('frequency')

plt.subplot(3,3,7)
patches = plt.bar(df['Gender'].value_counts().index,
                 df['Gender'].value_counts())
plt.bar_label(patches)
plt.title('count of Gender')
plt.xlabel('Gender')
plt.ylabel('frequency')

plt.show()


# pie_chart 
df['Learning_Disabilities'].unique()
plt.figure(figsize=(10,16))
''' df['Teacher_Quality'],df['Parental_Education_level'],
df['Distance_from_Home']--- missing values are present so pie chart connot be ploted
'''

plt.suptitle('Pie-chart of discrete_categorical variable')

plt.subplot(3,2,1)
plt.pie(df['School_Type'].value_counts(),
       labels = df['School_Type'].unique(),autopct = '%0.1f%%'
       ,explode = [0.1,0.1])
plt.title('pie-chart of school_type')

plt.subplot(3,2,2)
plt.pie(df['Peer_Influence'].value_counts(),
       labels = df['Peer_Influence'].unique(),autopct = '%0.1f%%'
       ,explode = [0.1,0.1,0.1])
plt.title('pie-chart of Peer_Influence')

plt.subplot(3,2,3)
plt.pie(df['Learning_Disabilities'].value_counts(),
       labels = df['Learning_Disabilities'].unique(),autopct = '%0.1f%%'
       ,explode = [0.1,0.1])
plt.title('pie-chart of Learning_Disabilities')

plt.subplot(3,2,4)
plt.pie(df['Gender'].value_counts(),
       labels = df['Gender'].unique(),autopct = '%0.1f%%'
       ,explode = [0.1,0.1])
plt.title('pie-chart of Gender')

plt.show()


# Bivariate plots
# scatter - plots
continuous


plt.figure(figsize=(13,16))

plt.suptitle('scatter plot for each continuous variable to exam_score')

plt.subplot(2,2,1)
plt.scatter(x=df['Hours_Studied'],y=df['Exam_Score'],label = 'Hours_studied & Exam_score')
plt.xlabel('Hours_Studied')
plt.ylabel('Exam_score')
plt.title('Scatter plot of Hours_Studied to exam_score')
plt.legend()

plt.subplot(2,2,2)
plt.scatter(x=df['Attendance'],y=df['Exam_Score'],label = 'Attendance & Exam_score')
plt.xlabel('Attendance')
plt.ylabel('Exam_score')
plt.title('Scatter plot of Attendance to exam_score')
plt.legend()

plt.subplot(2,2,3)
plt.scatter(x=df['Sleep_Hours'],y=df['Exam_Score'],label = 'Sleep_Hours & Exam_score')
plt.xlabel('Sleep_Hours')
plt.ylabel('Exam_score')
plt.title('Scatter plot of Sleep_Hours to exam_score')
plt.legend()

plt.subplot(2,2,4)
plt.scatter(x=df['Previous_Scores'],y=df['Exam_Score'],label = 'Previous_Scores & Exam_score')
plt.xlabel('Previous_Scores')
plt.ylabel('Exam_score')
plt.title('Scatter plot of Hours_Studied to exam_score')
plt.legend()
plt.show()

# correlation
df[continuous].corr()

# Multi-Variate plots
sns.pairplot(df)
plt.show()

#Step-4: Data-Cleaning

#1.Wrong data
#2.Missing Values
#3.Wrong datatypes
#4.Duplicates
#5.outliers

# 1. wrong_data

df['Exam_Score'].replace({101:100},inplace=True)
df['Exam_Score'].unique()

# 2.Missing value
df.isnull().sum()

# calculating percentage of missing values in data set
missing_persent =(df.isnull().sum()/len(df))*100
missing_persent

# as the missing value are less then 5% it is better if we replace them
# as all the missing values are present categorical variable it is better if we replace them to most frequent value.
df['Teacher_Quality'].fillna(df['Teacher_Quality'].mode()[0],inplace = True)

df['Parental_Education_Level'].fillna(df['Parental_Education_Level'].mode()[0],inplace =True)

df['Distance_from_Home'].fillna(df['Distance_from_Home'].mode()[0],inplace = True)

 # 3-wrong datatype
   # No wrong datatype

# 4-Duplicates

df.duplicated().sum()
# no duplicates present

# 5-outliers
sns.boxplot(df[continuous])
plt.show()

q1 = df['Hours_Studied'].quantile(0.25)

q3 = df['Hours_Studied'].quantile(0.75)

IQR = q3 - q1

print('IQR',IQR)

l_bound = q1 - (1.5 * IQR)
u_bound = q3 + (1.5 * IQR)

outliers = df[(df['Hours_Studied']< l_bound) | (df['Hours_Studied'] > u_bound)]

print('Outliers below lower_limit:',len(df[df['Hours_Studied'] < l_bound]))

print('Outliers above upper_limit:',len(df[df['Hours_Studied'] > u_bound]))


# Exam_score
q1 = df['Exam_Score'].quantile(0.25)
q3 = df['Exam_Score'].quantile(.75)

IQR = q3 - q1 

l_bound = q1 - (1.5 * IQR)
u_bound = q3 + (1.5 * IQR)

outliers =df[(df['Exam_Score'] < l_bound) | (df['Exam_Score'] > u_bound)]

print("Outliers below lower limit",len(df[df['Exam_Score'] < l_bound]))

print('Outliers above lower limit',len(df[df['Exam_Score'] > u_bound]))
















                 
