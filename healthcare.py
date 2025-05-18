import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(class_weight='balanced')

df = pd.read_csv(r'C:\project_intern\noshowappointment.csv')
df.head()
df.info()
df.describe()
df['No-show'].value_counts()
# Drop irrelevant columns
df.drop(['PatientId', 'AppointmentID', 'Neighbourhood'], axis=1, inplace=True)

# Convert dates
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

# Create new features
df['WaitingTime'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
df['DayOfWeek'] = df['AppointmentDay'].dt.day_name()

# Clean column names
df.columns = df.columns.str.replace('-', '_')

# Map target variable
df['No_show'] = df['No_show'].map({'Yes': 1, 'No': 0})
# No-show rate by weekday
sns.countplot(x='DayOfWeek', hue='No_show', data=df)
plt.title('No-show by Day of Week')
plt.xticks(rotation=45)
plt.show()

# SMS Received vs No-show
sns.countplot(x='SMS_received', hue='No_show', data=df)
plt.title('Effect of SMS Reminder')
plt.show()

# Waiting time impact
sns.boxplot(x='No_show', y='WaitingTime', data=df)
plt.title('Waiting Time vs No-show')
plt.show()

features = ['Age', 'SMS_received', 'WaitingTime']
X = df[features]
y = df['No_show']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=1))
# Step 1: Make a copy of test data
X_test_copy = X_test.copy()

# Step 2: Add actual and predicted labels
X_test_copy['Actual'] = y_test
X_test_copy['Predicted'] = y_pred

# Step 3: Save to a CSV file (Excel-readable)
X_test_copy.to_csv(r'C:\project_intern\predictions.csv', index=False)




