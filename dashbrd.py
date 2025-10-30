# 911callsdatacapstone.py
# ----------------------------------------
# Import necessary libraries
# ----------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------
# Load the dataset
# ----------------------------------------
df = pd.read_csv('911.csv')

# ----------------------------------------
# Convert timeStamp to datetime
# ----------------------------------------
df['timeStamp'] = pd.to_datetime(df['timeStamp'])

# ----------------------------------------
# Create additional time-based columns
# ----------------------------------------
df['Month'] = df['timeStamp'].dt.month
df['Day of Week'] = df['timeStamp'].dt.day_name()
df['Hour'] = df['timeStamp'].dt.hour
df['Date'] = df['timeStamp'].dt.date

# ----------------------------------------
# Extract Reason from 'title'
# ----------------------------------------
df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])

# ----------------------------------------
# Create a figure with subplots (4 rows × 3 columns)
# ----------------------------------------
fig, axes = plt.subplots(4, 3, figsize=(20, 20))
fig.suptitle('911 Calls Data Analysis Dashboard', fontsize=18, fontweight='bold')

# ----------------------------------------
# 1️⃣ Correlation Matrix Heatmap
# ----------------------------------------
corr = df.select_dtypes(include=['number']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=axes[0, 0])
axes[0, 0].set_title('1. Correlation Matrix Heatmap')

# ----------------------------------------
# 2️⃣ Distribution of Call Types
# ----------------------------------------
sns.countplot(data=df, x='title', order=df['title'].value_counts().index[:10], ax=axes[0, 1])
axes[0, 1].set_title('2. Distribution of Top 10 Call Types')
axes[0, 1].tick_params(axis='x', rotation=90)

# ----------------------------------------
# 3️⃣ Top 10 Townships by Number of Calls
# ----------------------------------------
sns.countplot(data=df, x='twp', order=df['twp'].value_counts().index[:10], ax=axes[0, 2])
axes[0, 2].set_title('3. Top 10 Townships by Number of Calls')
axes[0, 2].tick_params(axis='x', rotation=90)

# ----------------------------------------
# 4️⃣ Number of Calls Over Time (Resampled by Day)
# ----------------------------------------
df_time = df.set_index('timeStamp')
df_time.resample('D').size().plot(ax=axes[1, 0])
axes[1, 0].set_title('4. Number of Calls Over Time')
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Number of Calls')

# ----------------------------------------
# 5️⃣ Distribution of Calls by Reason
# ----------------------------------------
sns.countplot(data=df, x='Reason', order=df['Reason'].value_counts().index, ax=axes[1, 1])
axes[1, 1].set_title('5. Distribution of Calls by Reason')

# ----------------------------------------
# 6️⃣ Number of Calls by Day of Week (with Reason)
# ----------------------------------------
sns.countplot(data=df, x='Day of Week', hue='Reason',
              order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
              ax=axes[1, 2])
axes[1, 2].set_title('6. Number of Calls by Day of Week')
axes[1, 2].legend(loc='upper right')

# ----------------------------------------
# 7️⃣ Number of Calls by Month
# ----------------------------------------
sns.countplot(data=df, x='Month', order=range(1, 13), ax=axes[2, 0])
axes[2, 0].set_title('7. Number of Calls by Month')

# ----------------------------------------
# 8️⃣ Number of Calls Over Time (Daily)
# ----------------------------------------
df.groupby('Date').size().plot(ax=axes[2, 1])
axes[2, 1].set_title('8. Number of Calls Over Time (Daily)')
axes[2, 1].set_xlabel('Date')
axes[2, 1].set_ylabel('Number of Calls')

# ----------------------------------------
# 9️⃣ Linear Fit on Number of Calls per Month
# ----------------------------------------
by_month = df.groupby('Month').size().reset_index(name='Count')
sns.lineplot(x='Month', y='Count', data=by_month, ax=axes[2, 2], marker='o')
axes[2, 2].set_title('9. Linear Fit on Number of Calls per Month')

# ----------------------------------------
# 🔟 Number of 911 Calls per Month by Reason
# ----------------------------------------
calls_per_month = df.groupby(['Month', 'Reason']).size().reset_index(name='Count')
sns.lineplot(x='Month', y='Count', hue='Reason', data=calls_per_month, ax=axes[3, 0])
axes[3, 0].set_title('10. Number of 911 Calls per Month by Reason')

# ----------------------------------------
# 11️⃣ Heatmap of Calls by Day of Week and Hour
# ----------------------------------------
dayHour = df.groupby(by=['Day of Week', 'Hour']).count()['Reason'].unstack()
sns.heatmap(dayHour, cmap='viridis', ax=axes[3, 1])
axes[3, 1].set_title('11. Heatmap of Calls by Day of Week and Hour')

# ----------------------------------------
# Hide the last empty subplot
# ----------------------------------------
axes[3, 2].axis('off')

# ----------------------------------------
# Adjust layout
# ----------------------------------------
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

