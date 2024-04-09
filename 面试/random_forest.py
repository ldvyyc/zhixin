import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = 'dataset-20230501.csv'
df = pd.read_csv(file_path, header=None)
df.fillna("ffill", inplace=True)

# Separate features and target
X = df.iloc[:, 1:]  # All columns except the first one are features
y = df.iloc[:, 0]   # The first column is the target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model on the training data
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')

# print(y_pred)

# 使用交叉检验计算正确率
cv_accuracy = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy').mean()
print(f'Cross-validation Accuracy: {cv_accuracy}')

r_squared = rf_model.score(X_test, y_test)
print(f'R-squared: {r_squared}')

sharpe_ratio = y_pred.mean() / y_pred.std()
print(f'Sharpe Ratio: {sharpe_ratio}')



# 将连续的收益率转换为二分类问题（例如，正收益率为1，负收益率为0）
y_test_binary = (y_test > 0).astype(int)
y_pred_binary = (y_pred > 0).astype(int)

prediction_accuracy = accuracy_score(y_test_binary, y_pred_binary)
print(f'Prediction Accuracy: {prediction_accuracy}')
