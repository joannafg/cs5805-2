# type: ignore
# flake8: noqa
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Filter out rows where any of the feature columns or 'SO2' is NaN
filtered_data = sample_data.dropna(subset=['TEMP', 'PRES', 'DEWP', 'SO2'])

# Standardizing the relevant columns of the filtered data
scaler = StandardScaler()
scaled_columns = scaler.fit_transform(filtered_data[['TEMP', 'PRES', 'DEWP']])

# Converting scaled data back to a DataFrame
scaled_df = pd.DataFrame(scaled_columns, columns=['TEMP', 'PRES', 'DEWP'])

# Defining features (X) and target variable (y)
X = scaled_df
y = filtered_data['SO2']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Creating and fitting the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Creating the linear regression model
model = LinearRegression()

# Fitting the model to the training data
model.fit(X_train, y_train)


