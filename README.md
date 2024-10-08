Here's a README.md for the codebase:

# Fraud Detection Model

This project implements a fraud detection model using machine learning techniques. The main components are data preprocessing, feature engineering, model training, and evaluation.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Engineering](#feature-engineering)
5. [Model Training](#model-training)
6. [Evaluation](#evaluation)

## Installation

To set up the project, make sure you have Python installed. Then, install the required dependencies:

```
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Usage

To run the fraud detection model, execute the `main()` function in the script:

```python
main()
```

This function performs the entire pipeline from data preprocessing to model training and evaluation.

## Data Preprocessing

The data preprocessing steps include:

1. Loading and cleaning the data
2. Handling missing values
3. Encoding categorical variables
4. Normalizing numerical features

Key functions:


```40:56:src/notebook/Bati_bank_task.ipynb
in[3]: def data_summary(data):
    # logging('Summarizing the data ...')
    # data summary
    print(data.describe())
    # data info
    logging.info('Data info')
    print(data.info())
    logging.info('data types')
    print(data.dtypes)
    logging.info('top 5 rows')
    print(data.head())
    logging.info('Columns')
    print(data.columns)
    logging.info('data shapes')
    print(data.shape)
    logging.info('data null values')
    print(data.isnull().sum())
```


## Feature Engineering

Feature engineering involves creating new features and transforming existing ones to improve model performance. Key functions include:


```106:155:src/notebook/Bati_bank_task.ipynb
in[8]: def aggregate_features(data):
    # group by account id
    logger.info('Aggregating data ...')
    customer_group = data.groupby('CustomerId')
    data['TotalTransactionAmount'] = customer_group['Amount'].transform('sum')
    data['AverageTransactionAmount'] = customer_group['Amount'].transform('mean')
    data['TransactionCount'] = customer_group['TransactionId'].transform('count')
    data['TransactionAmountStd'] = customer_group['Amount'].transform('std')
    return data

in[9]: def extract_features(data):
    
    data['TransactionHour'] = data['TransactionStartTime'].dt.hour
    data['TransactionDay'] = data['TransactionStartTime'].dt.day
    data['TransactionMonth'] = data['TransactionStartTime'].dt.month
    data['TransactionYear'] = data['TransactionStartTime'].dt.year
    # print(data.columns)
    return data

in[10]: def encode_data(data):
    logger.info('Encoding data ...')
    # encode categorical columns
    logger.info('One hot encoding data ...')
    data = pd.get_dummies(data, columns=['ProductCategory', 'ChannelId'], drop_first=True)

    logger.info('Label encoding data ...')
    le = LabelEncoder()
    data['ProviderId'] = le.fit_transform(data['ProviderId'])
    data['ProductId'] = le.fit_transform(data['ProductId'])
    data['PricingStrategy'] = le.fit_transform(data['PricingStrategy'])
    return data

in[11]: def imputation(data):
    logger.info('Imputing missing values ...')
    imputer = SimpleImputer(strategy='mean')
    # impute missing values
    data = imputer.fit_transform(data)

    imputed_df = pd.DataFrame(data, columns=data.columns)
    return imputed_df

in[12]: def normalize_data(data):
    logger.info('Normalizing data ...')
    # normalize data
    scaler = StandardScaler()
    data[['Amount', 'Value', 'TotalTransactionAmount', 'AverageTransactionAmount']] = scaler.fit_transform(data[['Amount', 'Value', 'TotalTransactionAmount', 'AverageTransactionAmount']])

    # Or use MinMaxScaler for normalization
    min_max_scaler = MinMaxScaler()
    data[['Amount', 'Value', 'TotalTransactionAmount', 'AverageTransactionAmount']] = min_max_scaler.fit_transform(data[['Amount', 'Value', 'TotalTransactionAmount', 'AverageTransactionAmount']])
```


## Model Training

The project trains two models: Random Forest Classifier and Logistic Regression. The model with better accuracy is selected for final use.


```218:273:src/notebook/Bati_bank_task.ipynb
in[17]: def train_model(data):
   logger.info('Training model ...')
   # split data into training and testing
   print('in train model')
   print(data.head())
   X = data[['Amount',
       'Value','PricingStrategy',
       'TotalTransactionAmount', 'AverageTransactionAmount',
       'TransactionCount', 'TransactionAmountStd', 'TransactionHour',
       'TransactionDay', 'TransactionMonth', 'TransactionYear',
       'ProductCategory_data_bundles', 'ProductCategory_financial_services',
       'ProductCategory_movies', 'ProductCategory_other',
       'ProductCategory_ticket', 'ProductCategory_transport',
       'ProductCategory_tv', 'ProductCategory_utility_bill',
       'ChannelId_ChannelId_2', 'ChannelId_ChannelId_3',
       'ChannelId_ChannelId_5', 'Recency', 'Frequency', 'Monetary', 'Status',
       'RFMS_Score', 'Label']]
   # I want to convert all the columns to numeric

   y = data['FraudResult']
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # train a model
   # smote = SMOTE(sampling_strategy='auto')
   # X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
   r_model = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced')
   r_model.fit(X_test, y_test)

   l_model = LogisticRegression( class_weight='balanced', max_iter=1000, solver='liblinear')
   l_model.fit(X_train, y_train)

   # make predictions
   y_pred_ran = r_model.predict(X_test)
   y_pred_log = l_model.predict(X_test)

   

   # evaluate the models
   print('Random Forest Classifier')
   print('Accuracy: ', accuracy_score(y_test, y_pred_ran))
   print('Precision: ', precision_score(y_test, y_pred_ran))
   print('Recall: ', recall_score(y_test, y_pred_ran))
   print('F1 Score: ', f1_score(y_test, y_pred_ran))

   print('Logistic Regression')
   print('Accuracy: ', accuracy_score(y_test, y_pred_log))
   print('Precision: ', precision_score(y_test, y_pred_log))
   print('Recall: ', recall_score(y_test, y_pred_log))
   print('F1 Score: ', f1_score(y_test, y_pred_log))
   print('x test data')
   print('heads value ###########################################################')
   print(X_test.iloc[0])
   if accuracy_score(y_test, y_pred_ran) > accuracy_score(y_test, y_pred_log):
      return r_model
   return l_model
```


## Evaluation

The models are evaluated using various metrics such as accuracy, precision, recall, and F1 score. The evaluation results are printed for both Random Forest and Logistic Regression models.


```257:267:src/notebook/Bati_bank_task.ipynb
   print('Random Forest Classifier')
   print('Accuracy: ', accuracy_score(y_test, y_pred_ran))
   print('Precision: ', precision_score(y_test, y_pred_ran))
   print('Recall: ', recall_score(y_test, y_pred_ran))
   print('F1 Score: ', f1_score(y_test, y_pred_ran))

   print('Logistic Regression')
   print('Accuracy: ', accuracy_score(y_test, y_pred_log))
   print('Precision: ', precision_score(y_test, y_pred_log))
   print('Recall: ', recall_score(y_test, y_pred_log))
   print('F1 Score: ', f1_score(y_test, y_pred_log))
```


## Contributing

Contributions to improve the model or extend its functionality are welcome. Please submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License.