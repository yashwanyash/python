import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.preproessing import OneHotEncoder, StandardScaler

data = {
	'Age':[25,39,None,28,35]
	'Gender':['Female','Male','Male','Female','Male'],
	'Income':[50000,60000,45000,None,70000]
}

df=pd.DataFrame(data)

imputer=SimpleImputer(strategy='mean')
df[['Age','Income']]=imputer.fit_transform(df[['Age','Income']])

print("Data after handling missing values:")
print(df)

encoder=OneHotEncoder()
encoded_data=encoder.fit_transform(df[['Gender']]).toarray()

encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['Gender']))
print("\nData after categorical encoding;")
print(encoded_df)

scaler=StandardScaler()
scaled_data-scaler.fit_transform(df[['Age','Income']])

scaled_df=pd.DataFrame(scaled_data, columns=['Scaled Age', 'Scaled Income'])
print("\n Data after feature scaling:")
print(scaled_df)
