# whrcp

Algorithm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

data = pd.read_csv("/content/Operation Parameters.csv")
data.head()
column_names = data.columns.tolist()
print(column_names)
data=data.drop(['Day','Kiln - Running Hrs', 'Gross Generation Per Ton Clincker', 'Net Generation Per Ton Clincker', 'Average load', 'Total Steam Consumption', 'Specific Steam Consumption', 'Steam Consumption HP', 'Specific Steam Consumption HP', 'DM Water Consumption', 'Total Steam Generation', 'FG Inlet Temperature PH', 'FG Outlet Temperature PH', 'Steam Generation PH', 'FG Inlet Temperature AQC', 'FG Outlet Temperature AQC', 'Steam Generation AQC', 'Steam Pressure at Turbine Inlet', 'Steam Temp. at Turbine Inlet', ' Feed water temp.', 'TG I/L LP Steam Pressure', 'TG I/L LP Steam Temperature'], axis=1)

column_renames = {
    'Kiln - Clinker Production': 'Clinker_prod',
    'Gross Generation': 'Gross_gen',
    'Delta Temp. PH': 'DT_PH',
    'DP Across PH': 'DP_PH',
    'Delta Temp. AQC':'DT_AQC',
    'DP Across AQC':'DP_AQC'
}

# Rename the columns
data = data.rename(columns=column_renames)
data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 263 entries, 0 to 262
Data columns (total 6 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   Clinker_prod  263 non-null    int64  
 1   Gross_gen     263 non-null    int64  
 2   DT_PH         263 non-null    float64
 3   DP_PH         263 non-null    float64
 4   DT_AQC        263 non-null    float64
 5   DP_AQC        263 non-null    float64
dtypes: float64(4), int64(2)
memory usage: 12.5 KB

data.describe()
            Clinker_prod  	Gross_gen	   DT_PH	                    DP_PH	                   DT_AQC	                   DP_AQC
count	263.000000	263.000000	263.000000	263.000000	263.000000	263.000000
mean	5370.612167	171496.604563	179.615209	-58.516768	298.766236	-44.515171
std	535.340950	20178.469873	23.042871	9.146726	                  14.914830	21.036911
min	3345.000000	105240.000000	145.430000	-84.630000	178.390000	-87.270000
25%	4927.500000	157894.500000	163.580000	-64.870000	290.995000	-58.945000
50%	5460.000000	169808.000000	175.320000	-57.350000	300.370000	-44.920000
75%	5790.000000	186272.000000	185.585000	-51.940000	308.295000	-32.380000
max	6168.000000	225863.000000	287.910000	-14.560000	329.740000	123.640000

# unique_values = data['Steam Pressure at Turbine Inlet'].unique()

# print("Unique values in 'ColumnName':")
# print(unique_values)




#Key Parameters vs Gross Generation

plt.figure(figsize=(10, 6))

# Plotting scatterplots
plt.scatter(data['DT_PH'], data['Gross_gen'], color='r', label='Delta Temp. across PH vs Gross Generation')
plt.scatter(data['DT_AQC'], data['Gross_gen'], color='g', label='Delta Temp. across AQC vs Gross Generation')
plt.scatter(data['DP_PH'], data['Gross_gen'], color='b', label='DP Across PH vs Gross Generation')
plt.scatter(data['DP_AQC'], data['Gross_gen'], color='y', label='DP Across AQC vs Gross Generation')

plt.xlabel('Key Parameter Value')
plt.ylabel('Gross Generation')
plt.title('Scatter Plots of Key Parameters vs Gross Generation')
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 6))

# Plotting scatterplots
plt.scatter(data['DT_PH'], data['Clinker_prod'], color='r', label='Delta Temp. across PH vs Clinker Production')
plt.scatter(data['DT_AQC'], data['Clinker_prod'], color='g', label='Delta Temp. across AQC vs Clinker Production')
plt.scatter(data['DP_PH'], data['Clinker_prod'], color='b', label='DP Across PH vs Clinker Production')
plt.scatter(data['DP_AQC'], data['Clinker_prod'], color='y', label='DP Across AQC vs Clinker Production')

plt.xlabel('Key Parameter Value')
plt.ylabel('Clinker Production')
plt.title('Scatter Plots of key Parameters vs Clinker Production')
plt.legend()
plt.grid(True)
plt.show()



# Calculate Z-scores for each column
data['Z_DT_AQC'] = stats.zscore(data['DT_AQC'])
data['Z_Gross_gen'] = stats.zscore(data['Gross_gen'])
#x=data['DT_AQC'], y=data['Gross_gen']

# Define a threshold for Z-scores to identify outliers (e.g., Z > 3 or Z < -3)
threshold = 4
outliers = data[(data['Z_DT_AQC'].abs() > threshold) | (data['Z_Gross_gen'].abs() > threshold)]

# Print outliers
print("Outliers:\n", outliers)

# Filter the DataFrame to remove outliers
data_clean = data[(data['Z_DT_AQC'].abs() <= threshold) & (data['Z_Gross_gen'].abs() <= threshold)]

# Drop the z_score columns
data_clean = data_clean.drop(columns=['Z_DT_AQC', 'Z_Gross_gen'])
print("Clean Data:\n", data_clean)
Outliers:
      Clinker_prod  Gross_gen   DT_PH  DP_PH  DT_AQC  DP_AQC  Z_DT_AQC  \
56           3390     108544  287.91 -28.74  232.74   -5.90 -4.435325   
198          4755     152491  212.09 -53.09  178.39  -17.20 -8.086297   
199          4635     168288  231.82 -14.56  231.82  -14.56 -4.497126   

     Z_Gross_gen  
56     -3.125739  
198    -0.943671  
199    -0.159314  
Clean Data:
      Clinker_prod  Gross_gen   DT_PH  DP_PH  DT_AQC  DP_AQC
0            5775     179832  160.29 -59.95  306.96  -54.42
1            6120     191368  152.13 -64.21  299.46  -75.90
2            6150     187008  157.11 -64.34  301.63  -65.00
3            6135     187816  160.10 -63.22  308.49  -60.24
4            6135     206168  159.07 -65.17  298.53  -86.49
..            ...        ...     ...    ...     ...     ...
258          6015     175294  177.26 -58.10  299.64  -39.41
259          5895     174264  181.80 -64.92  301.97  -34.99
260          5625     167166  183.95 -62.70  306.75  -27.52
261          5565     175978  182.39 -63.82  294.70  -41.83
262          5550     169308  182.76 -69.20  292.70  -35.78

[260 rows x 6 columns]
# Calculate Z-scores for each column
data_clean['Z_DP_AQC'] = stats.zscore(data_clean['DP_AQC'])
data_clean['Z_Gross_gen'] = stats.zscore(data_clean['Gross_gen'])
#x=data['DT_AQC'], y=data['Gross_gen']

# Define a threshold for Z-scores to identify outliers (e.g., Z > 3 or Z < -3)
threshold = 4
outliers = data_clean[(data_clean['Z_DP_AQC'].abs() > threshold) | (data_clean['Z_Gross_gen'].abs() > threshold)]

# Print outliers
print("Outliers:\n", outliers)

# Filter the DataFrame to remove outliers
df = data_clean[(data_clean['Z_DP_AQC'].abs() <= threshold) & (data_clean['Z_Gross_gen'].abs() <= threshold)]

# Drop the z_score columns
df = df.drop(columns=['Z_DP_AQC', 'Z_Gross_gen'])
print("Clean Data:\n", df)

Outliers:
      Clinker_prod  Gross_gen   DT_PH  DP_PH  DT_AQC  DP_AQC  Z_DP_AQC  \
144          5025     166460  172.06 -50.88  292.61  123.64  8.091611   

     Z_Gross_gen  
144    -0.270406  
Clean Data:
      Clinker_prod  Gross_gen   DT_PH  DP_PH  DT_AQC  DP_AQC
0            5775     179832  160.29 -59.95  306.96  -54.42
1            6120     191368  152.13 -64.21  299.46  -75.90
2            6150     187008  157.11 -64.34  301.63  -65.00
3            6135     187816  160.10 -63.22  308.49  -60.24
4            6135     206168  159.07 -65.17  298.53  -86.49
..            ...        ...     ...    ...     ...     ...
258          6015     175294  177.26 -58.10  299.64  -39.41
259          5895     174264  181.80 -64.92  301.97  -34.99
260          5625     167166  183.95 -62.70  306.75  -27.52
261          5565     175978  182.39 -63.82  294.70  -41.83
262          5550     169308  182.76 -69.20  292.70  -35.78

[259 rows x 6 columns]

df.head()
          Clinker_prod	Gross_gen     DT_PH      DP_PH      DT_AQC      DP_AQC
0	5775	179832	       160.29     -59.95        306.96       -54.42
1	6120	191368	      152.13      -64.21        299.46        -75.90
2	6150	187008	      157.11     -64.34	         301.63        -65.00
3	6135	187816	      160.10     -63.22	         308.49         -60.24
4	6135	206168	      159.07     -65.17	          298.53        -86.49

#RUN THIS CELL TO CHECK THE SCATTER PLOT AFTER THE REMOVED OUTLIER

# plt.figure(figsize=(10, 6))

# # Plotting scatterplots
# plt.scatter(df['DT_PH'], df['Gross_gen'], color='r', label='Delta Temp. across PH vs Gross Generation')
# plt.scatter(df['DT_AQC'], df['Gross_gen'], color='g', label='Delta Temp. across AQC vs Gross Generation')
# plt.scatter(df['DP_PH'], df['Gross_gen'], color='b', label='DP Across PH vs Gross Generation')
# plt.scatter(df['DP_AQC'], df['Gross_gen'], color='y', label='DP Across AQC vs Gross Generation')

# plt.xlabel('Key Parameter Value')
# plt.ylabel('Gross Generation')
# plt.title('Scatter Plots of Key Parameters vs Gross Generation')
# plt.legend()
# plt.grid(True)
# plt.show()

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
print("Scaled Data \n", scaled_df)



Scaled Data 
      Clinker_prod  Gross_gen     DT_PH     DP_PH    DT_AQC    DP_AQC
0        0.860786   0.618390  0.146317  0.578935  0.639614  0.400952
1        0.982997   0.714026  0.065971  0.479005  0.520962  0.138777
2        0.993624   0.677881  0.115006  0.475956  0.555292  0.271817
3        0.988310   0.684579  0.144447  0.502228  0.663819  0.329916
4        0.988310   0.836723  0.134305  0.456486  0.506249  0.009520
..            ...        ...       ...       ...       ...       ...
254      0.945802   0.580768  0.313411  0.622332  0.523810  0.584157
255      0.903294   0.572229  0.358113  0.462350  0.560671  0.638106
256      0.807651   0.513385  0.379283  0.514426  0.636292  0.729281
257      0.786397   0.586439  0.363923  0.488154  0.445657  0.554620
258      0.781084   0.531142  0.367566  0.361952  0.414017  0.628463

[259 rows x 6 columns]

X = scaled_df[['DT_PH', 'DP_PH', 'DT_AQC', 'DP_AQC']]
y_clinker = scaled_df['Clinker_prod']
y_gross = scaled_df['Gross_gen']
X_train, X_test, y_clinker_train, y_clinker_test = train_test_split(X, y_clinker, test_size=0.3, random_state=42)
X_train, X_test, y_gross_train, y_gross_test = train_test_split(X, y_gross, test_size=0.3, random_state=42)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(n_estimators=100),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100)
}

# Train models for Clinker Production
clinker_models = {}
for name, model in models.items():
    model.fit(X_train, y_clinker_train)
    clinker_models[name] = model

# Train models for Gross Generation
gross_models = {}
for name, model in models.items():
    model.fit(X_train, y_gross_train)
    gross_models[name] = model

clinker_mse = {}
for name, model in clinker_models.items():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_clinker_test, y_pred)
    clinker_mse[name] = mse

# Evaluate models for Gross Generation
gross_mse = {}
for name, model in gross_models.items():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_gross_test, y_pred)
    gross_mse[name] = mse

print("Mean Squared Error for Clinker Production:")
for name, mse in clinker_mse.items():
    print(f"{name}: {mse}")

print("\nMean Squared Error for Gross Generation:")
for name, mse in gross_mse.items():
    print(f"{name}: {mse}")

Mean Squared Error for Clinker Production:
Linear Regression: 0.04540499654931677
Decision Tree: 0.04370648957459557
Random Forest: 0.04410652234139377
Gradient Boosting: 0.047437197306950896

Mean Squared Error for Gross Generation:
Linear Regression: 0.006634726434736136
Decision Tree: 0.011221554316646842
Random Forest: 0.006487867039092478
Gradient Boosting: 0.006205944953937228




import numpy as np

# Define the full actual ranges for each parameter
actual_ranges = {
    'DT_PH': (145.43, 246.99),
    'DP_PH': (-84.63, -42),
    'DT_AQC': (266.53, 329.74),
    'DP_AQC': (-87.27, -5.34),
}

# Function to scale values using min-max scaling
def scale_values(values, min_val, max_val):
    return (values - min_val) / (max_val - min_val)

# Generate a smaller parameter space within the actual ranges for optimization
param_space_actual = {
    'DT_PH': np.linspace(160, 200, 10),
    'DP_PH': np.linspace(-80, -50, 10),
    'DT_AQC': np.linspace(280, 310, 10),
    'DP_AQC': np.linspace(-80, -20, 10),
}

# Scale the parameter space
param_space_scaled = {
    'DT_PH': scale_values(param_space_actual['DT_PH'], actual_ranges['DT_PH'][0], actual_ranges['DT_PH'][1]),
    'DP_PH': scale_values(param_space_actual['DP_PH'], actual_ranges['DP_PH'][0], actual_ranges['DP_PH'][1]),
    'DT_AQC': scale_values(param_space_actual['DT_AQC'], actual_ranges['DT_AQC'][0], actual_ranges['DT_AQC'][1]),
    'DP_AQC': scale_values(param_space_actual['DP_AQC'], actual_ranges['DP_AQC'][0], actual_ranges['DP_AQC'][1]),
}

print("Scaled Parameter Space:")
print(param_space_scaled)


Scaled Parameter Space:
{'DT_PH': array([0.14346199, 0.18722375, 0.23098551, 0.27474728, 0.31850904,
       0.3622708 , 0.40603256, 0.44979432, 0.49355608, 0.53731784]), 'DP_PH': array([0.10860896, 0.18680116, 0.26499335, 0.34318555, 0.42137775,
       0.49956994, 0.57776214, 0.65595434, 0.73414653, 0.81233873]), 'DT_AQC': array([0.21309919, 0.26583347, 0.31856774, 0.37130201, 0.42403628,
       0.47677055, 0.52950483, 0.5822391 , 0.63497337, 0.68770764]), 'DP_AQC': array([0.08873429, 0.17010456, 0.25147484, 0.33284511, 0.41421539,
       0.49558566, 0.57695594, 0.65832621, 0.73969649, 0.82106676])}



# Assuming X_train, y_clinker_train, y_gross_train, X_test, y_clinker_test, y_gross_test are already defined
clinker_models = {}
gross_models = {}

for name, model in models.items():
    model.fit(X_train, y_clinker_train)
    clinker_models[name] = model

for name, model in models.items():
    model.fit(X_train, y_gross_train)
    gross_models[name] = model

# Evaluate the models on the test set
model_scores = []

for model_name, gross_model in gross_models.items():
    y_gross_pred = gross_model.predict(X_test)
    r2_gross = r2_score(y_gross_test, y_gross_pred)
    model_scores.append({'Model': model_name, 'Metric': 'R2', 'Gross_Gen_Score': r2_gross})

for model_name, clinker_model in clinker_models.items():
    y_clinker_pred = clinker_model.predict(X_test)
    r2_clinker = r2_score(y_clinker_test, y_clinker_pred)
    for score in model_scores:
        if score['Model'] == model_name:
            score['Clinker_Prod_Score'] = r2_clinker

# Convert model scores to DataFrame
model_scores_df = pd.DataFrame(model_scores)

# Find the best model for Gross Generation and Clinker Production
best_gross_gen_model_name = model_scores_df.loc[model_scores_df['Gross_Gen_Score'].idxmax()]['Model']
best_clinker_prod_model_name = model_scores_df.loc[model_scores_df['Clinker_Prod_Score'].idxmax()]['Model']

best_gross_gen_model = gross_models[best_gross_gen_model_name]
best_clinker_prod_model = clinker_models[best_clinker_prod_model_name]

print(f"Best model for Gross Generation: {best_gross_gen_model_name}")
print(f"Best model for Clinker Production: {best_clinker_prod_model_name}")

# Create a DataFrame to store results
results = []

# Loop over all combinations of scaled parameter values using the best models
for dt_ph in param_space_scaled['DT_PH']:
    for dp_ph in param_space_scaled['DP_PH']:
        for dt_aqc in param_space_scaled['DT_AQC']:
            for dp_aqc in param_space_scaled['DP_AQC']:
                test_data = pd.DataFrame({
                    'DT_PH': [dt_ph],
                    'DP_PH': [dp_ph],
                    'DT_AQC': [dt_aqc],
                    'DP_AQC': [dp_aqc],
                })

                # Predict Gross Generation and Clinker Production using the best models
                gross_gen_pred = best_gross_gen_model.predict(test_data)
                clinker_prod_pred = best_clinker_prod_model.predict(test_data)

                # Store the results
                results.append({
                    'DT_PH': dt_ph,
                    'DP_PH': dp_ph,
                    'DT_AQC': dt_aqc,
                    'DP_AQC': dp_aqc,
                    'Gross_Gen': gross_gen_pred[0],
                    'Clinker_Prod': clinker_prod_pred[0]
                })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Find the best parameter combination for Gross Generation
best_gross_gen = results_df.loc[results_df['Gross_Gen'].idxmax()]

# Find the best parameter combination for Clinker Production
best_clinker_prod = results_df.loc[results_df['Clinker_Prod'].idxmax()]

# Normalize the scores
results_df['Gross_Gen_Norm'] = (results_df['Gross_Gen'] - results_df['Gross_Gen'].min()) / (results_df['Gross_Gen'].max() - results_df['Gross_Gen'].min())
results_df['Clinker_Prod_Norm'] = (results_df['Clinker_Prod'] - results_df['Clinker_Prod'].min()) / (results_df['Clinker_Prod'].max() - results_df['Clinker_Prod'].min())

# # Calculate the combined score (you can adjust the weights as needed)
# results_df['Combined_Score'] = results_df['Gross_Gen_Norm'] + results_df['Clinker_Prod_Norm']

# # Find the best parameter combination for the combined score
# best_combined_params = results_df.loc[results_df['Combined_Score'].idxmax()]

# Calculate the harmonic mean of the normalized scores
results_df['Harmonic_Mean'] = 2 * (results_df['Gross_Gen_Norm'] * results_df['Clinker_Prod_Norm']) / (results_df['Gross_Gen_Norm'] + results_df['Clinker_Prod_Norm'])

# Find the best parameter combination for the harmonic mean
best_combined_params = results_df.loc[results_df['Harmonic_Mean'].idxmax()]


print("Best parameters for maximum Gross Generation:")
print(best_gross_gen)

print("\nBest parameters for maximum Clinker Production:")
print(best_clinker_prod)

print("\nBest parameters for maximizing both Gross Generation and Clinker Production (using Harmonic Mean):")
print(best_combined_params)

Best model for Gross Generation: Gradient Boosting
Best model for Clinker Production: Random Forest
Best parameters for maximum Gross Generation: 

DT_PH           0.537318
DP_PH           0.421378
DT_AQC          0.687708
DP_AQC          0.088734
Gross_Gen       0.885555
Clinker_Prod    0.815440
Name: 9490, dtype: float64

Best parameters for maximum Clinker Production: 

DT_PH           0.362271
DP_PH           0.421378
DT_AQC          0.582239
DP_AQC          0.088734
Gross_Gen       0.845512
Clinker_Prod    0.823568
Name: 5470, dtype: float64
Best parameters for maximizing both Gross Generation and Clinker Production (using Harmonic Mean): 

DT_PH                0.537318
DP_PH                0.421378
DT_AQC               0.687708
DP_AQC               0.088734
Gross_Gen            0.885555
Clinker_Prod         0.815440
Gross_Gen_Norm       1.000000
Clinker_Prod_Norm    0.984616
Harmonic_Mean        0.992248
Name: 9490, dtype: float64











def reverse_scale_values(scaled_values, min_val, max_val):
    return scaled_values * (max_val - min_val) + min_val

# Define the actual ranges for each parameter
actual_ranges = {
    'DT_PH': (145.43, 246.99),
    'DP_PH': (-84.63, -42),
    'DT_AQC': (266.53, 329.74),
    'DP_AQC': (-87.27, -5.34),
}

# Extract the best scaled parameters from your results
best_gross_gen_params = {
    'DT_PH': 0.537318,
    'DP_PH': 0.421378,
    'DT_AQC': 0.687708,
    'DP_AQC': 0.088734,
}

best_clinker_prod_params = {
    'DT_PH': 0.362271,
    'DP_PH': 0.421378,
    'DT_AQC': 0.582239,
    'DP_AQC': 0.088734,
}

best_combined_params = {
    'DT_PH': 0.537318,
    'DP_PH': 0.421378,
    'DT_AQC': 0.687708,
    'DP_AQC': 0.088734,
}

# Convert scaled values back to actual values
best_gross_gen_params_actual = {param: reverse_scale_values(value, actual_ranges[param][0], actual_ranges[param][1])
                                for param, value in best_gross_gen_params.items()}

best_clinker_prod_params_actual = {param: reverse_scale_values(value, actual_ranges[param][0], actual_ranges[param][1])
                                   for param, value in best_clinker_prod_params.items()}

best_combined_params_actual = {param: reverse_scale_values(value, actual_ranges[param][0], actual_ranges[param][1])
                               for param, value in best_combined_params.items()}

# Print the results
print("Best parameters for maximum Gross Generation (actual values):")
print(best_gross_gen_params_actual)

print("\nBest parameters for maximum Clinker Production (actual values):")
print(best_clinker_prod_params_actual)

print("\nBest parameters for maximizing both Gross Generation and Clinker Production (actual values):")
print(best_combined_params_actual)


Best parameters for maximum Gross Generation (actual values):
{'DT_PH': 200.00001608, 'DP_PH': -66.66665585999999, 'DT_AQC': 310.00002268, 'DP_AQC': -80.00002338}

Best parameters for maximum Clinker Production (actual values):
{'DT_PH': 182.22224276, 'DP_PH': -66.66665585999999, 'DT_AQC': 303.33332719, 'DP_AQC': -80.00002338}

Best parameters for maximizing both Gross Generation and Clinker Production (actual values):
{'DT_PH': 200.00001608, 'DP_PH': -66.66665585999999, 'DT_AQC': 310.00002268, 'DP_AQC': -80.00002338}
