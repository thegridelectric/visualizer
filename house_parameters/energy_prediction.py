import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pendulum
from energy_usage import energy_used
import statsmodels.formula.api as smf

ONPEAK_HOURS = [7,8,9,10,11] + [16,17,18,19]

df_raw = pd.read_csv('weather_data.csv', header=4)
df_raw['Year'] = [2000+int(x.split('/')[-1]) for x in list(df_raw['Date'])]
df_raw['HourlyDryBulbTemperature'] = pd.to_numeric(df_raw['HourlyDryBulbTemperature'], errors='coerce')
df_raw['HourlyWetBulbTemperature'] = pd.to_numeric(df_raw['HourlyWetBulbTemperature'], errors='coerce')
df_raw['HourlyWindSpeed'] = pd.to_numeric(df_raw['HourlyWindSpeed'], errors='coerce')
df_raw['HourlyWindGustSpeed'] = pd.to_numeric(df_raw['HourlyWindGustSpeed'], errors='coerce')
df_raw['HourlyWindDirection'] = pd.to_numeric(df_raw['HourlyWindDirection'], errors='coerce')
df_raw['HourlyPrecipitation'] = pd.to_numeric(df_raw['HourlyPrecipitation'], errors='coerce')
df_raw['HourlyRelativeHumidity'] = pd.to_numeric(df_raw['HourlyRelativeHumidity'], errors='coerce')
df_raw['HourlyStationPressure'] = pd.to_numeric(df_raw['HourlyStationPressure'], errors='coerce')

clean_data = {
    'date': [x for x in list(df_raw['DATE'])],
    'oat_dry': [x for x in list(df_raw['HourlyDryBulbTemperature'])],
    'oat_wet': [x for x in list(df_raw['HourlyWetBulbTemperature'])],
    'wind_speed': [x for x in list(df_raw['HourlyWindSpeed'])],
    'wind_gust_speed': [0 if pd.isna(x) else x for x in list(df_raw['HourlyWindGustSpeed'])],
    'wind_direction': [0 if pd.isna(x) else x for x in list(df_raw['HourlyWindDirection'])],
    'precipitation': [x for x in list(df_raw['HourlyPrecipitation'])],
    'humidity': [x for x in list(df_raw['HourlyRelativeHumidity'])],
    'pressure': [x for x in list(df_raw['HourlyStationPressure'])],
}

df = pd.DataFrame(clean_data)

# Add sky conditions
all_codes = ['BKN', 'OVC', 'VV', 'SCT', 'FEW', 'CLR']
sky_conditions = {}
for code in all_codes:
    sky_conditions[code] = [0]*len(df_raw)
for k in range(len(df_raw)):
    x = str(df_raw['HourlySkyConditions'][k])
    for i in range(x.count(':')):
        sky_conditions[x.split(':')[i][-3:]][k] = 1
for key in sky_conditions:
    df[key] = sky_conditions[key]

# plt.scatter(range(len(df)), df['date'].dt.minute)
# plt.show()

# Group data by hour, taking the mean
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%dT%H:%M:%S')
df['date_grouped'] = df['date'].dt.strftime('%Y-%m-%d %H')
df = df.groupby('date_grouped').mean()
df.reset_index(inplace=True, drop=True)
df = df.round(1)

# Drop rows which contain NaNs
num_nans = len(df)
df = df.dropna()
num_nans = num_nans - len(df)
print(f"There were {num_nans} rows with NaNs")

# Only keep data for the selected time
df_backup = df.copy()
df = df[df['date'] >= pendulum.datetime(2024, 12, 3, tz='America/New_York').naive()]
df = df[df['date'].dt.hour.isin(ONPEAK_HOURS)]
df.reset_index(inplace=True, drop=True)

# Find the energy used
df['energy'] = df.apply(lambda row: 
                             energy_used('oak', row['date'].year, row['date'].month, row['date'].day, row['date'].hour), 
                             axis=1
                             )
df['date'] = df['date'].dt.floor('h')

# Drop rows which contain NaNs again
num_nans = len(df)
df = df.dropna()
num_nans = num_nans - len(df)
print(f"There were {num_nans} rows with NaNs")

print(df)

# Predict energy used by the house using alpha, beta, gamma formula
alpha, beta, gamma = 10.3, -0.18, 0.0015
# alpha, beta, gamma = 9.1715, -0.1589, 0.0056
def energy_alpha_pred(oat, ws):
    return alpha + beta*oat + gamma*ws

df['energy_pred_alpha'] = df.apply(lambda row: energy_alpha_pred(row['oat_dry'], row['wind_speed']), axis=1)

# Split training and testing data
from sklearn.model_selection import train_test_split
features = [col for col in df.columns if col not in ['energy', 'date', 'energy_pred_alpha']]
X = df[features]
y = df['energy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test, df.loc[X_test.index, 'date']], axis=1)

# Fit the model on the training data
binary_columns = [x for x in df.columns if x in all_codes]
for col in binary_columns:
    df[col] = df[col].astype('category')
features = [col for col in df.columns if col != 'energy' and col != 'date' and col != 'energy_pred_alpha']
features = [col for col in df.columns if col in ['oat_dry', 'wind_speed']]
formula = 'energy ~ ' + ' + '.join(features)
formula_interactions = ' + '.join([f'{feat1}:{feat2}' for i, feat1 in enumerate(features) for feat2 in features[i+1:]])
mod = smf.ols(formula=formula, data=train_df)
res = mod.fit()

# Predict on the testing data
test_df['energy_pred_reg'] = res.predict(test_df)
test_df['energy_pred_alpha'] = test_df.apply(lambda row: energy_alpha_pred(row['oat_dry'], row['wind_speed']), axis=1)
test_df = test_df.sort_values(by='date')

plt.figure(figsize=(10,4))
plt.scatter(test_df.date, test_df.energy, label='used', alpha=0.7, color='gray')
plt.scatter(test_df.date, test_df.energy_pred_alpha, label='predicted - alpha', alpha=0.7)
plt.scatter(test_df.date, test_df.energy_pred_reg, label='predicted - new', alpha=0.7)

for day in test_df.date.dt.date.unique():
    start_time = pd.to_datetime(day).replace(hour=7, minute=0, second=0, microsecond=0)
    end_time = pd.to_datetime(day).replace(hour=12, minute=0, second=0, microsecond=0)
    plt.axvspan(start_time, end_time, color='gray', alpha=0.1)

for day in test_df.date.dt.date.unique():
    start_time = pd.to_datetime(day).replace(hour=16, minute=0, second=0, microsecond=0)
    end_time = pd.to_datetime(day).replace(hour=20, minute=0, second=0, microsecond=0)
    plt.axvspan(start_time, end_time, color='gray', alpha=0.1)

plt.ylim([-0.1,16])
plt.ylabel('Heat [kWh]')
plt.legend()
plt.show()

plt.figure(figsize=(10,4))
line_style='-'
plt.plot(test_df.date, test_df.energy, line_style, label='used', alpha=0.4, color='gray')
rmse_alpha = round(((test_df['energy'] - test_df['energy_pred_alpha'])**2).mean()**0.5,2)
rmse_reg = round(((test_df['energy'] - test_df['energy_pred_reg'])**2).mean()**0.5,2)
plt.plot(test_df.date, test_df.energy_pred_alpha, line_style, label=f'predicted - alpha/beta/gamma, RMSE={rmse_alpha}', alpha=0.6)
plt.plot(test_df.date, test_df.energy_pred_reg, line_style, label=f'predicted - new, RMSE={rmse_reg}', alpha=0.6)
plt.ylim([-0.1,16])
plt.ylabel('Heat [kWh]')
plt.legend()
plt.show()