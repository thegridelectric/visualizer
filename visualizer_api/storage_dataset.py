import os
import time
import dotenv
import pendulum
from sqlalchemy import create_engine, asc
from sqlalchemy.orm import sessionmaker
from config import Settings
from models import MessageSql
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score


class StorageDataset():
    def __init__(self, house_alias, start_ms, timezone):
        settings = Settings(_env_file=dotenv.find_dotenv())
        engine = create_engine(settings.db_url.get_secret_value())
        Session = sessionmaker(bind=engine)
        self.session = Session()
        self.house_alias = house_alias
        self.dataset_file = f"{self.house_alias}_storage_data.csv"
        self.start_ms = start_ms
        self.timezone_str = timezone
        self.data_format = {
            'time': [],
            'store_heat_in': [],
            'tank1-depth1-initial': [],
            'tank1-depth2-initial': [],
            'tank1-depth3-initial': [],
            'tank1-depth4-initial': [],
            'tank2-depth1-initial': [],
            'tank2-depth2-initial': [],
            'tank2-depth3-initial': [],
            'tank2-depth4-initial': [],
            'tank3-depth1-initial': [],
            'tank3-depth2-initial': [],
            'tank3-depth3-initial': [],
            'tank3-depth4-initial': [],
            'tank1-depth1-final': [],
            'tank1-depth2-final': [],
            'tank1-depth3-final': [],
            'tank1-depth4-final': [],
            'tank2-depth1-final': [],
            'tank2-depth2-final': [],
            'tank2-depth3-final': [],
            'tank2-depth4-final': [],
            'tank3-depth1-final': [],
            'tank3-depth2-final': [],
            'tank3-depth3-final': [],
            'tank3-depth4-final': [],
        }

    def generate_dataset(self):
        print("Generating dataset...")
        existing_dataset_dates = []
        if os.path.exists(self.dataset_file):
            df = pd.read_csv(self.dataset_file)
            existing_dataset_dates = list(df['time'])
        day_start_ms = int(pendulum.from_timestamp(self.start_ms/1000, tz=self.timezone_str).replace(hour=0, minute=0).timestamp()*1000)
        day_end_ms = day_start_ms + (24*60+7)*60*1000
        for day in range(200):
            if day_start_ms/1000 > time.time():
                print("Will not look for data in the future.")
                return
            if day_start_ms in existing_dataset_dates and day_start_ms+24*3600*1000 in existing_dataset_dates:
                print(f"\nAlready in dataset: {self.unix_ms_to_date(day_start_ms)}")
            else:
                self.add_data(day_start_ms, day_end_ms)
            day_start_ms += 24*3600*1000
            day_end_ms += 24*3600*1000

    def add_data(self, start_ms, end_ms):
        print(f"\nProcessing reports from: {self.unix_ms_to_date(start_ms)}")
        reports: List[MessageSql] = self.session.query(MessageSql).filter(
            MessageSql.from_alias.like(f'%{self.house_alias}%'),
            MessageSql.message_type_name == "report",
            MessageSql.message_persisted_ms >= start_ms,
            MessageSql.message_persisted_ms <= end_ms,
        ).order_by(asc(MessageSql.message_persisted_ms)).all()
        
        print(f"Found {len(reports)} reports in database")
        if not reports:
            return
        
        formatted_data = pd.DataFrame(self.data_format)

        hour_start_ms = int(start_ms) - 3600*1000
        hour_end_ms = int(start_ms)

        while hour_end_ms <= end_ms:
            hour_start_ms += 3600*1000
            hour_end_ms += 3600*1000

            channels = {}
            for message in [
                m for m in reports
                if self.house_alias in m.from_alias
                and m.message_persisted_ms >= hour_start_ms - 7*60*1000
                and m.message_persisted_ms <= hour_end_ms + 7*60*1000
                ]:
                for channel in message.payload['ChannelReadingList']:
                    channel_name = channel['ChannelName']
                    if 'tank' in channel_name and 'depth' in channel_name and 'micro' not in channel_name:
                        if channel_name not in channels:
                            channels[channel_name] = {'times': [], 'values': []}
                        channels[channel_name]['times'].extend(channel['ScadaReadTimeUnixMsList'])
                        channels[channel_name]['values'].extend(channel['ValueList'])
            if not channels:
                continue

            hour_start_times, hour_start_values = [], []
            hour_end_times, hour_end_values = [], []

            for channel in channels:
                sorted_times_values = sorted(zip(channels[channel]['times'], channels[channel]['values']))
                sorted_times, sorted_values = zip(*sorted_times_values)
                channels[channel]['times'] = list(sorted_times)      
                channels[channel]['values'] = list(sorted_values)

                times_from_start = [abs(time-hour_start_ms) for time in channels[channel]['times']]
                closest_index = times_from_start.index(min(times_from_start))
                hour_start_times.append(channels[channel]['times'][closest_index])
                hour_start_values.append(self.to_fahrenheit(channels[channel]['values'][closest_index]/1000))

                times_from_end = [abs(time-hour_end_ms) for time in channels[channel]['times']]
                closest_index = times_from_end.index(min(times_from_end))
                hour_end_times.append(channels[channel]['times'][closest_index])
                hour_end_values.append(self.to_fahrenheit(channels[channel]['values'][closest_index]/1000))

            if hour_end_times[-1] - hour_start_times[-1] < 45*60*1000:
                continue
            if len(hour_start_values)<12 or len(hour_end_values)<12:
                continue

            average_temp_start = sum(hour_start_values)/12
            average_temp_end = sum(hour_end_values)/12
            store_heat_in = round(3*120*3.785*4.187/3600*(average_temp_end-average_temp_start),1)

            row = [int(hour_start_ms), store_heat_in] + hour_start_values + hour_end_values
            formatted_data.loc[len(formatted_data)] = row 

        formatted_data.to_csv(
            self.dataset_file, 
            mode='a' if os.path.exists(self.dataset_file) else 'w',
            header=False if os.path.exists(self.dataset_file) else True, 
            index=False,
        )
        print(f"Added {len(formatted_data)} new lines to the dataset")

    def model_fit(self, plot=False):
        if not os.path.exists(self.dataset_file):
            print(f"Could not find {self.dataset_file}")
            return
        df = pd.read_csv(self.dataset_file)
        self.X = df[[c for c in df.columns if 'initial' in c or 'store_heat_in' in c]]
        self.y = df[[c for c in df.columns if 'final' in c]]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.1, random_state=42)
        self.ridge_model = Ridge(alpha=1.0)
        self.ridge_model.fit(self.X_train, self.y_train)
        print(f"Done fitting regression model with data from {self.dataset_file}")
        if plot:
            plt.hist(df['store_heat_in'], bins=50, label="All data", alpha=0.6)
            plt.hist(self.X_train['store_heat_in'], bins=50, label="Training data", alpha=0.6)
            plt.hist(self.X_test['store_heat_in'], bins=50, label="Testing data", alpha=0.6)
            plt.xlabel("Store heat in [kWh]")
            plt.ylabel("Occurences")
            plt.legend()
            plt.show()

    def test_model_performance(self, plot=False):
        y_pred_ridge = self.ridge_model.predict(self.X_test)
        mse_ridge = mean_squared_error(self.y_test, y_pred_ridge)
        r2_ridge = r2_score(self.y_test, y_pred_ridge)
        ridge_cv_score = cross_val_score(self.ridge_model, self.X, self.y, cv=5, scoring='neg_mean_squared_error')
        print(f"RMSE: {round(np.sqrt(abs(mse_ridge)),1)}")
        print(f"R2: {round(r2_ridge,1)}")
        print(f"Cross-validation RMSE: {round(np.sqrt(abs(ridge_cv_score.mean())),1)}")
        if plot:
            plt.scatter(self.y_test, y_pred_ridge, alpha=0.3)
            plt.plot(self.y_test, self.y_test, color='red')
            plt.xlabel('True final temperature [F]')
            plt.ylabel('Predicted final temperature [F]')
            plt.xlim([80,180])
            plt.ylim([80,180])
            plt.title(f"{self.dataset_file}\nRMSE: {round(np.sqrt(abs(mse_ridge)),1)}")
            plt.show()

    def unix_ms_to_date(self, time_ms):
        return str(pendulum.from_timestamp(time_ms/1000, tz=self.timezone_str).format('YYYY-MM-DD'))
    
    def to_fahrenheit(self, t):
        return round(t*9/5+32,1)
    

if __name__ == '__main__':
    house_alias = 'maple'
    timezone = 'America/New_York'
    start_ms = pendulum.datetime(2025,1,1, tz=timezone).timestamp()*1000
    
    s = StorageDataset(house_alias, start_ms, timezone)
    # s.generate_dataset()
    s.model_fit()