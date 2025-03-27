import dotenv
import pendulum
from sqlalchemy import create_engine, asc
from sqlalchemy.orm import sessionmaker
from config import Settings
from models import MessageSql
from typing import List
import pandas as pd
import os
import time


class StorageDatasetCreator():
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
        for day in range(30):
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

    def unix_ms_to_date(self, time_ms):
        return str(pendulum.from_timestamp(time_ms/1000, tz=self.timezone_str).format('YYYY-MM-DD'))
    
    def from_dict_msg(self, data):
        message = MessageSql(
            message_id=data["MessageId"],
            from_alias=data["FromAlias"],
            message_type_name=data["MessageTypeName"],
            message_persisted_ms=data["MessagePersistedMs"],
            payload=data["Payload"],
        )
        return message

    def to_fahrenheit(self, t):
        return round(t*9/5+32,1)
    

if __name__ == '__main__':
    start_ms = pendulum.datetime(2025,3,6, tz='America/New_York').timestamp()*1000
    s = StorageDatasetCreator('maple', start_ms, 'America/New_York')
    s.generate_dataset()