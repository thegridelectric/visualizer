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


class EnergyDataset():
    def __init__(self, house_alias, start_ms, end_ms, timezone):
        settings = Settings(_env_file=dotenv.find_dotenv())
        engine = create_engine(settings.db_url_no_async.get_secret_value())
        Session = sessionmaker(bind=engine)
        self.session = Session()
        self.house_alias = house_alias
        self.start_ms = start_ms
        self.end_ms = end_ms
        start_date = pendulum.from_timestamp(self.start_ms/1000, tz=timezone)
        end_date = pendulum.from_timestamp(self.end_ms/1000, tz=timezone)
        start_date_str = f"{start_date.year}-{start_date.month}-{start_date.day}"
        end_date_str = f"{end_date.year}-{end_date.month}-{end_date.day}"
        self.dataset_file = f"energy_data_{self.house_alias}_{start_date_str}_{end_date_str}.csv"
        self.timezone_str = timezone
        self.data_format = {
            'hour_start': [],
            'hp_elec_in': [],
            'hp_heat_out': [],
            'change_in_storage_and_buffer': [],
            'implied_heat_load': [],
            'average_store_temp_start': [],
            'average_store_temp_end': [],
            'average_buffer_temp_start': [],
            'average_buffer_temp_end': [],
        }

    def generate_dataset(self):
        print("Generating dataset...")
        existing_dataset_dates = []
        if os.path.exists(self.dataset_file):
            print(f"Found existing dataset: {self.dataset_file}")
            df = pd.read_csv(self.dataset_file)
            existing_dataset_dates = list(df['hour_start'])
        day_start_ms = int(pendulum.from_timestamp(self.start_ms/1000, tz=self.timezone_str).replace(hour=0, minute=0).timestamp()*1000)
        day_end_ms = day_start_ms + (24*60+7)*60*1000
        for day in range(200):
            if day_start_ms > self.end_ms or day_start_ms/1000 > time.time():
                print("\nDone.")
                return
            day_start = pendulum.from_timestamp(int(day_start_ms)/1000, tz="America/New_York").format('YYYY-MM-DD-HH:00')
            day_end = pendulum.from_timestamp(int(day_start_ms+24*3600*1000)/1000, tz="America/New_York").format('YYYY-MM-DD-HH:00')
            if day_start in existing_dataset_dates and day_end in existing_dataset_dates:
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
                    if channel_name not in channels:
                        channels[channel_name] = {'times': [], 'values': []}
                    channels[channel_name]['times'].extend(channel['ScadaReadTimeUnixMsList'])
                    channels[channel_name]['values'].extend(channel['ValueList'])
            if not channels:
                continue
            for channel in channels.keys():
                sorted_times_values = sorted(zip(channels[channel]['times'], channels[channel]['values']))
                sorted_times, sorted_values = zip(*sorted_times_values)
                channels[channel]['values'] = list(sorted_values)
                channels[channel]['times'] = list(sorted_times)

            # HP heat out, electricity in, cop
            hp_channels = ['hp-idu-pwr', 'hp-odu-pwr', 'primary-flow', 'hp-lwt', 'hp-ewt']
            missing_channels = [c for c in hp_channels if c not in channels]
            if missing_channels: 
                # print(f"Missing channels {missing_channels}")
                continue

            timestep_seconds = 1
            num_points = int((hour_end_ms - hour_start_ms) / (timestep_seconds * 1000) + 1)

            csv_times = np.linspace(hour_start_ms, hour_end_ms, num_points)
            csv_times_dt = pd.to_datetime(csv_times, unit='ms', utc=True)
            csv_times_dt = [x.tz_convert(self.timezone_str).replace(tzinfo=None) for x in csv_times_dt]

            csv_values = {}
            for channel in hp_channels:
                channels[channel]['times'] = pd.to_datetime(channels[channel]['times'], unit='ms', utc=True)
                channels[channel]['times'] = [x.tz_convert(self.timezone_str) for x in channels[channel]['times']]
                channels[channel]['times'] = [x.replace(tzinfo=None) for x in channels[channel]['times']]
                
                merged = pd.merge_asof(
                    pd.DataFrame({'times': csv_times_dt}),
                    pd.DataFrame(channels[channel]),
                    on='times',
                    direction='backward')
                csv_values[channel] = list(merged['values'])

            df = pd.DataFrame(csv_values)
            df['lift_C'] = df['hp-lwt'] - df['hp-ewt']
            df['lift_C'] = [x/1000 if x>0 else 0 for x in list(df['lift_C'])]
            df['flow_kgs'] = df['primary-flow'] / 100 / 60 * 3.78541 
            df['heat_power_kW'] = [m*4187*T/1000  for T,m in zip(df.lift_C, df.flow_kgs)]
            df['cumulative_energy_kWh'] = df['heat_power_kW'].cumsum()
            df['cumulative_energy_kWh'] = df['cumulative_energy_kWh'] / 3600 * timestep_seconds
            hp_heat_out = round(list(df['cumulative_energy_kWh'])[-1] - list(df['cumulative_energy_kWh'])[0],2)
            df['hp_power'] = df['hp-idu-pwr'] + df['hp-odu-pwr']
            hp_elec_in = round(np.mean(df['hp_power'])/1000,2)
            cop = 0 if hp_elec_in==0 else round(hp_heat_out/hp_elec_in,2)
            if np.isnan(hp_heat_out):
                hp_heat_out = 0
                hp_elec_in = 0
                cop = 0

            # Buffer heat out
            buffer_channels = [x for x in channels if 'buffer' in x and 'depth' in x and 'micro' not in x]

            hour_start_times, hour_start_values = [], []
            hour_end_times, hour_end_values = [], []

            for channel in buffer_channels:
                sorted_times_values = sorted(zip(channels[channel]['times'], channels[channel]['values']))
                sorted_times, sorted_values = zip(*sorted_times_values)
                channels[channel]['times'] = list(sorted_times)      
                channels[channel]['values'] = list(sorted_values)

                times_from_start = [abs(time-hour_start_ms) for time in channels[channel]['times']]
                closest_index = times_from_start.index(min(times_from_start))
                hour_start_times.append(channels[channel]['times'][closest_index])
                hour_start_values.append(channels[channel]['values'][closest_index]/1000)

                times_from_end = [abs(time-hour_end_ms) for time in channels[channel]['times']]
                closest_index = times_from_end.index(min(times_from_end))
                hour_end_times.append(channels[channel]['times'][closest_index])
                hour_end_values.append(channels[channel]['values'][closest_index]/1000)

            if hour_end_times[-1] - hour_start_times[-1] < 45*60*1000:
                continue
            
            BASELINE_TEMP = 30
            average_buffer_temp_start = round(sum(hour_start_values)/4,2)
            average_buffer_temp_end = round(sum(hour_end_values)/4,2)
            start_buffer = round(1*120*3.785*4.187/3600*(average_buffer_temp_start-BASELINE_TEMP),2)
            end_buffer = round(1*120*3.785*4.187/3600*(average_buffer_temp_end-BASELINE_TEMP),2)
            buffer_heat_out = round(1*120*3.785*4.187/3600*(average_buffer_temp_start-average_buffer_temp_end),2)

            # Storage heat out
            storage_channels = [x for x in channels if 'tank' in x and 'depth' in x and 'micro' not in x]

            hour_start_times, hour_start_values = [], []
            hour_end_times, hour_end_values = [], []

            for channel in storage_channels:
                sorted_times_values = sorted(zip(channels[channel]['times'], channels[channel]['values']))
                sorted_times, sorted_values = zip(*sorted_times_values)
                channels[channel]['times'] = list(sorted_times)      
                channels[channel]['values'] = list(sorted_values)

                times_from_start = [abs(time-hour_start_ms) for time in channels[channel]['times']]
                closest_index = times_from_start.index(min(times_from_start))
                hour_start_times.append(channels[channel]['times'][closest_index])
                hour_start_values.append(channels[channel]['values'][closest_index]/1000)

                times_from_end = [abs(time-hour_end_ms) for time in channels[channel]['times']]
                closest_index = times_from_end.index(min(times_from_end))
                hour_end_times.append(channels[channel]['times'][closest_index])
                hour_end_values.append(channels[channel]['values'][closest_index]/1000)

            if hour_end_times[-1] - hour_start_times[-1] < 45*60*1000:
                continue

            average_store_temp_start = round(sum(hour_start_values)/12,2)
            average_store_temp_end = round(sum(hour_end_values)/12,2)
            start_storage = round(3*120*3.785*4.187/3600*(average_store_temp_start-BASELINE_TEMP),2)
            end_storage = round(3*120*3.785*4.187/3600*(average_store_temp_end-BASELINE_TEMP),2)
            store_heat_out = round(3*120*3.785*4.187/3600*(average_store_temp_start-average_store_temp_end),2)

            # House heat in
            house_heat_in = round(hp_heat_out + store_heat_out + buffer_heat_out,2)
            # print(f"HP: {hp_heat_out}, Store: {store_heat_out}, Buffer: {buffer_heat_out} => House {house_heat_in}")

            hour_start = pendulum.from_timestamp(int(hour_start_ms)/1000, tz="America/New_York").format('YYYY-MM-DD-HH:00')
            start_storage_and_buffer = round(start_storage + start_buffer,2)
            end_storage_and_buffer = round(end_storage + end_buffer,2)
            change_in_storage_and_buffer = round(end_storage_and_buffer - start_storage_and_buffer,2)
            implied_heat_load = round(hp_heat_out - change_in_storage_and_buffer,2)
            change_buffer = round(end_buffer - start_buffer, 2)
            change_storage = round(end_storage - start_storage, 2)
            row = [
                hour_start, 
                hp_elec_in, 
                hp_heat_out, 
                change_in_storage_and_buffer,
                implied_heat_load,
                self.to_fahrenheit(average_store_temp_start),
                self.to_fahrenheit(average_store_temp_end),
                self.to_fahrenheit(average_buffer_temp_start),
                self.to_fahrenheit(average_buffer_temp_end),
                ]
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
    
    def to_fahrenheit(self, t):
        return round(t*9/5+32,1)
    
def generate(house_alias, start_year, start_month, start_day, end_year, end_month, end_day):
    timezone = 'America/New_York'
    start_ms = pendulum.datetime(start_year, start_month, start_day, tz=timezone).timestamp()*1000
    end_ms = pendulum.datetime(end_year, end_month, end_day, tz=timezone).timestamp()*1000
    s = EnergyDataset(house_alias, start_ms, end_ms, timezone)
    s.generate_dataset()

if __name__ == '__main__':
    start_date = input("\nHi George\nEnter start date YYYY/MM/DD: ")
    end_date = input("Enter end date YYYY/MM/DD: ")
    START_YEAR, START_MONTH, START_DAY = [int(x) for x in start_date.split('/')]
    END_YEAR, END_MONTH, END_DAY = [int(x) for x in end_date.split('/')]

    generate(
        house_alias='beech', 
        start_year=START_YEAR, 
        start_month=START_MONTH, 
        start_day=START_DAY,
        end_year=END_YEAR,
        end_month=END_MONTH,
        end_day=END_DAY
    )