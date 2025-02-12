from flo import DGraph
from named_types import FloParamsHouse0
import dotenv
import pendulum
from sqlalchemy import create_engine, desc, asc, or_
from sqlalchemy.orm import sessionmaker
from fake_config import Settings
from fake_models import MessageSql
import matplotlib.pyplot as plt
from typing import List
import numpy as np
import pandas as pd
from datetime import timedelta

PRINT = False


class FloEval():

    def __init__(self, house_alias, start_ms, timezone):
        settings = Settings(_env_file=dotenv.find_dotenv())
        engine = create_engine(settings.db_url.get_secret_value())
        Session = sessionmaker(bind=engine)
        self.session = Session()
        self.house_alias = house_alias
        self.flo_start_ms = start_ms
        self.timezone_str = timezone
        self.selected_messages = None
        self.get_flo_params()

    def get_flo_params(self):
        flo_params_messages: List[MessageSql] = self.session.query(MessageSql).filter(
            MessageSql.message_type_name == "flo.params.house0",
            MessageSql.from_alias.like(f'%{self.house_alias}%'),
            MessageSql.message_persisted_ms >= self.flo_start_ms - 0.5*3600*1000,
            MessageSql.message_persisted_ms <= self.flo_start_ms + 48*3600*1000,
        ).order_by(desc(MessageSql.message_persisted_ms)).all()

        # Remove duplicates
        self.flo_params_messages: List[MessageSql] = []
        for m in flo_params_messages:
            if not [x for x in self.flo_params_messages if x.payload['StartUnixS'] == m.payload['StartUnixS']]:
                self.flo_params_messages.append(m)

        flo_params_msg = self.flo_params_messages[0]
        print(f"Found FLO at {pendulum.from_timestamp(flo_params_msg.message_persisted_ms/1000, tz=self.timezone_str)}")
        self.flo_params = FloParamsHouse0(**flo_params_msg.payload)

    def get_load_forecast(self):
        alpha = self.flo_params.AlphaTimes10/10
        beta = self.flo_params.BetaTimes100/100
        gamma = self.flo_params.GammaEx6/1e6
        # Forecast at the time of the selected FLO
        oat_forecast = self.flo_params.OatForecastF
        ws_forecast = self.flo_params.WindSpeedForecastMph
        self.load_forecast = [
            alpha + beta*oat + gamma*ws if alpha + beta*oat + gamma*ws>0 else 0 for oat,ws in zip(oat_forecast, ws_forecast)
            ]
        # First forecast for each hour
        oat_true = [x.payload['OatForecastF'][0] for x in self.flo_params_messages]
        ws_true = [x.payload['WindSpeedForecastMph'][0] for x in self.flo_params_messages]
        times = [pendulum.from_timestamp(x.payload['StartUnixS'],tz=self.timezone_str) for x in self.flo_params_messages]
        self.load_forecast_true_weather = [
            alpha + beta*oat + gamma*ws if alpha + beta*oat + gamma*ws>0 else 0 for oat,ws in zip(oat_true, ws_true)
            ]
        sorted_times_values = sorted(zip(times, self.load_forecast_true_weather))
        sorted_times, sorted_values = zip(*sorted_times_values)
        self.load_forecast_true_weather = list(sorted_values)

    def get_hp_forecast(self):
        g = DGraph(self.flo_params)
        g.solve_dijkstra()
        sp_heat_out = []
        the_end = False
        node_i = g.initial_node
        while not the_end:
            if node_i.next_node is None:
                the_end = True
                sp_heat_out.append(edge_i.hp_heat_out)
            else:
                edge_i = [e for e in g.edges[node_i] if e.head==node_i.next_node][0]
                sp_heat_out.append(edge_i.hp_heat_out)
            node_i = node_i.next_node
        self.hp_forecast = sp_heat_out

    def get_hp_and_load_true(self):
        self.get_hp_and_tanks_energy_out()
        self.load_true = [x+y for x,y in zip(self.hp_true, self.tanks_energy_out)]

    def get_selected_messages(self):
        start_ms = self.flo_start_ms + 0*3600*1000
        end_ms = self.flo_start_ms + 48*3600*1000
        self.selected_messages: List[MessageSql] = self.session.query(MessageSql).filter(
            MessageSql.from_alias.like(f'%{self.house_alias}%'),
            or_(
                MessageSql.message_type_name == "report",
                MessageSql.message_type_name == "energy.instruction"
                ),
            MessageSql.message_persisted_ms >= start_ms,
            MessageSql.message_persisted_ms <= end_ms,
        ).order_by(asc(MessageSql.message_persisted_ms)).all()
        if not self.selected_messages:
            raise Exception('No messages found!')
        
    def compare_power_instruction_vs_real(self):
        if self.selected_messages is None:
            self.get_selected_messages()

        self.energy_instructions_avg_power = []
        self.hp_true_avg_power = []
        for hour in range(48):
            start_ms = self.flo_start_ms + hour*3600*1000
            end_ms = self.flo_start_ms + (hour+1)*3600*1000

            channels = {}
            channels['energy-instruction'] = {'times': [], 'values': []}

            for message in [x for x in self.selected_messages 
                            if f'{self.house_alias}' in x.from_alias
                            and x.message_persisted_ms >= start_ms-5*60*1000
                            and x.message_persisted_ms <= end_ms+5*60*1000]:
                if message.message_type_name == 'report':
                    for channel in message.payload['ChannelReadingList']:                    
                        channel_name = channel['ChannelName']
                        # Store the values and times for the channel
                        if (channel_name in ['hp-idu-pwr', 'hp-odu-pwr']):
                            if channel_name not in channels:
                                channels[channel_name] = {
                                    'values': channel['ValueList'],
                                    'times': channel['ScadaReadTimeUnixMsList']
                                }
                            else:
                                channels[channel_name]['values'].extend(channel['ValueList'])
                                channels[channel_name]['times'].extend(channel['ScadaReadTimeUnixMsList'])
                if message.message_type_name == 'energy.instruction':
                    channels['energy-instruction']['times'].append(message.payload['SlotStartS'])
                    channels['energy-instruction']['values'].append(message.payload['AvgPowerWatts']/1000)
                                    
            for key in channels.keys():
                if key == 'energy-instruction' and not channels['energy-instruction']['values']:
                    continue
                sorted_times_values = sorted(zip(channels[key]['times'], channels[key]['values']))
                sorted_times, sorted_values = zip(*sorted_times_values)
                channels[key]['values'] = list(sorted_values)
                channels[key]['times'] = list(sorted_times)

            if channels['energy-instruction']['values']:
                self.energy_instructions_avg_power.append(channels['energy-instruction']['values'][0])
            else:
                self.energy_instructions_avg_power.append(np.nan)

            # -----------------------------------------
            # Average power out - HP
            # -----------------------------------------

            timestep_seconds = 1
            num_points = int((end_ms - start_ms) / (timestep_seconds * 1000) + 1)

            csv_times = np.linspace(start_ms, end_ms, num_points)
            csv_times_dt = pd.to_datetime(csv_times, unit='ms', utc=True)
            csv_times_dt = [x.tz_convert(self.timezone_str).replace(tzinfo=None) for x in csv_times_dt]

            csv_values = {}
            for channel in [x for x in channels if 'hp' in x or 'primary' in x]:

                channels[channel]['times'] = pd.to_datetime(channels[channel]['times'], unit='ms', utc=True)
                channels[channel]['times'] = [x.tz_convert(self.timezone_str) for x in channels[channel]['times']]
                channels[channel]['times'] = [x.replace(tzinfo=None) for x in channels[channel]['times']]
                
                merged = pd.merge_asof(
                    pd.DataFrame({'times': csv_times_dt}),
                    pd.DataFrame(channels[channel]),
                    on='times',
                    direction='backward')
                csv_values[channel] = list(merged['values'])

            try:
                df = pd.DataFrame(csv_values)
                df['hp_power'] = df['hp-idu-pwr'] + df['hp-odu-pwr']
                self.hp_true_avg_power.append(round(np.mean(df['hp_power'])/1000,2))
                if PRINT: print(f"The HP avg power was {self.hp_true_avg_power[-1]} kWh at hour {hour}")
            except Exception as e:
                self.hp_true_avg_power.append(np.nan)
                if PRINT: print(f"Could not find HP avg power: {e}") 

        start_time = pendulum.from_timestamp(self.flo_start_ms/1000, tz=self.timezone_str)
        hours = [(start_time+timedelta(hours=i)) for i in range(60)]
        plt.figure(figsize=(13,4))
        plt.step(hours[:len(self.hp_true_avg_power)], self.hp_true_avg_power, where='post', 
                 color='tab:blue', alpha=0.7, label='Real average power')
        plt.step(hours[:len(self.energy_instructions_avg_power)], self.energy_instructions_avg_power, 
                 where='post', color='tab:blue', alpha=0.5, linestyle='dashed', label='EnergyInstruction')
        plt.ylabel('HP power [kW]')
        plt.legend(loc='upper right')
        plt.ylim([-1, 15])
        plt.show()
        

    def get_hp_and_tanks_energy_out(self):
        if self.selected_messages is None:
            self.get_selected_messages()
        self.tanks_energy_out = []
        self.hp_true = []
        for hour in range(48):
            start_ms = self.flo_start_ms + hour*3600*1000
            end_ms = self.flo_start_ms + (hour+1)*3600*1000

            channels = {}
            for message in [x for x in self.selected_messages 
                            if f'{self.house_alias}' in x.from_alias
                            and x.message_persisted_ms >= start_ms-5*60*1000
                            and x.message_persisted_ms <= end_ms+5*60*1000]:
                if message.message_type_name == 'report':
                    for channel in message.payload['ChannelReadingList']:
                        channel_name = channel['ChannelName']
                        # Store the values and times for the channel
                        if ((('buffer-depth' in channel_name or ('tank' in channel_name and 'depth' in channel_name)) and 'micro' not in channel_name)
                            or channel_name in ['hp-ewt', 'hp-lwt', 'primary-flow']):
                            if channel_name not in channels:
                                channels[channel_name] = {
                                    'values': channel['ValueList'],
                                    'times': channel['ScadaReadTimeUnixMsList']
                                }
                            else:
                                channels[channel_name]['values'].extend(channel['ValueList'])
                                channels[channel_name]['times'].extend(channel['ScadaReadTimeUnixMsList'])
            
            for key in channels.keys():
                sorted_times_values = sorted(zip(channels[key]['times'], channels[key]['values']))
                sorted_times, sorted_values = zip(*sorted_times_values)
                channels[key]['values'] = list(sorted_values)
                channels[key]['times'] = list(sorted_times)

            # -----------------------------------------
            # Energy out - HP
            # -----------------------------------------

            timestep_seconds = 1
            num_points = int((end_ms - start_ms) / (timestep_seconds * 1000) + 1)

            csv_times = np.linspace(start_ms, end_ms, num_points)
            csv_times_dt = pd.to_datetime(csv_times, unit='ms', utc=True)
            csv_times_dt = [x.tz_convert(self.timezone_str).replace(tzinfo=None) for x in csv_times_dt]

            csv_values = {}
            for channel in [x for x in channels if 'hp' in x or 'primary' in x]:

                channels[channel]['times'] = pd.to_datetime(channels[channel]['times'], unit='ms', utc=True)
                channels[channel]['times'] = [x.tz_convert(self.timezone_str) for x in channels[channel]['times']]
                channels[channel]['times'] = [x.replace(tzinfo=None) for x in channels[channel]['times']]
                
                merged = pd.merge_asof(
                    pd.DataFrame({'times': csv_times_dt}),
                    pd.DataFrame(channels[channel]),
                    on='times',
                    direction='backward')
                csv_values[channel] = list(merged['values'])

            try:
                df = pd.DataFrame(csv_values)
                df['lift_C'] = df['hp-lwt'] - df['hp-ewt']
                df['lift_C'] = [x/1000 if x>0 else 0 for x in list(df['lift_C'])]
                df['flow_kgs'] = df['primary-flow'] / 100 / 60 * 3.78541 
                df['heat_power_kW'] = [m*4187*T/1000  for T,m in zip(df.lift_C, df.flow_kgs)]
                df['cumulative_energy_kWh'] = df['heat_power_kW'].cumsum()
                df['cumulative_energy_kWh'] = df['cumulative_energy_kWh'] / 3600 * timestep_seconds
                self.hp_true.append(round(list(df['cumulative_energy_kWh'])[-1] - list(df['cumulative_energy_kWh'])[0],2))
                if PRINT: print(f"The HP gave out {self.hp_true[-1]} kWh at hour {hour}")
            except Exception as e:
                self.hp_true.append(np.nan)
                if PRINT: print(f"Could not find HP energy: {e}")            

            # -----------------------------------------
            # Energy out - tanks
            # -----------------------------------------

            # Buffer energy
            first_values_buffer, first_times_buffer = [], []
            last_values_buffer, last_times_buffer = [], []
            for buffer_key in [x for x in channels if 'buffer' in x]:
                # Find the value closest to the start
                differences = [abs(time - start_ms) for time in channels[buffer_key]['times']]
                if not differences:
                    print("No data found")
                    self.tanks_energy_out.append(np.nan)
                    continue
                closest_index = differences.index(min(differences))
                first_values_buffer.append(channels[buffer_key]['values'][closest_index])
                first_times_buffer.append(channels[buffer_key]['times'][closest_index])
                # Find the value closest to the end
                differences = [abs(time - end_ms) for time in channels[buffer_key]['times']]
                if not differences:
                    print("No data found")
                    self.tanks_energy_out.append(np.nan)
                    continue
                closest_index = differences.index(min(differences))
                last_values_buffer.append(channels[buffer_key]['values'][closest_index])
                last_times_buffer.append(channels[buffer_key]['times'][closest_index])
            if last_times_buffer and first_times_buffer:
                if last_times_buffer[-1] - first_times_buffer[-1] < 50*60*1000:
                    if PRINT: 
                        print("Not enough time between first and last value:")
                        print(f"-First: {pendulum.from_timestamp(first_times_buffer[-1]/1000, tz=self.timezone_str)}")
                        print(f"-Last: {pendulum.from_timestamp(last_times_buffer[-1]/1000, tz=self.timezone_str)}")
                    self.tanks_energy_out.append(np.nan)
                    continue
            if len(first_values_buffer) != 4 or len(last_values_buffer) != 4:
                if PRINT: 
                    print("Some buffer temperatures are missing, try another day/period")
                self.tanks_energy_out.append(np.nan)
                continue
            else:
                first_values_buffer = [x/1000 for x in first_values_buffer]
                last_values_buffer = [x/1000 for x in last_values_buffer]
                buffer_avg_before = sum(first_values_buffer)/4
                buffer_avg_after = sum(last_values_buffer)/4
                buffer_energy_used = 120 * 3.785 * 4.187/3600 * (buffer_avg_before - buffer_avg_after)
                # print(f"Buffer before: {[to_fahrenheit(x) for x in first_values_buffer]}")
                # print(f"Buffer after: {[to_fahrenheit(x) for x in last_values_buffer]}")
                if PRINT: print(f"Buffer used: {buffer_energy_used}")

            # Storage energy
            first_values_store, first_times_store = [], []
            last_values_store, last_times_store = [], []
            for store_key in [x for x in channels if 'tank' in x]:
                # Get the closest value to start on onpeak
                differences = [abs(time - start_ms) for time in channels[store_key]['times']]
                if not differences:
                    print("No data found")
                    self.tanks_energy_out.append(np.nan)
                    continue
                closest_index = differences.index(min(differences))
                first_values_store.append(channels[store_key]['values'][closest_index])
                first_times_store.append(channels[store_key]['times'][closest_index])
                # Get the closest value to end of onpeak
                differences = [abs(time - end_ms) for time in channels[store_key]['times']]
                if not differences:
                    print("No data found")
                    self.tanks_energy_out.append(np.nan)
                    continue
                closest_index = differences.index(min(differences))
                last_values_store.append(channels[store_key]['values'][closest_index])
                last_times_store.append(channels[store_key]['times'][closest_index])

            if last_times_store and first_times_store:
                if last_times_store[-1] - first_times_store[-1] < 30*60*1000:
                    if PRINT: 
                        print("Not enough time between first and last value (store):")
                        print(f"-First: {pendulum.from_timestamp(first_times_store[-1]/1000, tz=self.timezone_str)}")
                        print(f"-Last: {pendulum.from_timestamp(last_times_store[-1]/1000, tz=self.timezone_str)}")
                    self.tanks_energy_out.append(np.nan)
                    continue
            if len(first_values_store) != 12 or len(last_values_store) != 12:
                if PRINT: 
                    print("Some storage temperatures are missing, try another day/period")
                self.tanks_energy_out.append(np.nan)
                continue
            else:
                first_values_store = [x/1000 for x in first_values_store]
                last_values_store = [x/1000 for x in last_values_store]
                store_avg_before = sum(first_values_store)/12
                store_avg_after = sum(last_values_store)/12
                store_energy_used = 3 * 120 * 3.785 * 4.187/3600 * (store_avg_before - store_avg_after)
                # print(f"Store before: {[to_fahrenheit(x) for x in first_values_store]}")
                # print(f"Store after: {[to_fahrenheit(x) for x in last_values_store]}")
                if PRINT: print(f"Store used: {store_energy_used}")

            total_energy_used = round(store_energy_used + buffer_energy_used,2)
            if PRINT: print(f"Total: {total_energy_used} kWh")
            self.tanks_energy_out.append(total_energy_used)

        if PRINT: print(self.tanks_energy_out)

    def plot(self):
        self.get_hp_forecast()
        self.get_load_forecast()
        self.get_hp_and_load_true()
        self.compare_power_instruction_vs_real()

        start_time = pendulum.from_timestamp(self.flo_start_ms/1000, tz=self.timezone_str)
        hours = [(start_time+timedelta(hours=i)) for i in range(60)]

        plt.figure(figsize=(13,4))
        plt.step(hours[:len(self.hp_true)], self.hp_true, where='post', color='tab:blue', alpha=0.7, label='HP')
        plt.step(hours[:len(self.hp_forecast)], self.hp_forecast, where='post', 
                 color='tab:blue', alpha=0.5, linestyle='dashed', label='HP predicted')
        plt.step(hours[:len(self.load_true)], self.load_true, where='post', color='tab:red', alpha=0.7, label='House losses')
        plt.step(hours[:len(self.load_forecast_true_weather)], self.load_forecast_true_weather, where='post', color='tab:orange', alpha=0.5, linestyle='dashed', label='House losses predicted with true weather')
        plt.step(hours[:len(self.load_forecast)], self.load_forecast, where='post', 
                 color='tab:red', alpha=0.5, linestyle='dashed', label='House losses predicted')
        plt.ylabel('Heat [kWh]')
        plt.legend(loc='upper right')
        plt.ylim([-1, 20])
        flo_time = pendulum.from_timestamp(self.flo_start_ms/1000, tz=self.timezone_str).replace(second=0, microsecond=0)
        plt.title(f"Prediction from FLO at {self.house_alias} at {flo_time}")
        # plt.xticks(list(range(0, 49, 2)))
        plt.show()