import io
import gc
import os
import csv
import time
import uuid
import pytz
import dotenv
import uvicorn
import zipfile
import pendulum
import traceback
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Optional, Union
import asyncio
import async_timeout
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
from sqlalchemy import asc, or_, and_, desc, cast, BigInteger
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.future import select
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.colors as pc
from datetime import datetime
from pathlib import Path
from config import Settings
from models import MessageSql
from named_types import FloParamsHouse0
from flo import DGraph
from dgraph_visualizer import DGraphVisualizer


class Prices(BaseModel):
    unix_s: List[float]
    lmp: List[float]
    dist: List[float]
    energy: List[float]
    
class BaseRequest(BaseModel):
    house_alias: str
    password: str
    unique_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    def __hash__(self):
        return hash(self.unique_id)

class DataRequest(BaseRequest):
    start_ms: int
    end_ms: int
    selected_channels: List[str]
    confirm_with_user: Optional[bool] = False
    darkmode: Optional[bool] = False

class CsvRequest(DataRequest):
    timestep: int

class MessagesRequest(DataRequest):
    selected_message_types: List[str]

class DijkstraRequest(BaseRequest):
    time_ms: int


class VisualizerApi():
    def __init__(self):
        self.settings = Settings(_env_file=dotenv.find_dotenv())
        engine = create_async_engine(self.settings.db_url.get_secret_value())
        self.running_locally = self.settings.running_locally
        self.AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
        self.admin_user_password = self.settings.visualizer_api_password.get_secret_value()
        self.timezone_str = 'America/New_York'
        self.timeout_seconds = 5*60
        self.top_states_order = ['HomeAlone', 'Atn', 'Dormant']
        self.ha_states_order = [
            'HpOffStoreDischarge', 'HpOffStoreOff', 'HpOnStoreOff', 
            'HpOnStoreCharge', 'StratBoss', 'Initializing', 'Dormant'
            ]
        self.aa_states_order = self.ha_states_order.copy()
        self.whitewire_threshold_watts = {'beech': 100, 'elm': 0.9, 'default': 20}
        self.zone_color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']*3
        self.data = {}
        self.timestamp_min_max = {}
        print(f"Running {'locally' if self.running_locally else 'on EC2'}")

    def start(self):
        self.app = FastAPI()
        self.app.add_middleware(
            CORSMiddleware,
            # TODO: allow_origins=["https://thegridelectric.github.io"]
            allow_origins=["*"], 
            allow_credentials=True,
            allow_methods=["*"],
        )
        self.app.post("/login")(self.check_password)
        self.app.post("/plots")(self.get_plots)
        self.app.post("/csv")(self.get_csv)
        self.app.post("/messages")(self.get_messages)
        self.app.post("/flo")(self.get_flo)
        self.app.post("/aggregate-plot")(self.get_aggregate_plot)
        self.app.post("/prices")(self.receive_prices)
        uvicorn.run(self.app, host="0.0.0.0", port=8000)

    def to_datetime(self, time_ms):
        return pendulum.from_timestamp(time_ms/1000, tz=self.timezone_str)

    def to_fahrenheit(self, t):
        return t*9/5+32
    
    def to_hex(self, rgba):
        r, g, b, a = (int(c * 255) for c in rgba)
        return f'#{r:02x}{g:02x}{b:02x}'

    def check_password(self, request: BaseRequest):
        if request.password == self.admin_user_password:
            return True
        house_owner_password = getattr(self.settings, f"{request.house_alias}_owner_password", None)
        if house_owner_password:
            house_owner_password.get_secret_value()
            if request.password == house_owner_password:
                return True
        return False
    
    def check_request(self, request: BaseRequest, aggregate=False):
        if not self.check_password(request):
            return {"success": False, "message": "Wrong password.", "reload": False}
        if isinstance(request, Union[DataRequest, CsvRequest]) and not request.confirm_with_user:
            if (request.end_ms - request.start_ms)/1000/3600/24 > 3:
                warning_message = f"That's a lot of data! Are you sure you want to proceed?"
                return {"success": False, "message": warning_message, "reload": False, "confirm_with_user": True}
        if not self.running_locally: 
            if isinstance(request, DataRequest) and not isinstance(request, CsvRequest): 
                if (request.end_ms-request.start_ms)/1000/3600/24 > 5:
                    warning_message = "Plotting data for this many days is not permitted. Please reduce the range and try again."
                    return {"success": False, "message": warning_message, "reload": False}
                if aggregate and (request.end_ms-request.start_ms)/1000/3600/24 > 2:
                    warning_message = "Plotting data for this many days is not permitted. Please reduce the range and try again."
                    return {"success": False, "message": warning_message, "reload": False}
            if isinstance(request, CsvRequest) and (request.end_ms-request.start_ms)/1000/3600/24 > 21:
                warning_message = "Downloading data for this many days is not permitted. Please reduce the range and try again."
                return {"success": False, "message": warning_message, "reload": False}
            if isinstance(request, MessagesRequest) and (request.end_ms-request.start_ms)/1000/3600/24 > 31:
                warning_message = "Downloading messages for this many days is not permitted. Please reduce the range and try again."
                return {"success": False, "message": warning_message, "reload": False}
        return None
    
    async def receive_prices(self, request: Prices):
        try:
            rows = []
            project_dir = os.path.dirname(os.path.abspath(__file__))
            elec_file = os.path.join(project_dir, 'price_forecast_dates.csv')
            with open(elec_file, mode='r', newline='') as file:
                reader = csv.reader(file)
                header = next(reader)
                rows = list(reader)

            updated_prices = {float(timestamp): (lmp, dist) 
                            for timestamp, lmp, dist in zip(request.unix_s, request.lmp, request.dist)}

            # Update the rows based on the new prices
            for row in rows:
                try:
                    unix_timestamp = float(row[0])
                    if unix_timestamp in updated_prices:
                        lmp, dist = updated_prices[unix_timestamp]
                        row[1] = dist
                        row[2] = lmp
                except Exception as e:
                    print(f"Error processing row {row}: {e}")
                    continue

            with open(elec_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerows(rows)
            print(f"Prices updated successfully in {elec_file}")

        except Exception as e:
            print(f"Error updating prices: {e}")

    async def get_data(self, request: BaseRequest):
        try:
            error = self.check_request(request)
            if error or request.selected_channels==['bids']:
                if error: print(error)
                return error
            
            self.data[request] = {}
            async with self.AsyncSessionLocal() as session:
                import time
                query_start = time.time()
                print("Querying journaldb...")
                
                # Use select() instead of session.query()
                stmt = select(MessageSql).filter(
                    MessageSql.from_alias.like(f'%.{request.house_alias}.%'),
                    MessageSql.message_persisted_ms <= cast(int(request.end_ms), BigInteger),
                    or_(
                        and_(
                            or_(
                                MessageSql.message_type_name == "batched.readings",
                                MessageSql.message_type_name == "report",
                            ),
                            MessageSql.message_persisted_ms >= cast(int(request.start_ms), BigInteger),
                        ),
                        and_(
                            MessageSql.message_type_name == "snapshot.spaceheat",
                            MessageSql.message_persisted_ms >= cast(int(request.end_ms - 10*60*1000), BigInteger),
                        ),
                        and_(
                            MessageSql.message_type_name == "weather.forecast",
                            MessageSql.message_persisted_ms >= cast(int(request.start_ms - 24 * 3600 * 1000), BigInteger),
                        )
                    )
                ).order_by(asc(MessageSql.message_persisted_ms))
                
                # Execute the statement asynchronously
                result = await session.execute(stmt)
                all_raw_messages = result.scalars().all()  # Use scalars() to retrieve the data
                
                print(f"Time to fetch data: {round(time.time() - query_start, 1)} seconds")

            if not all_raw_messages:
                warning_message = f"No data found for house '{request.house_alias}' in the selected timeframe."
                return {"success": False, "message": warning_message, "reload": False}
            
            # Process reports
            reports: List[MessageSql] = sorted(
                [x for x in all_raw_messages if x.message_type_name in ['report', 'batched.readings']],
                key = lambda x: x.message_persisted_ms
                )
            self.data[request]['channels'] = {}
            for message in reports:
                for channel in message.payload['ChannelReadingList']:
                    if message.message_type_name == 'report':
                        channel_name = channel['ChannelName']
                    elif message.message_type_name == 'batched.readings':
                        for dc in message.payload['DataChannelList']:
                            if dc['Id'] == channel['ChannelId']:
                                channel_name = dc['Name']
                    if not channel['ValueList'] or not channel['ScadaReadTimeUnixMsList']:
                        continue
                    if len(channel['ValueList'])!=len(channel['ScadaReadTimeUnixMsList']):
                        continue
                    if channel_name not in self.data[request]['channels']:
                        self.data[request]['channels'][channel_name] = {'values': [], 'times': []}
                    self.data[request]['channels'][channel_name]['values'].extend(channel['ValueList'])
                    self.data[request]['channels'][channel_name]['times'].extend(channel['ScadaReadTimeUnixMsList'])
                
            # Process snapshots
            max_timestamp = max(max(self.data[request]['channels'][channel_name]['times']) for channel_name in self.data[request]['channels'])
            snapshots = sorted(
                    [x for x in all_raw_messages if x.message_type_name=='snapshot.spaceheat'
                    and x.message_persisted_ms >= max_timestamp], 
                    key = lambda x: x.message_persisted_ms
                    )
            for snapshot in snapshots:
                for snap in snapshot.payload['LatestReadingList']:
                    if snap['ChannelName'] in self.data[request]['channels']:
                        self.data[request]['channels'][snap['ChannelName']]['times'].append(snap['ScadaReadTimeUnixMs'])
                        self.data[request]['channels'][snap['ChannelName']]['values'].append(snap['Value'])

            # Get minimum and maximum timestamp for plots
            max_timestamp = max(max(self.data[request]['channels'][x]['times']) for x in self.data[request]['channels'])
            min_timestamp = min(min(self.data[request]['channels'][x]['times']) for x in self.data[request]['channels'])
            min_timestamp += -(max_timestamp-min_timestamp)*0.05
            max_timestamp += (max_timestamp-min_timestamp)*0.05
            self.data[request]['min_timestamp'] = self.to_datetime(min_timestamp)
            self.data[request]['max_timestamp'] = self.to_datetime(max_timestamp)

            # Sort values according to time and convert to datetime
            for channel_name in self.data[request]['channels'].keys():
                sorted_times_values = sorted(zip(self.data[request]['channels'][channel_name]['times'], self.data[request]['channels'][channel_name]['values']))
                sorted_times, sorted_values = zip(*sorted_times_values)
                self.data[request]['channels'][channel_name]['values'] = list(sorted_values)
                self.data[request]['channels'][channel_name]['times'] = pd.to_datetime(list(sorted_times), unit='ms', utc=True)
                self.data[request]['channels'][channel_name]['times'] = self.data[request]['channels'][channel_name]['times'].tz_convert(self.timezone_str)
                self.data[request]['channels'][channel_name]['times'] = [x.replace(tzinfo=None) for x in self.data[request]['channels'][channel_name]['times']]        

            # Find all zone channels
            self.data[request]['channels_by_zone'] = {}
            for channel_name in self.data[request]['channels'].keys():
                if 'zone' in channel_name and 'gw-temp' not in channel_name:
                    zone_number = channel_name.split('-')[0]
                    if zone_number not in self.data[request]['channels_by_zone']:
                        self.data[request]['channels_by_zone'][zone_number] = {}
                    if 'state' in channel_name:
                        self.data[request]['channels_by_zone'][zone_number]['state'] = channel_name
                    elif 'whitewire' in channel_name:
                        self.data[request]['channels_by_zone'][zone_number]['whitewire'] = channel_name
                    elif 'temp' in channel_name:
                        self.data[request]['channels_by_zone'][zone_number]['temp'] = channel_name
                    elif 'set' in channel_name:
                        self.data[request]['channels_by_zone'][zone_number]['set'] = channel_name

            # Relays
            relays = {}
            for message in reports:
                if 'StateList' not in message.payload:
                    continue
                for state in message.payload['StateList']:
                    if state['MachineHandle'] not in relays:
                        relays[state['MachineHandle']] = {'times': [], 'values': []}
                    relays[state['MachineHandle']]['times'].extend([self.to_datetime(x) for x in state['UnixMsList']])
                    relays[state['MachineHandle']]['values'].extend(state['StateList'])

            # Top state
            self.data[request]['top_states'] = {'all': {'times':[], 'values':[]}}
            if 'auto' in relays:
                for time, state in zip(relays['auto']['times'], relays['auto']['values']):
                    if state not in self.top_states_order:
                        print(f"Warning: {state} is not a known top state")
                        continue
                    if state not in self.data[request]['top_states']:
                        self.data[request]['top_states'][state] = {'times':[], 'values':[]}
                    self.data[request]['top_states']['all']['times'].append(time)
                    self.data[request]['top_states']['all']['values'].append(self.top_states_order.index(state))
                    self.data[request]['top_states'][state]['times'].append(time)
                    self.data[request]['top_states'][state]['values'].append(self.top_states_order.index(state))
            if "Dormant" in self.data[request]['top_states']:
                self.data[request]['top_states']['Admin'] = self.data[request]['top_states']['Dormant']
                del self.data[request]['top_states']['Dormant']
            
            # HomeAlone state
            self.data[request]['ha_states'] = {'all': {'times':[], 'values':[]}}
            if 'auto.h.n' in relays or 'auto.h' in relays:
                ha_handle = 'auto.h.n' if 'auto.h.n' in relays else 'auto.h'
                for time, state in zip(relays[ha_handle]['times'], relays[ha_handle]['values']):
                    if state not in self.ha_states_order:
                        print(f"Warning: {state} is not a known HA state")
                        continue
                    if state not in self.data[request]['ha_states']:
                        self.data[request]['ha_states'][state] = {'times':[], 'values':[]}
                    self.data[request]['ha_states']['all']['times'].append(time)
                    self.data[request]['ha_states']['all']['values'].append(self.ha_states_order.index(state))
                    self.data[request]['ha_states'][state]['times'].append(time)
                    self.data[request]['ha_states'][state]['values'].append(self.ha_states_order.index(state))

            # AtomicAlly state
            self.data[request]['aa_states'] = {'all': {'times':[], 'values':[]}}
            if 'a.aa' in relays:
                for time, state in zip(relays['a.aa']['times'], relays['a.aa']['values']):
                    if state not in self.aa_states_order:
                        print(f"Warning: {state} is not a known AA state")
                        continue
                    if state not in self.data[request]['aa_states']:
                        self.data[request]['aa_states'][state] = {'times':[], 'values':[]}
                    self.data[request]['aa_states']['all']['times'].append(time)
                    self.data[request]['aa_states']['all']['values'].append(self.aa_states_order.index(state))
                    self.data[request]['aa_states'][state]['times'].append(time)
                    self.data[request]['aa_states'][state]['values'].append(self.aa_states_order.index(state))

            # Weather forecasts
            weather_forecasts: List[MessageSql] = []
            if isinstance(request, DataRequest):
                weather_forecasts = sorted(
                    [x for x in all_raw_messages if x.message_type_name=='weather.forecast'], 
                    key = lambda x: x.message_persisted_ms
                    )
            self.data[request]['weather_forecasts'] = weather_forecasts.copy()
            return None
        except Exception as e:
            print(f"An error occurred in get_data():\n{traceback.format_exc()}")
            return {"success": False, "message": "An error occurred when getting data", "reload": False}
        
    async def get_aggregate_data(self, request: DataRequest):
        try:
            error = self.check_request(request, aggregate=True)
            if error:
                print(error)
                return error
            
            self.data[request] = {}
            self.timestamp_min_max[request] = {}
            async with self.AsyncSessionLocal() as session:
                import time
                query_start = time.time()
                print("Querying journaldb...")
                stmt = select(MessageSql).filter(
                    MessageSql.from_alias.like(f'%.scada%'),
                    or_(
                        MessageSql.message_type_name == "batched.readings",
                        MessageSql.message_type_name == "report",
                        MessageSql.message_type_name == "snapshot.spaceheat",
                    ),
                    MessageSql.message_persisted_ms >= request.start_ms,
                    MessageSql.message_persisted_ms <= request.end_ms + 10*60*1000,
                ).order_by(asc(MessageSql.message_persisted_ms))

                result = await session.execute(stmt)  # Use async execute
                all_raw_messages = result.scalars().all()  # Get the results
                print(f"Time to fetch data: {round(time.time()-query_start,1)} seconds")

            if not all_raw_messages:
                warning_message = f"No data found for the aggregation in the selected timeframe."
                return {"success": False, "message": warning_message, "reload": False}
            
            for house_alias in set([message.from_alias for message in all_raw_messages]):
                self.data[request][house_alias] = {}

                # Process reports
                reports: List[MessageSql] = sorted([
                    x for x in all_raw_messages 
                    if x.message_type_name in ['report', 'batched.readings']
                    and x.from_alias == house_alias
                    ], key = lambda x: x.message_persisted_ms
                    )
                self.data[request][house_alias] = {}
                for message in reports:
                    for channel in message.payload['ChannelReadingList']:
                        if message.message_type_name == 'report':
                            channel_name = channel['ChannelName']
                        elif message.message_type_name == 'batched.readings':
                            for dc in message.payload['DataChannelList']:
                                if dc['Id'] == channel['ChannelId']:
                                    channel_name = dc['Name']
                        if not channel['ValueList'] or not channel['ScadaReadTimeUnixMsList']:
                            continue
                        if len(channel['ValueList'])!=len(channel['ScadaReadTimeUnixMsList']):
                            continue
                        if ((channel_name not in ['hp-idu-pwr', 'hp-odu-pwr'] and 'depth' not in channel_name) 
                            or 'micro' in channel_name):
                            continue
                        if channel_name not in self.data[request][house_alias]:
                            self.data[request][house_alias][channel_name] = {'values': [], 'times': []}
                        self.data[request][house_alias][channel_name]['values'].extend(channel['ValueList'])
                        self.data[request][house_alias][channel_name]['times'].extend(channel['ScadaReadTimeUnixMsList'])

                # Process snapshots
                max_timestamp = max(max(self.data[request][house_alias][channel_name]['times']) for channel_name in self.data[request][house_alias])
                snapshots = sorted(
                        [x for x in all_raw_messages if x.message_type_name=='snapshot.spaceheat'
                        and x.message_persisted_ms >= max_timestamp], 
                        key = lambda x: x.message_persisted_ms
                        )
                for snapshot in snapshots:
                    for snap in snapshot.payload['LatestReadingList']:
                        if snap['ChannelName'] in self.data[request][house_alias]:
                            self.data[request][house_alias][snap['ChannelName']]['times'].append(snap['ScadaReadTimeUnixMs'])
                            self.data[request][house_alias][snap['ChannelName']]['values'].append(snap['Value'])

                # Get minimum and maximum timestamp for plots
                max_timestamp = max(max(self.data[request][house_alias][x]['times']) for x in self.data[request][house_alias])
                min_timestamp = min(min(self.data[request][house_alias][x]['times']) for x in self.data[request][house_alias])
                min_timestamp += -(max_timestamp-min_timestamp)*0.05
                max_timestamp += (max_timestamp-min_timestamp)*0.05
                self.timestamp_min_max[request][house_alias] = {
                    'min_timestamp': self.to_datetime(min_timestamp),
                    'max_timestamp': self.to_datetime(max_timestamp+5*60*60*1000)
                }

                # Sort values according to time and convert to datetime
                for channel_name in self.data[request][house_alias].keys():
                    sorted_times_values = sorted(zip(self.data[request][house_alias][channel_name]['times'], self.data[request][house_alias][channel_name]['values']))
                    sorted_times, sorted_values = zip(*sorted_times_values)
                    self.data[request][house_alias][channel_name]['values'] = list(sorted_values)
                    self.data[request][house_alias][channel_name]['times'] = pd.to_datetime(list(sorted_times), unit='ms', utc=True)
                    self.data[request][house_alias][channel_name]['times'] = self.data[request][house_alias][channel_name]['times'].tz_convert(self.timezone_str)
                    self.data[request][house_alias][channel_name]['times'] = [x.replace(tzinfo=None) for x in self.data[request][house_alias][channel_name]['times']]        
                
            # Re-sample to equal timesteps
            print("Re-sampling...")
            start_ms = request.start_ms
            end_ms = request.end_ms + (10*60*1000 if query_start-request.end_ms/1000>10*60 else 0)
            timestep_s = 30
            num_points = int((end_ms - start_ms) / (timestep_s * 1000) + 1)
            sampling_times = np.linspace(start_ms, end_ms, num_points)
            sampling_times = pd.to_datetime(sampling_times, unit='ms', utc=True)
            sampling_times = [x.tz_convert(self.timezone_str).replace(tzinfo=None) for x in sampling_times]

            agg_data = {}
            for house_alias in [x for x in self.data[request] if 'maple' not in x]:
                agg_data[house_alias] = {'timestamps': sampling_times}
                for channel in self.data[request][house_alias]:
                    sampled = await asyncio.to_thread(
                        pd.merge_asof, 
                        pd.DataFrame({'times': sampling_times}),
                        pd.DataFrame(self.data[request][house_alias][channel]),
                        on='times', 
                        direction='backward'
                        )
                    sampled['values'] = sampled['values'].bfill()
                    agg_data[house_alias][channel] = list(sampled['values'])

                # Compute average temperature and energy
                temperature_channels = [value for key, value in agg_data[house_alias].items() if 'depth' in key]
                num_lists = len(temperature_channels)
                num_elements = len(temperature_channels[0])
                sums = [0] * num_elements
                for channel in temperature_channels:
                    for i in range(num_elements):
                        sums[i] += channel[i]
                averaged_temperature = [sum_value / num_lists for sum_value in sums]
                m_total_kg = 120*4*3.785
                agg_data[house_alias]['energy'] = [m_total_kg*4.187/3600*(avg_temp/1000-30) for avg_temp in averaged_temperature]
                agg_data[house_alias] = {k: v for k, v in agg_data[house_alias].items() if 'depth' not in k}
            
            energy_list, hp_list = [], []
            for i in range(len(agg_data[house_alias]['energy'])):
                energy_list.append(sum([agg_data[ha]['energy'][i] for ha in agg_data]))
                hp_list.append(sum([(agg_data[ha]['hp-idu-pwr'][i]+agg_data[ha]['hp-odu-pwr'][i])/1000 for ha in agg_data]))
            # Remove the last minutes of the energy plot to avoid wierd behaviour
            energy_list = [
                energy if t<datetime.fromtimestamp(end_ms/1000-10*60,pytz.timezone(self.timezone_str)).replace(tzinfo=None) else np.nan
                for t, energy in zip(sampling_times, energy_list)
                ]
            hp_list = [
                power if t<=datetime.fromtimestamp(end_ms/1000-10*60,pytz.timezone(self.timezone_str)).replace(tzinfo=None) else np.nan
                for t, power in zip(sampling_times, hp_list)
                ]
            self.data[request] = {'timestamp': sampling_times, 'hp':hp_list, 'energy': energy_list}
            print("Done.")

        except Exception as e:
            print(f"An error occurred in get_aggregate_data():\n{traceback.format_exc()}")
            return {"success": False, "message": "An error occurred when getting aggregate data", "reload": False}
    
    async def get_messages(self, request: MessagesRequest):
        print("Recieved message request")
        try:
            error = self.check_request(request)
            if error:
                print(error)
                return error
            async with async_timeout.timeout(self.timeout_seconds):
                print("Querying journaldb for messages...")

                async with self.AsyncSessionLocal() as session:
                    stmt = select(MessageSql).filter(
                        MessageSql.from_alias.like(f"%{f'.{request.house_alias}.' if request.house_alias else ''}%"),
                        MessageSql.message_type_name.in_(request.selected_message_types),
                        MessageSql.message_persisted_ms >= request.start_ms,
                        MessageSql.message_persisted_ms <= request.end_ms,
                    ).order_by(asc(MessageSql.message_persisted_ms))

                    result = await session.execute(stmt)
                    messages: List[MessageSql] = result.scalars().all()

                if not messages:
                    print("No messages found.")
                    return {"success": False, "message": f"No data found.", "reload":False}
                
                # Collecting all messages
                levels = {'critical': 1, 'error': 2, 'warning': 3, 'info': 4, 'debug': 5, 'trace': 6}
                sources, pb_types, summaries, details, times_created = [], [], [], [], []
                
                # Problem Events
                sorted_problem_types = sorted(
                    [m for m in messages if m.message_type_name == 'gridworks.event.problem'],
                    key=lambda x: (levels[x.payload['ProblemType']], x.payload['TimeCreatedMs'])
                )
                for message in sorted_problem_types:
                    source = message.payload['Src']
                    if ".scada" in source and source.split('.')[-1] in ['scada', 's2']:
                        source = source.split('.scada')[0].split('.')[-1]
                    sources.append(source)
                    pb_types.append(message.payload['ProblemType'])
                    summaries.append(message.payload['Summary'])
                    details.append(message.payload['Details'].replace('<','').replace('>','').replace('\n','<br>'))
                    times_created.append(str(self.to_datetime(message.payload['TimeCreatedMs']).replace(microsecond=0)))
                
                # Glitches
                sorted_glitches = sorted(
                    [m for m in messages if m.message_type_name == 'glitch'],
                    key=lambda x: (levels[str(x.payload['Type']).lower()], x.payload['CreatedMs'])
                )
                for message in sorted_glitches:
                    source = message.payload['FromGNodeAlias']
                    if ".scada" in source and source.split('.')[-1] in ['scada', 's2']:
                        source = source.split('.scada')[0].split('.')[-1]
                    sources.append(source)
                    pb_types.append(str(message.payload['Type']).lower())
                    summaries.append(message.payload['Summary'])
                    details.append(message.payload['Details'].replace('<','').replace('>','').replace('\n','<br>'))
                    times_created.append(str(self.to_datetime(message.payload['CreatedMs']).replace(microsecond=0)))
                
                summary_table = {
                    'critical': str(len([x for x in pb_types if x=='critical'])),
                    'error': str(len([x for x in pb_types if x=='error'])),
                    'warning': str(len([x for x in pb_types if x=='warning'])),
                    'info': str(len([x for x in pb_types if x=='info'])),
                    'debug': str(len([x for x in pb_types if x=='debug'])),
                    'trace': str(len([x for x in pb_types if x=='trace'])),
                }
                for key in summary_table.keys():
                    if summary_table[key]=='0':
                        summary_table[key]=''

                return {
                    "Log level": pb_types,
                    "From node": sources,
                    "Summary": summaries,
                    "Details": details,
                    "Time created": times_created,
                    "SummaryTable": summary_table
                }
            
        except asyncio.TimeoutError:
            print("Timed out in get_messages()")
            return {"success": False, "message": "The request timed out.", "reload": False}
        except Exception as e:
            print(f"An error occurred in get_messages():\n{traceback.format_exc()}")
            return {"success": False, "message": "An error occurred while getting messages", "reload": False}
        
    async def get_csv(self, request: CsvRequest):
        try:
            async with async_timeout.timeout(self.timeout_seconds):
                error = await self.get_data(request)
                if error:
                    print(error)
                    return error
                
                # Find the channels to export
                if 'all-data' in request.selected_channels:
                    channels_to_export = list(self.data[request]['channels'].keys())
                else:
                    channels_to_export = []
                    for channel in request.selected_channels:
                        if channel in self.data[request]['channels']:
                            channels_to_export.append(channel)
                        elif channel == 'zone-heat-calls':
                            for c in self.data[request]['channels'].keys():
                                if 'zone' in c:
                                    channels_to_export.append(c)
                        elif channel == 'buffer-depths':
                            for c in self.data[request]['channels'].keys():
                                if 'depth' in c and 'buffer' in c and 'micro' not in c:
                                    channels_to_export.append(c)
                        elif channel == 'storage-depths':
                            for c in self.data[request]['channels'].keys():
                                if 'depth' in c and 'tank' in c and 'micro' not in c:
                                    channels_to_export.append(c)
                        elif channel == 'relays':
                            for c in self.data[request]['channels'].keys():
                                if 'relay' in c:
                                    channels_to_export.append(c)
                        elif channel == 'zone-heat-calls':
                            for c in self.data[request]['channels'].keys():
                                if 'zone' in c:
                                    channels_to_export.append(c)
                        elif channel == 'store-energy':
                            for c in self.data[request]['channels'].keys():
                                if 'required-energy' in c or 'available-energy':
                                    channels_to_export.append(c)

                # Check the amount of data that will be generated
                num_points = int((request.end_ms - request.start_ms) / (request.timestep * 1000) + 1)
                if num_points * len(channels_to_export) > 3600 * 24 * 10 * len(self.data[request]['channels']):
                    error_message = f"This request would generate too many data points ({num_points*len(channels_to_export)})."
                    error_message += "\n\nSuggestions:\n- Increase the time step\n- Reduce the number of channels"
                    error_message += "\n- Reduce the difference between the start and end time"
                    return {"success": False, "message": error_message, "reload": False}

                # Create the timestamps on which the data will be sampled
                csv_times = np.linspace(request.start_ms, request.end_ms, num_points)
                csv_times = pd.to_datetime(csv_times, unit='ms', utc=True)
                csv_times = [x.tz_convert(self.timezone_str).replace(tzinfo=None) for x in csv_times]
                
                # Re-sample the data to the desired time step
                print(f"Sampling data with {request.timestep}-second time step...")
                request_start = time.time()
                csv_data = {'timestamps': csv_times}
                for channel in channels_to_export:
                    sampled = await asyncio.to_thread(
                        pd.merge_asof, 
                        pd.DataFrame({'times': csv_times}),
                        pd.DataFrame(self.data[request]['channels'][channel]),
                        on='times', 
                        direction='backward'
                        )
                    csv_data[channel] = list(sampled['values'])
                df = pd.DataFrame(csv_data)
                print("Done.")

                # Build file name
                start_date = self.to_datetime(request.start_ms) 
                end_date = self.to_datetime(request.end_ms)
                formatted_start_date = start_date.to_iso8601_string()[:16].replace('T', '-')
                formatted_end_date = end_date.to_iso8601_string()[:16].replace('T', '-')
                filename = f'{request.house_alias}_{request.timestep}s_{formatted_start_date}-{formatted_end_date}.csv'.replace(':','_')

                # Send back as a CSV
                csv_buffer = io.StringIO()
                csv_buffer.write(filename+'\n')
                df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                return StreamingResponse(
                    iter([csv_buffer.getvalue()]),
                    media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={filename}"}
                )
        except asyncio.TimeoutError:
            print("Timed out in get_csv()")
            return {"success": False, "message": "The request timed out.", "reload": False}
        except Exception as e:
            print(f"An error occurred in get_csv():\n{traceback.format_exc()}")
            return {"success": False, "message": "An error occurred while getting CSV", "reload": False}
        finally:
            if request in self.data:
                del self.data[request]
                print(f"Deleted request data")
            print(f"Unfinished requests in data: {len(self.data)}")
        
    async def get_flo(self, request: DijkstraRequest):
        try:
            async with async_timeout.timeout(self.timeout_seconds):
                print("Finding latest FLO run...")
                flo_params_msg = None
                async with self.AsyncSessionLocal() as session:
                    stmt = select(MessageSql).filter(
                        MessageSql.message_type_name == "flo.params.house0",
                        MessageSql.from_alias.like(f'%{request.house_alias}%'),
                        MessageSql.message_persisted_ms >= request.time_ms - 48*3600*1000,
                        MessageSql.message_persisted_ms <= request.time_ms,
                    ).order_by(desc(MessageSql.message_persisted_ms))
                    result = await session.execute(stmt)
                    flo_params_msg: MessageSql = result.scalars().first()
                
                print(f"Found FLO run at {self.to_datetime(flo_params_msg.message_persisted_ms)}")

                if not flo_params_msg:
                    print(f"Could not find a FLO run in the 48 hours prior to {self.to_datetime(request.time_ms)}")
                    if os.path.exists('result.xlsx'):
                        os.remove('result.xlsx')
                    return

                print("Running Dijkstra and saving analysis to excel...")
                flo_params = FloParamsHouse0(**flo_params_msg.payload)
                g = DGraph(flo_params)
                g.solve_dijkstra()
                v = DGraphVisualizer(g)
                v.export_to_excel()
                del g 
                del v
                gc.collect()
                print("Done.")
                
                if os.path.exists('result.xlsx'):
                    return FileResponse(
                        'result.xlsx',
                        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        headers={"Content-Disposition": "attachment; filename=file.xlsx"}
                        )
                else:
                    return {"error": "File not found"}
        except asyncio.TimeoutError:
            print("Timed out in get_flo()")
            return {"success": False, "message": "The request timed out.", "reload": False}
        except Exception as e:
            print(f"An error occurred in get_flo():\n{traceback.format_exc()}")
            return {"success": False, "message": "An error occurred while getting FLO", "reload": False}
        finally:
            if request in self.data:
                del self.data[request]
                print(f"Deleted request data")
            print(f"Unfinished requests in data: {len(self.data)}")
        
    async def get_bids(self, request: DataRequest):
        try:
            async with async_timeout.timeout(self.timeout_seconds):
                print("Getting bids...")

                async with self.AsyncSessionLocal() as session:
                    stmt = select(MessageSql).filter(
                        MessageSql.message_type_name == "flo.params.house0",
                        MessageSql.from_alias.like(f'%{request.house_alias}%'),
                        MessageSql.message_persisted_ms >= request.start_ms,
                        MessageSql.message_persisted_ms <= request.end_ms,
                    ).order_by(desc(MessageSql.payload['StartUnixS']))
                    
                    result = await session.execute(stmt)
                    flo_params_messages = result.scalars().all()

                    flo_params_messages = [FloParamsHouse0(**x.payload) for x in flo_params_messages]
                print(f"Found {len(flo_params_messages)} FLOs for {request.house_alias}")

                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    for i in range(len(flo_params_messages)):
                        g = DGraph(flo_params_messages[i])
                        g.solve_dijkstra()
                        g.generate_bid()
                        prices = [x.PriceTimes1000 for x in g.pq_pairs]
                        quantities = [x.QuantityTimes1000/1000 for x in g.pq_pairs]
                        # To plot quantities on x-axis and prices on y-axis
                        ps, qs = [], []
                        index_p = 0
                        expected_price_usd_mwh = g.params.elec_price_forecast[0] * 10
                        for p in sorted(list(range(min(prices), max(prices)+1)) + [expected_price_usd_mwh*1000]):
                            ps.append(p/1000)
                            if index_p+1 < len(prices) and p >= prices[index_p+1]:
                                index_p += 1
                            if p == expected_price_usd_mwh*1000:
                                interesection = (quantities[index_p], expected_price_usd_mwh)
                            qs.append(quantities[index_p])
                        # Plot
                        plt.plot(qs, ps, label='demand (bid)')
                        prices = [x.PriceTimes1000/1000 for x in g.pq_pairs]
                        plt.scatter(quantities, prices)
                        plt.plot(
                            [min(quantities)-1, max(quantities)+1],[expected_price_usd_mwh]*2, 
                            label="supply (expected market price)"
                            )
                        plt.scatter(interesection[0], interesection[1])
                        plt.text(
                            interesection[0]+0.25, interesection[1]+15, 
                            f'({round(interesection[0],3)}, {round(interesection[1],1)})', 
                            fontsize=10, color='tab:orange'
                            )
                        plt.xticks(quantities)
                        if min([abs(x-expected_price_usd_mwh) for x in prices]) < 5:
                            plt.yticks(prices)
                        else:
                            plt.yticks(prices + [expected_price_usd_mwh])
                        plt.ylabel("Price [USD/MWh]")
                        plt.xlabel("Quantity [kWh]")
                        plt.title(self.to_datetime(g.params.start_time*1000).strftime('%Y-%m-%d %H:%M'))
                        plt.grid(alpha=0.3)
                        plt.legend()
                        plt.tight_layout()
                        # Append plot to zip
                        img_buf = io.BytesIO()
                        plt.savefig(img_buf, format='png', dpi=300)
                        img_buf.seek(0)
                        zip_file.writestr(f'pq_plot_{i}.png', img_buf.getvalue())
                        plt.close()

                del g
                gc.collect()
                zip_buffer.seek(0)
                return StreamingResponse(
                    zip_buffer, 
                    media_type='application/zip', 
                    headers={"Content-Disposition": "attachment; filename=plots.zip"}
                    )
        except asyncio.TimeoutError:
            print("Timed out in get_bids()")
            return {"success": False, "message": "The request timed out.", "reload": False}
        except Exception as e:
            print(f"An error occurred in get_bids():\n{traceback.format_exc()}")
            return {"success": False, "message": "An error occurred while getting bids", "reload": False}
        
    async def get_aggregate_plot(self, request: DataRequest):
        if request.selected_channels == ['prices']:
            result = await self.get_aggregate_price_plot(request)
            return result
        try:
            async with async_timeout.timeout(self.timeout_seconds):
                error = await self.get_aggregate_data(request)
                if error:
                    print(error)
                    return error
                print("No error")
                
                # Get plots, zip and return
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    print("Getting plot1...")
                    html_buffer = await self.plot_aggregate(request)
                    zip_file.writestr('plot1.html', html_buffer.read())

                    print("Getting plot2...")
                    html_buffer = await self.plot_prices(request, aggregate=True)
                    zip_file.writestr('plot2.html', html_buffer.read())

                zip_buffer.seek(0)

                return StreamingResponse(
                    zip_buffer, 
                    media_type='application/zip', 
                    headers={"Content-Disposition": "attachment; filename=plots.zip"}
                    )
        except asyncio.TimeoutError:
            print("Timed out in get_aggregate_plot()")
            return {"success": False, "message": "The request timed out.", "reload": False}
        except Exception as e:
            print(f"An error occurred in get_aggregate_plot():\n{traceback.format_exc()}")
            return {"success": False, "message": "An error occurred while getting aggregate plot", "reload": False}
        finally:
            if request in self.data:
                del self.data[request]
                print(f"Deleted request data")
            print(f"Unfinished requests in data: {len(self.data)}")
        
    async def get_aggregate_price_plot(self, request: DataRequest):
        try:    
                # Get plots, zip and return
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    html_buffer = await self.plot_prices(request, aggregate=True)
                    zip_file.writestr('plot.html', html_buffer.read())

                zip_buffer.seek(0)

                return StreamingResponse(
                    zip_buffer, 
                    media_type='application/zip', 
                    headers={"Content-Disposition": "attachment; filename=plots.zip"}
                    )
        except asyncio.TimeoutError:
            print("Timed out in get_aggregate_plot()")
            return {"success": False, "message": "The request timed out.", "reload": False}
        except Exception as e:
            print(f"An error occurred in get_aggregate_plot():\n{traceback.format_exc()}")
            return {"success": False, "message": "An error occurred while getting aggregate plot", "reload": False}
        finally:
            if request in self.data:
                del self.data[request]
                print(f"Deleted request data")
            print(f"Unfinished requests in data: {len(self.data)}")

    async def get_plots(self, request: DataRequest):
        try:
            async with async_timeout.timeout(self.timeout_seconds):
                error = await self.get_data(request)
                if error:
                    print(error)
                    return error
                
                # If the request is just to plot bids
                if request.selected_channels == ['bids']: 
                    zip_bids = await self.get_bids(request)
                    return zip_bids
                
                # Get plots, zip and return
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    html_buffer = await self.plot_heatpump(request)
                    zip_file.writestr('plot1.html', html_buffer.read())

                    html_buffer = await self.plot_prices(request)
                    zip_file.writestr('plot2.html', html_buffer.read())
                    
                    html_buffer = await self.plot_distribution(request)
                    zip_file.writestr('plot3.html', html_buffer.read())
                    
                    html_buffer = await self.plot_heatcalls(request)
                    zip_file.writestr('plot4.html', html_buffer.read())
                    
                    html_buffer = await self.plot_zones(request)
                    zip_file.writestr('plot5.html', html_buffer.read())

                    html_buffer = await self.plot_buffer(request)
                    zip_file.writestr('plot6.html', html_buffer.read())
                    
                    html_buffer = await self.plot_storage(request)
                    zip_file.writestr('plot7.html', html_buffer.read())
                    
                    html_buffer = await self.plot_top_state(request)
                    zip_file.writestr('plot8.html', html_buffer.read())
                    
                    html_buffer = await self.plot_ha_state(request)
                    zip_file.writestr('plot9.html', html_buffer.read())
                    
                    html_buffer = await self.plot_aa_state(request)
                    zip_file.writestr('plot10.html', html_buffer.read())
                    
                    html_buffer = await self.plot_weather(request)
                    zip_file.writestr('plot11.html', html_buffer.read())
                    
                zip_buffer.seek(0)

                return StreamingResponse(
                    zip_buffer, 
                    media_type='application/zip', 
                    headers={"Content-Disposition": "attachment; filename=plots.zip"}
                    )
                
        except asyncio.TimeoutError:
            print("Timed out in get_plots()")
            return {"success": False, "message": "The request timed out.", "reload": False}
        except Exception as e:
            print(f"An error occurred in get_plots():\n{traceback.format_exc()}")
            return {"success": False, "message": "An error occurred while getting plots", "reload": False}
        finally:
            if request in self.data:
                del self.data[request]
                print(f"Deleted request data")
            print(f"Unfinished requests in data: {len(self.data)}")

    async def plot_aggregate(self, request: BaseRequest):
        plot_start = time.time()
        self.data[request]['energy'] = [x-min(self.data[request]['energy']) for x in self.data[request]['energy']]
        
        df = pd.DataFrame(self.data[request])
        df['timestamp'] = df['timestamp'] - pd.Timedelta(minutes=5)
        df_resampled = df.resample('5min', on='timestamp').agg({'energy': 'mean', 'hp': 'mean'}).reset_index()
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=df_resampled['timestamp'],
                y=df_resampled['energy'],
                name='Aggregated storage',
                yaxis='y2',
                opacity=0.6 if request.darkmode else 0.2,
                marker=dict(color='#2a4ca2', line=dict(width=0)),
                hovertemplate="%{x|%H:%M:%S} | %{y:.1f} kWh<extra></extra>",
            )
        )
        # fig.add_trace(
        #     go.Bar(
        #         x=df_resampled['timestamp'], 
        #         y=[x if x>0.9 else 0 for x in list(df_resampled['hp'])], 
        #         opacity=0.7,
        #         yaxis='y2',
        #         marker=dict(color='#d62728', line=dict(width=0)),
        #         name='Aggregated load',
        #         hovertemplate="%{x|%H:%M:%S} | %{y:.1f} kW<extra></extra>",
        #         )
        #     )

        fig.add_trace(
            go.Scatter(
                x=self.data[request]['timestamp'], 
                y=self.data[request]['energy'], 
                mode='lines',
                opacity=0,
                line=dict(color='#2a4ca2', dash='solid'),
                name='Aggregated storage',
                yaxis='y2',
                hovertemplate="%{x|%H:%M:%S} | %{y:.1f} kWh<extra></extra>",
                showlegend=False
                )
            )
        
        fig.add_trace(
            go.Scatter(
                x=self.data[request]['timestamp'], 
                y=self.data[request]['hp'], 
                mode='lines',
                opacity=0.9,
                line=dict(color='#d62728', dash='solid'),
                name='Aggregated load',
                hovertemplate="%{x|%H:%M:%S} | %{y:.1f} kW<extra></extra>",
                showlegend=True,
                zorder=10
                )
            )
        fig.update_layout(yaxis=dict(title='Power [kWe]'))
        fig.update_layout(yaxis2=dict(title='Relative thermal energy [kWht]'))
        fig.update_layout(
            # title=dict(text='', x=0.5, xanchor='center'),
            margin=dict(t=30, b=30),
            plot_bgcolor='#313131' if request.darkmode else '#F5F5F7',
            paper_bgcolor='#313131' if request.darkmode else '#F5F5F7',
            font_color='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
            title_font_color='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
            xaxis=dict(
                range=[self.to_datetime(request.start_ms), self.to_datetime(request.end_ms+(
                    5*3600*1000 if time.time()-request.end_ms/1000<5*3600 else 0))],
                mirror=True,
                ticks='outside',
                showline=False,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                showgrid=False
                ),
            yaxis=dict(
                range = [0, max(self.data[request]['hp'])*1.3],
                mirror=True,
                ticks='outside',
                showline=False,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                zeroline=False,
                showgrid=False, 
                gridwidth=1, 
                gridcolor='#424242' if request.darkmode else 'LightGray'
                ),
            yaxis2=dict(
                range = [0, max(df_resampled['energy'])*1.2],
                mirror=True,
                ticks='outside',
                zeroline=False,
                showline=False,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                showgrid=False,
                overlaying='y', 
                side='right'
                ),
            legend=dict(
                x=0,
                y=1,
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(0, 0, 0, 0)'
                )
            )
        html_buffer = io.StringIO()
        fig.write_html(html_buffer, config={'displayModeBar': False})
        html_buffer.seek(0)
        print(f"Aggregation plot done in {round(time.time()-plot_start,1)} seconds")
        return html_buffer
        
        
    async def plot_heatpump(self, request: DataRequest):
        plot_start = time.time()
        fig = go.Figure()
        # Temperatures
        plotting_temperatures = False
        if 'hp-lwt' in request.selected_channels and 'hp-lwt' in self.data[request]['channels']:
            plotting_temperatures = True
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['hp-lwt']['times'], 
                    y=[self.to_fahrenheit(x/1000) for x in self.data[request]['channels']['hp-lwt']['values']], 
                    mode='lines+markers' if 'show-points'in request.selected_channels else 'lines',
                    opacity=0.7,
                    line=dict(color='#d62728', dash='solid'),
                    name='HP LWT',
                    hovertemplate="%{x|%H:%M:%S} | %{y:.1f}°F<extra></extra>"
                    )
                )
        if 'hp-ewt' in request.selected_channels and 'hp-ewt' in self.data[request]['channels']:
            plotting_temperatures = True
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['hp-ewt']['times'], 
                    y=[self.to_fahrenheit(x/1000) for x in self.data[request]['channels']['hp-ewt']['values']], 
                    mode='lines+markers' if 'show-points'in request.selected_channels else 'lines', 
                    opacity=0.7,
                    line=dict(color='#1f77b4', dash='solid'),
                    name='HP EWT',
                    hovertemplate="%{x|%H:%M:%S} | %{y:.1f}°F<extra></extra>"
                    )
                )
        # Select yaxis for plotting power/flow
        y_axis_power = 'y2' if plotting_temperatures else 'y'
        # Power and flow
        plotting_power = False
        if 'hp-odu-pwr' in request.selected_channels and 'hp-odu-pwr' in self.data[request]['channels']:
            plotting_power = True
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['hp-odu-pwr']['times'], 
                    y=[x/1000 for x in self.data[request]['channels']['hp-odu-pwr']['values']], 
                    mode='lines+markers' if 'show-points'in request.selected_channels else 'lines', 
                    opacity=0.7,
                    line=dict(color='#2ca02c', dash='solid'),
                    name='HP outdoor power',
                    yaxis=y_axis_power,
                    hovertemplate="%{x|%H:%M:%S} | %{y:.1f} kW<extra></extra>"
                    )
                )
        if 'hp-idu-pwr' in request.selected_channels and 'hp-idu-pwr' in self.data[request]['channels']:
            plotting_power = True
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['hp-idu-pwr']['times'], 
                    y=[x/1000 for x in self.data[request]['channels']['hp-idu-pwr']['values']], 
                    mode='lines+markers' if 'show-points'in request.selected_channels else 'lines', 
                    opacity=0.7,
                    line=dict(color='#ff7f0e', dash='solid'),
                    name='HP indoor power',
                    yaxis=y_axis_power,
                    hovertemplate="%{x|%H:%M:%S} | %{y:.1f} kW<extra></extra>"
                    )
                ) 
        if 'oil-boiler-pwr' in request.selected_channels and 'oil-boiler-pwr' in self.data[request]['channels']:
            plotting_power = True
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['oil-boiler-pwr']['times'], 
                    y=[x/100 for x in self.data[request]['channels']['oil-boiler-pwr']['values']], 
                    mode='lines+markers' if 'show-points'in request.selected_channels else 'lines', 
                    opacity=0.7,
                    line=dict(color='#f0f0f0' if request.darkmode else '#5e5e5e', dash='solid'),
                    name='Oil boiler power x10',
                    yaxis=y_axis_power,
                    hovertemplate="%{x|%H:%M:%S} | %{y:.1f}/10 kW<extra></extra>"
                    )
                ) 
        if 'primary-flow' in request.selected_channels and 'primary-flow' in self.data[request]['channels']:
            plotting_power = True
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['primary-flow']['times'],
                    y=[x/100 for x in self.data[request]['channels']['primary-flow']['values']], 
                    mode='lines+markers' if 'show-points'in request.selected_channels else 'lines', 
                    opacity=0.4,
                    line=dict(color='purple', dash='solid'),
                    name='Primary pump flow',
                    yaxis=y_axis_power,
                    hovertemplate="%{x|%H:%M:%S} | %{y:.1f} GPM<extra></extra>"
                    )
                )
        if 'primary-pump-pwr' in request.selected_channels and 'primary-pump-pwr' in self.data[request]['channels']:
            plotting_power = True
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['primary-pump-pwr']['times'], 
                    y=[x/1000*100 for x in self.data[request]['channels']['primary-pump-pwr']['values']], 
                    mode='lines+markers' if 'show-points'in request.selected_channels else 'lines', 
                    opacity=0.7,
                    line=dict(color='pink', dash='solid'),
                    name='Primary pump power x100',
                    yaxis=y_axis_power,
                    visible='legendonly',
                    hovertemplate="%{x|%H:%M:%S} | %{y:.1f}/100 kW<extra></extra>"
                    )
                )
        # Layout
        if plotting_power and plotting_temperatures:
            fig.update_layout(yaxis=dict(title='Temperature [F]', range=[0,260]))
            fig.update_layout(yaxis2=dict(title='Power [kW] or Flow [GPM]', range=[0,35]))
        elif plotting_temperatures and not plotting_power:
            fig.update_layout(yaxis=dict(title='Temperature [F]'))
        elif plotting_power and not plotting_temperatures:
            fig.update_layout(yaxis=dict(title='Power [kW] or Flow [GPM]', range=[0,10]))
        fig.update_layout(
            title=dict(text='Heat pump', x=0.5, xanchor='center'),
            margin=dict(t=30, b=30),
            plot_bgcolor='#222222' if request.darkmode else 'white',
            paper_bgcolor='#222222' if request.darkmode else 'white',
            font_color='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
            title_font_color='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
            xaxis=dict(
                range=[self.data[request]['min_timestamp'], self.data[request]['max_timestamp']],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                showgrid=False
                ),
            yaxis=dict(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                zeroline=False,
                showgrid=True, 
                gridwidth=1, 
                gridcolor='#424242' if request.darkmode else 'LightGray'
                ),
            yaxis2=dict(
                mirror=True,
                ticks='outside',
                zeroline=False,
                showline=True,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                showgrid=False,
                overlaying='y', 
                side='right'
                ),
            legend=dict(
                x=0,
                y=1,
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(0, 0, 0, 0)'
                )
            )
        html_buffer = io.StringIO()
        fig.write_html(html_buffer, config={'displayModeBar': False})
        html_buffer.seek(0)
        print(f"Heat pump plot done in {round(time.time()-plot_start,1)} seconds")
        return html_buffer

    async def plot_distribution(self, request: DataRequest):
        plot_start = time.time()
        fig = go.Figure()
        # Temperature
        plotting_temperatures = False
        if 'dist-swt' in request.selected_channels and 'dist-swt' in self.data[request]['channels']:
            plotting_temperatures = True
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['dist-swt']['times'], 
                    y=[self.to_fahrenheit(x/1000) for x in self.data[request]['channels']['dist-swt']['values']], 
                    mode='lines+markers' if 'show-points'in request.selected_channels else 'lines', 
                    opacity=0.7,
                    line=dict(color='#d62728', dash='solid'),
                    name='Distribution SWT',
                    hovertemplate="%{x|%H:%M:%S} | %{y:.1f}°F<extra></extra>"
                    )
                )
        if 'dist-rwt' in request.selected_channels and 'dist-rwt' in self.data[request]['channels']:
            plotting_temperatures = True
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['dist-rwt']['times'], 
                    y=[self.to_fahrenheit(x/1000) for x in self.data[request]['channels']['dist-rwt']['values']], 
                    mode='lines+markers' if 'show-points'in request.selected_channels else 'lines', 
                    opacity=0.7,
                    line=dict(color='#1f77b4', dash='solid'),
                    name='Distribution RWT',
                    hovertemplate="%{x|%H:%M:%S} | %{y:.1f}°F<extra></extra>"
                    )
                )
        # Select yaxis for plotting power/flow
        y_axis_power = 'y2' if plotting_temperatures else 'y'
        # Power and flow
        plotting_power = False   
        if 'dist-flow' in request.selected_channels and 'dist-flow' in self.data[request]['channels']:
            plotting_power = True
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['dist-flow']['times'], 
                    y=[x/100 for x in self.data[request]['channels']['dist-flow']['values']], 
                    mode='lines+markers' if 'show-points'in request.selected_channels else 'lines', 
                    opacity=0.4,
                    line=dict(color='purple', dash='solid'),
                    name='Distribution flow',
                    yaxis = y_axis_power,
                    hovertemplate="%{x|%H:%M:%S} | %{y:.1f} GPM<extra></extra>"
                    )
                )
        if 'dist-flow' in request.selected_channels and 'dist-flow2' in self.data[request]['channels']:
            plotting_power = True
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['dist-flow2']['times'], 
                    y=[x/100 for x in self.data[request]['channels']['dist-flow2']['values']], 
                    mode='lines+markers' if 'show-points'in request.selected_channels else 'lines', 
                    opacity=0.4,
                    line=dict(color='orange', dash='solid'),
                    name='Distribution flow 2',
                    yaxis = y_axis_power,
                    hovertemplate="%{x|%H:%M:%S} | %{y:.1f} GPM<extra></extra>"
                    )
                )
        if 'dist-pump-pwr' in request.selected_channels and 'dist-pump-pwr' in self.data[request]['channels']:
            plotting_power = True
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['dist-pump-pwr']['times'], 
                    y=[x/10 for x in self.data[request]['channels']['dist-pump-pwr']['values']], 
                    mode='lines+markers' if 'show-points'in request.selected_channels else 'lines', 
                    opacity=0.7,
                    line=dict(color='pink', dash='solid'),
                    name='Distribution pump power /10',
                    yaxis = y_axis_power,
                    visible='legendonly',
                    hovertemplate="%{x|%H:%M:%S} | %{y:.1f}*10 W<extra></extra>"
                    )
                )

        if plotting_temperatures and plotting_power:
            fig.update_layout(yaxis=dict(title='Temperature [F]', range=[0,260]))
            fig.update_layout(yaxis2=dict(title='Flow [GPM] or Power [W]', range=[0,20]))
        elif plotting_temperatures and not plotting_power:
            fig.update_layout(yaxis=dict(title='Temperature [F]'))
        elif plotting_power and not plotting_temperatures:
            fig.update_layout(yaxis=dict('Flow [GPM] or Power [W]'))

        fig.update_layout(
            title=dict(text='Distribution', x=0.5, xanchor='center'),
            plot_bgcolor='#222222' if request.darkmode else 'white',
            paper_bgcolor='#222222' if request.darkmode else 'white',
            font_color='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
            title_font_color='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
            margin=dict(t=30, b=30),
            xaxis=dict(
                range=[self.data[request]['min_timestamp'], self.data[request]['max_timestamp']],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                showgrid=False
                ),
            yaxis=dict(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                zeroline=False,
                showgrid=True, 
                gridwidth=1, 
                gridcolor='#424242' if request.darkmode else 'LightGray'
                ),
            yaxis2=dict(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                zeroline=False,
                overlaying='y', 
                side='right', 
                showgrid=False,
                ),
            legend=dict(
                x=0,
                y=1,
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(0, 0, 0, 0)'
                )
            )
        html_buffer = io.StringIO()
        fig.write_html(html_buffer, config={'displayModeBar': False})
        html_buffer.seek(0) 
        print(f"Distribution plot done in {round(time.time()-plot_start,1)} seconds")
        return html_buffer
    
    async def plot_heatcalls(self, request: DataRequest):
        plot_start = time.time()
        fig = go.Figure()
        if 'zone-heat-calls' in request.selected_channels:
            for zone in self.data[request]['channels_by_zone']:
                if 'whitewire' not in self.data[request]['channels_by_zone'][zone]:
                    continue
                whitewire_ch = self.data[request]['channels_by_zone'][zone]['whitewire']
                zone_number = int(whitewire_ch[4])
                zone_color = self.zone_color[zone_number-1]
                # Interpret whitewire readings as active or not based on threshold
                if request.house_alias in self.whitewire_threshold_watts:
                    threshold = self.whitewire_threshold_watts[request.house_alias]
                else:
                    threshold = self.whitewire_threshold_watts['default']
                self.data[request]['channels'][whitewire_ch]['values'] = [
                    int(abs(x)>threshold) for x in self.data[request]['channels'][whitewire_ch]['values']
                    ]
                ww_times = self.data[request]['channels'][whitewire_ch]['times']
                ww_values = self.data[request]['channels'][whitewire_ch]['values']
                # Plot heat calls as periods
                last_was_1 = False
                heatcall_period_start = None
                for i in range(len(ww_values)):
                    if ww_values[i] == 1:
                        # Start a heat call period
                        if not last_was_1 or 'show-points' in request.selected_channels and i>0: 
                            fig.add_trace(
                                go.Scatter(
                                    x=[ww_times[i], ww_times[i]],
                                    y=[zone_number-1, zone_number],
                                    mode='lines',
                                    line=dict(color=zone_color, width=2),
                                    opacity=0.7,
                                    name=self.data[request]['channels_by_zone'][zone]['whitewire'].replace('-whitewire',''),
                                    showlegend=False,
                                    hovertemplate="%{x|%H:%M:%S}<extra></extra>"
                                )
                            )
                        if i >= len(ww_values)-1:
                            continue
                        if not heatcall_period_start:
                            heatcall_period_start = ww_times[i]
                        if ww_values[i+1] != 1:
                            # End a heat call period
                            fig.add_trace(
                                go.Scatter(
                                    x=[ww_times[i+1], ww_times[i+1]],
                                    y=[zone_number-1, zone_number],
                                    mode='lines',
                                    line=dict(color=zone_color, width=2),
                                    opacity=0.7,
                                    name=self.data[request]['channels_by_zone'][zone]['whitewire'].replace('-whitewire',''),
                                    showlegend=False,
                                    hovertemplate="%{x|%H:%M:%S}<extra></extra>"
                                )
                            )
                        if ww_values[i+1] != 1 or i+1==len(ww_values)-1:
                            # Add shading between heat call period start and end
                            if heatcall_period_start:
                                fig.add_shape(
                                    type='rect',
                                    x0=heatcall_period_start,
                                    y0=zone_number - 1,
                                    x1=ww_times[i+1],
                                    y1=zone_number,
                                    line=dict(color=zone_color, width=0),
                                    fillcolor=zone_color,
                                    opacity=0.2,
                                    name=self.data[request]['channels_by_zone'][zone]['whitewire'].replace('-whitewire', ''),
                                )
                                heatcall_period_start = None
                        last_was_1 = True
                    else:
                        last_was_1 = False
                fig.add_trace(
                    go.Scatter(
                        x=[None], 
                        y=[None],
                        mode='lines',
                        line=dict(color=zone_color, width=2),
                        name=self.data[request]['channels_by_zone'][zone]['whitewire'].replace('-whitewire','')
                    )
                )

        fig.update_layout(
            title=dict(text='Heat calls', x=0.5, xanchor='center'),
            plot_bgcolor='#222222' if request.darkmode else 'white',
            paper_bgcolor='#222222' if request.darkmode else 'white',
            font_color='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
            title_font_color='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
            margin=dict(t=30, b=30),
            xaxis=dict(
                range=[self.data[request]['min_timestamp'], self.data[request]['max_timestamp']],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                showgrid=False
                ),
            yaxis=dict(
                range = [-0.5, len(self.data[request]['channels_by_zone'].keys())*1.3],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                zeroline=False,
                showgrid=True, 
                gridwidth=1, 
                gridcolor='#424242' if request.darkmode else 'LightGray', 
                tickvals=list(range(len(self.data[request]['channels_by_zone'].keys())+1)),
                ),
            yaxis2=dict(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                ),
            legend=dict(
                x=0,
                y=1,
                xanchor='left',
                yanchor='top',
                orientation='h',
                bgcolor='rgba(0, 0, 0, 0)'
            )
        )
        html_buffer = io.StringIO()
        fig.write_html(html_buffer, config={'displayModeBar': False})
        html_buffer.seek(0)
        print(f"Heat calls plot done in {round(time.time()-plot_start,1)} seconds")
        return html_buffer
    
    async def plot_zones(self, request: DataRequest):
        plot_start = time.time()
        fig = go.Figure()

        # Zone temperature and setpoint
        min_zones, max_zones = 45, 80
        for zone in self.data[request]['channels_by_zone']:
            if 'temp' in self.data[request]['channels_by_zone'][zone]:
                temp_channel = self.data[request]['channels_by_zone'][zone]['temp']
                fig.add_trace(
                    go.Scatter(
                        x=self.data[request]['channels'][temp_channel]['times'], 
                        y=[x/1000 for x in self.data[request]['channels'][temp_channel]['values']], 
                        mode='lines+markers' if 'show-points'in request.selected_channels else 'lines', 
                        opacity=0.7,
                        line=dict(color=self.zone_color[int(zone[4])-1], dash='solid'),
                        name=self.data[request]['channels_by_zone'][zone]['temp'].replace('-temp',''),
                        hovertemplate="%{x|%H:%M:%S} | %{y:.1f}°F<extra></extra>"
                        )
                    )
                min_zones = min(min_zones, min(self.data[request]['channels'][temp_channel]['values'])/1000)
                max_zones = max(max_zones, max(self.data[request]['channels'][temp_channel]['values'])/1000)
            if 'set' in self.data[request]['channels_by_zone'][zone]:
                set_channel = self.data[request]['channels_by_zone'][zone]['set']
                fig.add_trace(
                    go.Scatter(
                        x=self.data[request]['channels'][set_channel]['times'], 
                        y=[x/1000 for x in self.data[request]['channels'][set_channel]['values']], 
                        mode='lines+markers' if 'show-points'in request.selected_channels else 'lines', 
                        opacity=0.7,
                        line=dict(color=self.zone_color[int(zone[4])-1], dash='dash'),
                        name=self.data[request]['channels_by_zone'][zone]['set'].replace('-set',''),
                        showlegend=False,
                        hovertemplate="%{x|%H:%M:%S} | %{y:.1f}°F<extra></extra>"
                        )
                    )
                min_zones = min(min_zones, min(self.data[request]['channels'][set_channel]['values'])/1000)
                max_zones = max(max_zones, max(self.data[request]['channels'][set_channel]['values'])/1000)

        # Outside air temperature
        min_oat, max_oat = 70, 80    
        if 'oat' in request.selected_channels and 'oat' in self.data[request]['channels']:
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['oat']['times'], 
                    y=[self.to_fahrenheit(x/1000) for x in self.data[request]['channels']['oat']['values']], 
                    mode='lines+markers' if 'show-points'in request.selected_channels else 'lines', 
                    opacity=0.8,
                    line=dict(color='gray' if request.darkmode else '#d6d6d6', dash='solid'),
                    name='Outside air',
                    yaxis='y2',
                    hovertemplate="%{x|%H:%M:%S} | %{y:.1f}°F<extra></extra>"
                    )
                )
            min_oat = self.to_fahrenheit(min(self.data[request]['channels']['oat']['values'])/1000)
            max_oat = self.to_fahrenheit(max(self.data[request]['channels']['oat']['values'])/1000)
            fig.update_layout(yaxis2=dict(title='Outside air temperature [F]'))

        fig.update_layout(yaxis=dict(title='Zone temperature [F]'))
        fig.update_layout(
            title=dict(text='Zones', x=0.5, xanchor='center'),
            plot_bgcolor='#222222' if request.darkmode else 'white',
            paper_bgcolor='#222222' if request.darkmode else 'white',
            font_color='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
            title_font_color='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
            margin=dict(t=30, b=30),
            xaxis=dict(
                range=[self.data[request]['min_timestamp'], self.data[request]['max_timestamp']],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                showgrid=False
                ),
            yaxis=dict(
                range = [min_zones-30,max_zones+20],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                zeroline=False,
                showgrid=True, 
                gridwidth=1, 
                gridcolor='#424242' if request.darkmode else 'LightGray'
                ),
            yaxis2=dict(
                range = [min_oat-2, max_oat+20],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                overlaying='y', 
                side='right', 
                zeroline=False,
                showgrid=False, 
                ),
            legend=dict(
                x=0,
                y=1,
                xanchor='left',
                yanchor='top',
                orientation='h',
                bgcolor='rgba(0, 0, 0, 0)'
                )
            )
        html_buffer = io.StringIO()
        fig.write_html(html_buffer, config={'displayModeBar': False})
        html_buffer.seek(0)
        print(f"Zones plot done in {round(time.time()-plot_start,1)} seconds")
        return html_buffer
    
    async def plot_buffer(self, request: DataRequest):
        plot_start = time.time()
        fig = go.Figure()

        gradient = plt.get_cmap('coolwarm', 4)
        buffer_colors = {
            'buffer-depth1': gradient(3),
            'buffer-depth2': gradient(2),
            'buffer-depth3': gradient(1),
            'buffer-depth4': gradient(0)
            }
        buffer_layer_colors = {key: self.to_hex(value) for key, value in buffer_colors.items()}

        min_buffer_temp, max_buffer_temp = 1e5, 0
        if 'buffer-depths' in request.selected_channels:
            buffer_channels = sorted([key for key in self.data[request]['channels'].keys() if 'buffer-depth' in key and 'micro-v' not in key])
            for buffer_channel in buffer_channels:
                min_buffer_temp = min(min_buffer_temp, min([self.to_fahrenheit(x/1000) for x in self.data[request]['channels'][buffer_channel]['values']]))
                max_buffer_temp = max(max_buffer_temp, max([self.to_fahrenheit(x/1000) for x in self.data[request]['channels'][buffer_channel]['values']]))
                fig.add_trace(
                    go.Scatter(
                        x=self.data[request]['channels'][buffer_channel]['times'], 
                        y=[self.to_fahrenheit(x/1000) for x in self.data[request]['channels'][buffer_channel]['values']], 
                        mode='lines+markers' if 'show-points'in request.selected_channels else 'lines', 
                        opacity=0.7,
                        name=buffer_channel.replace('buffer-',''),
                        line=dict(color=buffer_layer_colors[buffer_channel], dash='solid'),
                        hovertemplate="%{x|%H:%M:%S} | %{y:.1f}°F<extra></extra>"
                        )
                    )  
        if 'buffer-hot-pipe' in request.selected_channels and 'buffer-hot-pipe' in self.data[request]['channels']:
            min_buffer_temp = min(min_buffer_temp, min([self.to_fahrenheit(x/1000) for x in self.data[request]['channels']['buffer-hot-pipe']['values']]))
            max_buffer_temp = max(max_buffer_temp, max([self.to_fahrenheit(x/1000) for x in self.data[request]['channels']['buffer-hot-pipe']['values']]))
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['buffer-hot-pipe']['times'], 
                    y=[self.to_fahrenheit(x/1000) for x in self.data[request]['channels']['buffer-hot-pipe']['values']], 
                    mode='lines+markers' if 'show-points'in request.selected_channels else 'lines', 
                    opacity=0.7,
                    name='Hot pipe',
                    line=dict(color='#d62728', dash='solid'),
                    hovertemplate="%{x|%H:%M:%S} | %{y:.1f}°F<extra></extra>"
                    )
                )
        if 'buffer-cold-pipe' in request.selected_channels and 'buffer-cold-pipe' in self.data[request]['channels']:
            min_buffer_temp = min(min_buffer_temp, min([self.to_fahrenheit(x/1000) for x in self.data[request]['channels']['buffer-cold-pipe']['values']]))
            max_buffer_temp = max(max_buffer_temp, max([self.to_fahrenheit(x/1000) for x in self.data[request]['channels']['buffer-cold-pipe']['values']]))
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['buffer-cold-pipe']['times'], 
                    y=[self.to_fahrenheit(x/1000) for x in self.data[request]['channels']['buffer-cold-pipe']['values']], 
                    mode='lines+markers' if 'show-points'in request.selected_channels else 'lines', 
                    opacity=0.7,
                    name='Cold pipe',
                    line=dict(color='#1f77b4', dash='solid'),
                    hovertemplate="%{x|%H:%M:%S} | %{y:.1f}°F<extra></extra>"
                    )
                )
               
        fig.update_layout(
            title=dict(text='Buffer', x=0.5, xanchor='center'),
            plot_bgcolor='#222222' if request.darkmode else 'white',
            paper_bgcolor='#222222' if request.darkmode else 'white',
            font_color='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
            title_font_color='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
            margin=dict(t=30, b=30),
            xaxis=dict(
                range=[self.data[request]['min_timestamp'], self.data[request]['max_timestamp']],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                showgrid=False,
                ),
            yaxis=dict(
                range = [min_buffer_temp-15, max_buffer_temp+30],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                title='Temperature [F]', 
                zeroline=False,
                showgrid=True, 
                gridwidth=1, 
                gridcolor='#424242' if request.darkmode else 'LightGray',
                ),
            legend=dict(
                x=0,
                y=1,
                xanchor='left',
                yanchor='top',
                orientation='h',
                bgcolor='rgba(0, 0, 0, 0)'
                )
            )

        html_buffer = io.StringIO()
        fig.write_html(html_buffer, config={'displayModeBar': False})
        html_buffer.seek(0)
        print(f"Buffer plot done in {round(time.time()-plot_start,1)} seconds")
        return html_buffer

    async def plot_storage(self, request: DataRequest):
        plot_start = time.time()
        fig = go.Figure()

        gradient = plt.get_cmap('coolwarm', 12)
        storage_colors = {
            'tank1-depth1': gradient(11),
            'tank1-depth2': gradient(10),
            'tank1-depth3': gradient(9),
            'tank1-depth4': gradient(8),
            'tank2-depth1': gradient(7),
            'tank2-depth2': gradient(6),
            'tank2-depth3': gradient(5),
            'tank2-depth4': gradient(4),
            'tank3-depth1': gradient(3),
            'tank3-depth2': gradient(2),
            'tank3-depth3': gradient(1),
            'tank3-depth4': gradient(0),
            }
        storage_layer_colors = {key: self.to_hex(value) for key, value in storage_colors.items()}
        
        # Temperature
        plotting_temperatures = False
        min_store_temp, max_store_temp = 1e5, 0
        if 'storage-depths' in request.selected_channels:
            plotting_temperatures = True
            tank_channels = sorted([key for key in self.data[request]['channels'].keys() if 'tank' in key and 'micro-v' not in key])
            for tank_channel in tank_channels:
                min_store_temp = min(min_store_temp, min([self.to_fahrenheit(x/1000) for x in self.data[request]['channels'][tank_channel]['values']]))
                max_store_temp = max(max_store_temp, max([self.to_fahrenheit(x/1000) for x in self.data[request]['channels'][tank_channel]['values']]))
                fig.add_trace(
                    go.Scatter(
                        x=self.data[request]['channels'][tank_channel]['times'], 
                        y=[self.to_fahrenheit(x/1000) for x in self.data[request]['channels'][tank_channel]['values']], 
                        mode='lines+markers' if 'show-points'in request.selected_channels else 'lines', opacity=0.7,
                        name=tank_channel.replace('storage-',''),
                        line=dict(color=storage_layer_colors[tank_channel], dash='solid'),
                        hovertemplate="%{x|%H:%M:%S} | %{y:.1f}°F"
                        )
                    )
        if 'store-hot-pipe' in request.selected_channels and 'store-hot-pipe' in self.data[request]['channels']:
            plotting_temperatures = True
            min_store_temp = min(min_store_temp, min([self.to_fahrenheit(x/1000) for x in self.data[request]['channels']['store-hot-pipe']['values']]))
            max_store_temp = max(max_store_temp, max([self.to_fahrenheit(x/1000) for x in self.data[request]['channels']['store-hot-pipe']['values']]))
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['store-hot-pipe']['times'], 
                    y=[self.to_fahrenheit(x/1000) for x in self.data[request]['channels']['store-hot-pipe']['values']], 
                    mode='lines+markers' if 'show-points'in request.selected_channels else 'lines', 
                    opacity=0.7,
                    name='Hot pipe',
                    line=dict(color='#d62728', dash='solid'),
                    hovertemplate="%{x|%H:%M:%S} | %{y:.1f}°F<extra></extra>"
                    )
                )
        if 'store-cold-pipe' in request.selected_channels and 'store-cold-pipe' in self.data[request]['channels']:
            plotting_temperatures = True
            min_store_temp = min(min_store_temp, min([self.to_fahrenheit(x/1000) for x in self.data[request]['channels']['store-cold-pipe']['values']]))
            max_store_temp = max(max_store_temp, max([self.to_fahrenheit(x/1000) for x in self.data[request]['channels']['store-cold-pipe']['values']]))
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['store-cold-pipe']['times'], 
                    y=[self.to_fahrenheit(x/1000) for x in self.data[request]['channels']['store-cold-pipe']['values']], 
                    mode='lines+markers' if 'show-points'in request.selected_channels else 'lines', 
                    opacity=0.7,
                    name='Cold pipe',
                    line=dict(color='#1f77b4', dash='solid'),
                    hovertemplate="%{x|%H:%M:%S} | %{y:.1f}°F<extra></extra>"
                    )
                )
        # Select yaxis for plotting power/flow
        y_axis_power = 'y2' if plotting_temperatures else 'y'
        # Power and flow
        plotting_power = False
        if 'store-pump-pwr' in request.selected_channels and 'store-pump-pwr' in self.data[request]['channels']:
            plotting_power = True
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['store-pump-pwr']['times'], 
                    y=[x for x in self.data[request]['channels']['store-pump-pwr']['values']], 
                    mode='lines+markers' if 'show-points'in request.selected_channels else 'lines', 
                    opacity=0.7,
                    line=dict(color='pink', dash='solid'),
                    name='Storage pump power x1000',
                    yaxis=y_axis_power,
                    visible='legendonly',
                    hovertemplate="%{x|%H:%M:%S} | %{y:.1f}/1000 kW<extra></extra>"
                    )
                )
        if 'store-flow' in request.selected_channels and 'store-flow' in self.data[request]['channels']:
            plotting_power = True
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['store-flow']['times'], 
                    y=[x/100*10 for x in self.data[request]['channels']['store-flow']['values']], 
                    mode='lines+markers' if 'show-points'in request.selected_channels else 'lines', 
                    opacity=0.4,
                    line=dict(color='purple', dash='solid'),
                    name='Storage pump flow x10',
                    yaxis=y_axis_power,
                    hovertemplate="%{x|%H:%M:%S} | %{y:.1f}/10 GPM<extra></extra>"
                    )
                )
        max_power = 60
        if 'store-energy' in request.selected_channels and 'usable-energy' in self.data[request]['channels']:
            plotting_power = True
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['usable-energy']['times'], 
                    y=[x/1000 for x in self.data[request]['channels']['usable-energy']['values']], 
                    mode='lines+markers' if 'show-points'in request.selected_channels else 'lines', 
                    opacity=0.4,
                    line=dict(color='#2ca02c', dash='solid'),
                    name='Usable',
                    yaxis=y_axis_power,
                    visible='legendonly',
                    hovertemplate="%{x|%H:%M:%S} | %{y:.1f} kWh<extra></extra>"
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['required-energy']['times'], 
                    y=[x/1000 for x in self.data[request]['channels']['required-energy']['values']], 
                    mode='lines+markers' if 'show-points'in request.selected_channels else 'lines', 
                    opacity=0.4,
                    line=dict(color='#2ca02c', dash='dash'),
                    name='Required',
                    yaxis=y_axis_power,
                    visible='legendonly',
                    hovertemplate="%{x|%H:%M:%S} | %{y:.1f} kWh<extra></extra>"
                    )
                )
            max_power = max([x/1000 for x in self.data[request]['channels']['required-energy']['values']])*4
            
        if plotting_temperatures and plotting_power:
            fig.update_layout(yaxis=dict(title='Temperature [F]', range=[min_store_temp-80, max_store_temp+80]))
            fig.update_layout(yaxis2=dict(title='GPM, kW, or kWh', range=[-1, max_power]))
        elif plotting_temperatures and not plotting_power:
            min_store_temp = 20 if min_store_temp<0 else min_store_temp
            fig.update_layout(yaxis=dict(title='Temperature [F]', range=[min_store_temp-20, max_store_temp+60]))
        elif plotting_power and not plotting_temperatures:
            fig.update_layout(yaxis=dict(title='GPM, kW, or kWh'))

        fig.update_layout(
            title=dict(text='Storage', x=0.5, xanchor='center'),
            plot_bgcolor='#222222' if request.darkmode else 'white',
            paper_bgcolor='#222222' if request.darkmode else 'white',
            font_color='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
            title_font_color='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
            margin=dict(t=30, b=30),
            xaxis=dict(
                range=[self.data[request]['min_timestamp'], self.data[request]['max_timestamp']],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                showgrid=False,
                ),
            yaxis=dict(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                zeroline=False,
                showgrid=True, 
                gridwidth=1, 
                gridcolor='#424242' if request.darkmode else 'LightGray'
                ),
            yaxis2=dict(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                overlaying='y', 
                side='right', 
                zeroline=False,
                showgrid=False, 
                ),
            legend=dict(
                x=0,
                y=1,
                xanchor='left',
                yanchor='top',
                orientation='h',
                bgcolor='rgba(0, 0, 0, 0)'
            )
        )

        html_buffer = io.StringIO()
        fig.write_html(html_buffer, config={'displayModeBar': False})
        html_buffer.seek(0)
        print(f"Storage plot done in {round(time.time()-plot_start,1)} seconds")
        return html_buffer
    
    async def plot_top_state(self, request: DataRequest):
        plot_start = time.time()
        fig = go.Figure()

        top_state_color = {
            'HomeAlone': '#EF553B',
            'Atn': '#00CC96',
            'Admin': '#636EFA'
        }
        
        if self.data[request]['top_states']:
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['top_states']['all']['times'],
                    y=self.data[request]['top_states']['all']['values'],
                    mode='lines',
                    line=dict(color='#f0f0f0' if request.darkmode else '#5e5e5e', width=2),
                    opacity=0.3,
                    showlegend=False,
                    line_shape='hv'
                )
            )
            for state in self.data[request]['top_states'].keys():
                if state != 'all' and state in top_state_color:
                    fig.add_trace(
                        go.Scatter(
                            x=self.data[request]['top_states'][state]['times'],
                            y=self.data[request]['top_states'][state]['values'],
                            mode='markers',
                            marker=dict(color=top_state_color[state], size=10),
                            opacity=0.8,
                            name=state,
                            hovertemplate="%{x|%H:%M:%S}"
                        )
                    )

        fig.update_layout(
            title=dict(text='Top State', x=0.5, xanchor='center'),
            plot_bgcolor='#222222' if request.darkmode else 'white',
            paper_bgcolor='#222222' if request.darkmode else 'white',
            font_color='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
            title_font_color='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
            margin=dict(t=30, b=30),
            xaxis=dict(
                range=[self.data[request]['min_timestamp'], self.data[request]['max_timestamp']],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                showgrid=False
                ),
            yaxis=dict(
                range = [-0.6, len(self.data[request]['top_states'])-1+0.2],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                zeroline=False,
                showgrid=True, 
                gridwidth=1, 
                gridcolor='#424242' if request.darkmode else 'LightGray', 
                tickvals=list(range(len(self.data[request]['top_states'])-1)),
                ),
            legend=dict(
                x=0,
                y=1,
                xanchor='left',
                yanchor='top',
                orientation='h',
                bgcolor='rgba(0, 0, 0, 0)'
            )
        )
        
        html_buffer = io.StringIO()
        fig.write_html(html_buffer, config={'displayModeBar': False})
        html_buffer.seek(0)
        print(f"Top state plot done in {round(time.time()-plot_start,1)} seconds")
        return html_buffer
    
    async def plot_ha_state(self, request: DataRequest):
        plot_start = time.time()
        fig = go.Figure()

        ha_state_color = {
            'HpOffStoreDischarge': '#EF553B',
            'HpOffStoreOff': '#00CC96',
            'HpOnStoreOff': '#636EFA',
            'HpOnStoreCharge': '#feca52',
            'Initializing': '#a3a3a3',
            'StratBoss': '#ee93fa',
            'Dormant': '#4f4f4f'
        }

        if self.data[request]['ha_states']!={}:
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['ha_states']['all']['times'],
                    y=self.data[request]['ha_states']['all']['values'],
                    mode='lines',
                    line=dict(color='#f0f0f0' if request.darkmode else '#5e5e5e', width=2),
                    opacity=0.3,
                    showlegend=False,
                    line_shape='hv'
                )
            )
            for state in self.data[request]['ha_states'].keys():
                if state != 'all' and state in ha_state_color:
                    fig.add_trace(
                        go.Scatter(
                            x=self.data[request]['ha_states'][state]['times'],
                            y=self.data[request]['ha_states'][state]['values'],
                            mode='markers',
                            marker=dict(color=ha_state_color[state], size=10),
                            opacity=0.8,
                            name=state,
                            hovertemplate="%{x|%H:%M:%S}"
                        )
                    )

        fig.update_layout(
            title=dict(text='HomeAlone State', x=0.5, xanchor='center'),
            plot_bgcolor='#222222' if request.darkmode else 'white',
            paper_bgcolor='#222222' if request.darkmode else 'white',
            font_color='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
            title_font_color='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
            margin=dict(t=30, b=30),
            xaxis=dict(
                range=[self.data[request]['min_timestamp'], self.data[request]['max_timestamp']],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                showgrid=False
                ),
            yaxis=dict(
                range = [-0.6, 8-0.8],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                zeroline=False,
                showgrid=True, 
                gridwidth=1, 
                gridcolor='#424242' if request.darkmode else 'LightGray', 
                tickvals=list(range(6)),
                ),
            legend=dict(
                x=0,
                y=1,
                xanchor='left',
                yanchor='top',
                orientation='h',
                bgcolor='rgba(0, 0, 0, 0)'
            )
        )

        html_buffer = io.StringIO()
        fig.write_html(html_buffer, config={'displayModeBar': False})
        html_buffer.seek(0)
        print(f"HA state plot done in {round(time.time()-plot_start,1)} seconds")
        return html_buffer
    
    async def plot_aa_state(self, request: DataRequest):
        plot_start = time.time()
        fig = go.Figure()

        aa_state_color = {
            'HpOffStoreDischarge': '#EF553B',
            'HpOffStoreOff': '#00CC96',
            'HpOnStoreOff': '#636EFA',
            'HpOnStoreCharge': '#feca52',
            'Initializing': '#a3a3a3',
            'StratBoss': '#ee93fa',
            'Dormant': '#4f4f4f'
        }

        if self.data[request]['aa_states']!={}:
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['aa_states']['all']['times'],
                    y=self.data[request]['aa_states']['all']['values'],
                    mode='lines',
                    line=dict(color='#f0f0f0' if request.darkmode else '#5e5e5e', width=2),
                    opacity=0.3,
                    showlegend=False,
                    line_shape='hv'
                )
            )
            for state in self.data[request]['aa_states'].keys():
                if state != 'all' and state in aa_state_color:
                    fig.add_trace(
                        go.Scatter(
                            x=self.data[request]['aa_states'][state]['times'],
                            y=self.data[request]['aa_states'][state]['values'],
                            mode='markers',
                            marker=dict(color=aa_state_color[state], size=10),
                            opacity=0.8,
                            name=state,
                            hovertemplate="%{x|%H:%M:%S}"
                        )
                    )

        fig.update_layout(
            title=dict(text='AtomicAlly State', x=0.5, xanchor='center'),
            plot_bgcolor='#222222' if request.darkmode else 'white',
            paper_bgcolor='#222222' if request.darkmode else 'white',
            font_color='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
            title_font_color='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
            margin=dict(t=30, b=30),
            xaxis=dict(
                range=[self.data[request]['min_timestamp'], self.data[request]['max_timestamp']],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                showgrid=False
                ),
            yaxis=dict(
                range = [-0.6, 8-0.8],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                zeroline=False,
                showgrid=True, 
                gridwidth=1, 
                gridcolor='#424242' if request.darkmode else 'LightGray', 
                tickvals=list(range(7)),
                ),
            legend=dict(
                x=0,
                y=1,
                xanchor='left',
                yanchor='top',
                orientation='h',
                bgcolor='rgba(0, 0, 0, 0)'
            )
        )

        html_buffer = io.StringIO()
        fig.write_html(html_buffer, config={'displayModeBar': False})
        html_buffer.seek(0)
        print(f"AA state plot done in {round(time.time()-plot_start,1)} seconds")
        return html_buffer 

    async def plot_weather(self, request: DataRequest):
        plot_start = time.time()
        fig = go.Figure()
                
        oat_forecasts, ws_forecasts = {}, {}
        for message in self.data[request]['weather_forecasts']:
            forecast_start_time = int((message.message_persisted_ms/1000//3600)*3600)
            oat_forecasts[forecast_start_time] = message.payload['OatF']
            ws_forecasts[forecast_start_time] = message.payload['WindSpeedMph']

        color_scale = pc.diverging.RdBu[::-1]
        for i, weather_time in enumerate(oat_forecasts):
            forecast_times = [int(weather_time) + 3600*i for i in range(len(oat_forecasts[weather_time]))]
            forecast_times = [self.to_datetime(x*1000) for x in forecast_times]
            color = 'red' if i == len(oat_forecasts)-1 else color_scale[int((i/len(oat_forecasts))*(len(color_scale)-1))]
            fig.add_trace(
                go.Scatter(
                    x=forecast_times,
                    y=oat_forecasts[weather_time],
                    mode='lines',
                    line=dict(color=color, width=2),
                    opacity=0.2 if i<len(oat_forecasts)-1 else 1,
                    showlegend=False,
                    line_shape='hv',
                    hovertemplate="%{x|%H:%M:%S} | %{y}°F<extra></extra>"
                )
            )

        fig.update_layout(
            title=dict(text='Weather Forecasts', x=0.5, xanchor='center'),
            plot_bgcolor='#222222' if request.darkmode else 'white',
            paper_bgcolor='#222222' if request.darkmode else 'white',
            font_color='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
            title_font_color='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
            margin=dict(t=30, b=30),
            xaxis=dict(
                range=[self.data[request]['min_timestamp'], self.data[request]['max_timestamp']],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                showgrid=False
                ),
            yaxis=dict(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                zeroline=False,
                showgrid=True, 
                gridwidth=1, 
                gridcolor='#424242' if request.darkmode else 'LightGray', 
                ),
            legend=dict(
                x=0,
                y=1,
                xanchor='left',
                yanchor='top',
                orientation='h',
                bgcolor='rgba(0, 0, 0, 0)'
            )
        )

        html_buffer = io.StringIO()
        fig.write_html(html_buffer, config={'displayModeBar': False})
        html_buffer.seek(0)
        print(f"Weather plot done in {round(time.time()-plot_start,1)} seconds")
        return html_buffer
    
    async def plot_prices(self, request: Union[DataRequest, BaseRequest], aggregate=False):
        if not isinstance(request, DataRequest) and not aggregate:
            raise Exception()
        
        plot_start = time.time()
        fig = go.Figure()

        # Open and read the price CSV file
        csv_times, csv_dist, csv_lmp = [], [], []
        project_dir = os.path.dirname(os.path.abspath(__file__))
        elec_file = os.path.join(project_dir, 'price_forecast_dates.csv')
        with open(elec_file, newline='', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)
            for row in csvreader:
                csv_times.append(float(row[0]))
                csv_dist.append(float(row[1])/10)
                csv_lmp.append(float(row[2])/10)

        request_hours = int((request.end_ms - request.start_ms)/1000 / 3600)
        price_times_s = [request.start_ms/1000 + x*3600 for x in range(request_hours+2+48)]
        lmp_values = [lmp for time, dist, lmp in zip(csv_times, csv_dist, csv_lmp) if time in price_times_s]
        total_price_values = [lmp+dist for time, dist, lmp in zip(csv_times, csv_dist, csv_lmp) if time in price_times_s]
        csv_times = [time for time in csv_times if time in price_times_s]
        price_times = [self.to_datetime(x*1000) for x in csv_times]

        fig.add_trace(
            go.Scatter(
                x=price_times,
                y=total_price_values,
                mode='lines',
                line=dict(color='#269638' if aggregate else ('#f0f0f0' if request.darkmode else '#5e5e5e')), #42f560
                opacity=0.8,
                showlegend=True,
                line_shape='hv',
                name='Total',
                hovertemplate="%{x|%H:%M:%S} | %{y:.2f} cts/kWh"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=price_times,
                y=lmp_values,
                mode='lines',
                line=dict(color='#269638' if aggregate else ('#f0f0f0' if request.darkmode else '#5e5e5e'), dash='dot'),
                opacity=0.4,
                showlegend=True,
                line_shape='hv',
                yaxis='y2',
                name='LMP',
                hovertemplate="%{x|%H:%M:%S} | %{y:.2f} cts/kWh"
            )
        )

        if aggregate:
            min_timestep, max_timestep = self.to_datetime(request.start_ms), self.to_datetime(request.end_ms+(
                    5*3600*1000 if time.time()-request.end_ms/1000<5*3600 else 0))
            plot_bgcolor='#313131' if request.darkmode else '#F5F5F7'
            paper_bgcolor='#313131' if request.darkmode else '#F5F5F7'
            font_color='#b5b5b5' if request.darkmode else 'rgb(42,63,96)'
            title_font_color='#b5b5b5' if request.darkmode else 'rgb(42,63,96)'
        else: 
            min_timestep = self.data[request]['min_timestamp']
            max_timestep = self.data[request]['max_timestamp']
            plot_bgcolor='#222222' if request.darkmode else 'white'
            paper_bgcolor='#222222' if request.darkmode else 'white'
            font_color='#b5b5b5' if request.darkmode else 'rgb(42,63,96)'
            title_font_color='#b5b5b5' if request.darkmode else 'rgb(42,63,96)'
            
        fig.update_layout(yaxis=dict(title='Total price [cts/kWh]'))
        fig.update_layout(yaxis2=dict(title='LMP [cts/kWh]'))
        fig.update_layout(
            # shapes = shapes_list,
            title=dict(text='Price Forecast' if not aggregate else '', x=0.5, xanchor='center'),
            plot_bgcolor=plot_bgcolor,
            paper_bgcolor=paper_bgcolor,
            font_color=font_color,
            title_font_color=title_font_color,
            margin=dict(t=30, b=30),
            xaxis=dict(
                range=[min_timestep, max_timestep],
                mirror=True,
                ticks='outside',
                showline=False if aggregate else True,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                showgrid=False
                ),
            yaxis=dict(
                mirror=True,
                ticks='outside',
                showline=False if aggregate else True,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                zeroline=False,
                showgrid=False if aggregate else True, 
                gridwidth=1, 
                gridcolor='#424242' if request.darkmode else 'LightGray', 
                ),
            yaxis2=dict(
                mirror=True,
                ticks='outside',
                zeroline=False,
                showline=False if aggregate else True,
                linecolor='#b5b5b5' if request.darkmode else 'rgb(42,63,96)',
                showgrid=False,
                overlaying='y', 
                side='right'
                ),
            legend=dict(
                x=0,
                y=1,
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(0, 0, 0, 0)'
            )
        )

        html_buffer = io.StringIO()
        fig.write_html(html_buffer, config={'displayModeBar': False})
        html_buffer.seek(0)
        print(f"Prices plot done in {round(time.time()-plot_start,1)} seconds")   
        return html_buffer             


if __name__ == "__main__":
    a = VisualizerApi()
    a.start()
