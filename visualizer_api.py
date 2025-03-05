import io
import os
import csv
import time
import uuid
import dotenv
import uvicorn
import zipfile
import pendulum
import traceback
import numpy as np
import pandas as pd
from datetime import timedelta
from pydantic import BaseModel, Field
from typing import List, Optional, Union
import asyncio
import async_timeout
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
from sqlalchemy import create_engine, asc, or_, and_, desc
from sqlalchemy.orm import sessionmaker
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.colors as pc
from config import Settings
from models import MessageSql
from flo import DGraph
from hinge import FloHinge
from named_types import FloParamsHouse0
    
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
    def __init__(self, running_locally):
        self.running_locally = running_locally
        self.settings = Settings(_env_file=dotenv.find_dotenv())
        engine = create_engine(self.settings.db_url.get_secret_value())
        self.Session = sessionmaker(bind=engine)
        self.admin_user_password = self.settings.visualizer_api_password.get_secret_value()
        self.timezone_str = 'America/New_York'
        self.timeout_seconds = 5*60
        self.top_states_order = ['HomeAlone', 'Atn', 'Dormant']
        self.ha_states_order = [
            'HpOffStoreDischarge', 'HpOffStoreOff', 'HpOnStoreOff', 
            'HpOnStoreCharge', 'StratBoss', 'Initializing', 'Dormant'
            ]
        self.aa_states_order = self.ha_states_order.copy()
        self.whitewire_threshold_watts = {'beech': 100, 'default': 20}
        self.zone_color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']*3
        self.data = {}

    def start(self):
        self.app = FastAPI()
        self.app.add_middleware(
            CORSMiddleware,
            # TODO: allow_origins=["https://thegridelectric.github.io"]
            allow_origins=["*"], 
            allow_credentials=True,
            allow_methods=["*"],
        )
        self.app.post("/plots")(self.get_plots)
        self.app.post("/csv")(self.get_csv)
        self.app.post("/messages")(self.get_messages)
        self.app.post("/flo")(self.get_flo)
        uvicorn.run(self.app, host="0.0.0.0", port=8000)

    def to_datetime(self, time_ms):
        return pendulum.from_timestamp(time_ms/1000, tz=self.timezone_str)

    def to_fahrenheit(self, t):
        return t*9/5+32
    
    def to_hex(self, rgba):
        r, g, b, a = (int(c * 255) for c in rgba)
        return f'#{r:02x}{g:02x}{b:02x}'

    def check_password(self, house_alias, password):
        if password == self.admin_user_password:
            return True
        house_owner_password = getattr(self.settings, f"{house_alias}_owner_password").get_secret_value()
        if password == house_owner_password:
            return True
        return False
    
    def check_request(self, request: Union[DataRequest, CsvRequest, DijkstraRequest, MessagesRequest]):
        if not self.check_password(request.house_alias, request.password):
            return {"success": False, "message": "Wrong password.", "reload": True}
        if not isinstance(request, MessagesRequest) and request.house_alias == '':
            return {"success": False, "message": "Please enter a house alias.", "reload": True}
        if isinstance(request, Union[DataRequest, CsvRequest]) and not request.confirm_with_user:
            if (request.end_ms - request.start_ms)/1000/3600/24 > 3:
                warning_message = f"That's a lot of data! Are you sure you want to proceed?"
                return {"success": False, "message": warning_message, "reload": False, "confirm_with_user": True}
        if not self.running_locally: 
            if (isinstance(request, DataRequest) and not isinstance(request, CsvRequest) 
                and (request.end_ms-request.start_ms)/1000/3600/24 > 5):
                warning_message = "Plotting data for this many days is not permitted. Please reduce the range and try again."
                return {"success": False, "message": warning_message, "reload": False}
            if isinstance(request, CsvRequest) and (request.end_ms-request.start_ms)/1000/3600/24 > 21:
                warning_message = "Downloading data for this many days is not permitted. Please reduce the range and try again."
                return {"success": False, "message": warning_message, "reload": False}
            if isinstance(request, MessagesRequest) and (request.end_ms-request.start_ms)/1000/3600/24 > 31:
                warning_message = "Downloading messages for this many days is not permitted. Please reduce the range and try again."
                return {"success": False, "message": warning_message, "reload": False}
        return None
    
    def get_data(self, request: Union[DataRequest, CsvRequest, DijkstraRequest]):
        try:
            error = self.check_request(request)
            if error or request.selected_channels==['bids']:
                return error
            
            self.data[request] = {}
            with self.Session() as session:
                import time
                querry_start = time.time()
                print("Querying journaldb...")
                all_raw_messages = session.query(MessageSql).filter(
                    MessageSql.from_alias.like(f'%.{request.house_alias}.%'),
                    MessageSql.message_persisted_ms <= request.end_ms,
                    or_(
                        and_(
                            or_(
                                MessageSql.message_type_name == "batched.readings",
                                MessageSql.message_type_name == "report",
                                MessageSql.message_type_name == "snapshot.spaceheat",
                            ),
                            MessageSql.message_persisted_ms >= request.start_ms,
                        ),
                        and_(
                            MessageSql.message_type_name == "weather.forecast",
                            MessageSql.message_persisted_ms >= request.start_ms - 24*3600*1000,
                        )
                    )
                ).order_by(asc(MessageSql.message_persisted_ms)).all()
                print(f"Time to fetch data: {round(time.time()-querry_start,1)} seconds")

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
    
    async def get_messages(self, request: MessagesRequest):
        try:
            error = self.check_request(request)
            if error:
                return error
            async with async_timeout.timeout(self.timeout_seconds):
                print("Querrying journaldb for messages...")
                with self.Session() as session:
                    messages: List[MessageSql] = session.query(MessageSql).filter(
                        MessageSql.from_alias.like(f'%.{request.house_alias}.%'),
                        MessageSql.message_type_name.in_(request.selected_message_types),
                        MessageSql.message_persisted_ms >= request.start_ms,
                        MessageSql.message_persisted_ms <= request.end_ms,
                    ).order_by(asc(MessageSql.message_persisted_ms)).all()
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
                    times_created.append(str(self.to_datetime(message.payload['CreatedMs']/1000).replace(microsecond=0)))
                
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
            return {"success": False, "message": "An error occured while getting messages", "reload": False}
        
    async def get_csv(self, request: CsvRequest):
        try:
            async with async_timeout.timeout(self.timeout_seconds):
                error = await asyncio.to_thread(self.get_data, request)
                if error:
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
            return {"success": False, "message": "An error occured while getting CSV", "reload": False}
        finally:
            if request in self.data:
                del self.data[request]
                print(f"Deleted request data")
            print(f"Unfinished requests in data: {len(self.data)}")
        
    async def get_flo(self, request: DijkstraRequest):
        try:
            async with async_timeout.timeout(self.timeout_seconds):
                print("Finding latest FLO run...")
                with self.Session() as session:
                    flo_params_msg = session.query(MessageSql).filter(
                        MessageSql.message_type_name == "flo.params.house0",
                        MessageSql.from_alias.like(f'%{request.house_alias}%'),
                        MessageSql.message_persisted_ms >= request.time_ms - 48*3600*1000,
                        MessageSql.message_persisted_ms <= request.time_ms,
                    ).order_by(desc(MessageSql.message_persisted_ms)).first()
                print(f"Found FLO run at {self.to_datetime(flo_params_msg.message_persisted_ms)}")

                if not flo_params_msg:
                    print(f"Could not find a FLO run in the 48 hours prior to {self.to_datetime(request.time_ms)}")
                    if os.path.exists('result.xlsx'):
                        os.remove('result.xlsx')
                    return

                print("Running Dijkstra and saving analysis to excel...")
                houses_in_hinge = []
                flo_params = FloParamsHouse0(**flo_params_msg.payload)
                if request.house_alias in houses_in_hinge:
                    h = FloHinge(flo_params, hinge_hours=5, num_nodes=[10,3,3,3,3])
                    h.export_to_excel()
                else:
                    g = DGraph(flo_params)
                    g.solve_dijkstra()
                    g.export_to_excel()
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
            return {"success": False, "message": "An error occured while getting FLO", "reload": False}
        finally:
            if request in self.data:
                del self.data[request]
                print(f"Deleted request data")
            print(f"Unfinished requests in data: {len(self.data)}")
        
    async def get_bids(self, request: DataRequest):
        try:
            async with async_timeout.timeout(self.timeout_seconds):
                print("Getting bids...")
                
                with self.Session() as session:
                    flo_params_messages = session.query(MessageSql).filter(
                        MessageSql.message_type_name == "flo.params.house0",
                        MessageSql.from_alias.like(f'%{request.house_alias}%'),
                        MessageSql.message_persisted_ms >= request.start_ms,
                        MessageSql.message_persisted_ms <= request.end_ms,
                    ).order_by(desc(MessageSql.payload['StartUnixS'])).all()
                    flo_params_messages = [FloParamsHouse0(**x.payload) for x in flo_params_messages]
                print(f"Found {len(flo_params_messages)} FLOs for {request.house_alias}")

                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    for i in range(len(flo_params_messages)):
                        g = DGraph(flo_params_messages[i])
                        g.solve_dijkstra()
                        pq_pairs = g.generate_bid()
                        prices = [x.PriceTimes1000 for x in pq_pairs]
                        quantities = [x.QuantityTimes1000/1000 for x in pq_pairs]
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
                        prices = [x.PriceTimes1000/1000 for x in pq_pairs]
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
            return {"success": False, "message": "An error occured while getting bids", "reload": False}

    async def get_plots(self, request: DataRequest):
        try:
            async with async_timeout.timeout(self.timeout_seconds):
                error = await asyncio.to_thread(self.get_data, request)
                if error:
                    return error
                
                # If the request is just to plot bids
                if request.selected_channels == ['bids']: 
                    zip_bids = await self.get_bids(request)
                    return zip_bids
                
                # Plot colors depend on user's dark mode settings
                self.plot_background_hex = '#222222' if request.darkmode else 'white'
                self.gridcolor_hex = '#424242' if request.darkmode else 'LightGray'
                self.fontcolor_hex = '#b5b5b5' if request.darkmode else 'rgb(42,63,96)'
                self.home_alone_line = '#f0f0f0' if request.darkmode else '#5e5e5e'
                self.oat_color = 'gray' if request.darkmode else '#d6d6d6'

                # Show markers if the user selected the "show points" option
                self.line_style = 'lines+markers' if 'show-points'in request.selected_channels else 'lines'

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
            return {"success": False, "message": "An error occured while getting plots", "reload": False}
        finally:
            if request in self.data:
                del self.data[request]
                print(f"Deleted request data")
            print(f"Unfinished requests in data: {len(self.data)}")
        
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
                    mode=self.line_style,
                    opacity=0.7,
                    line=dict(color='#d62728', dash='solid'),
                    name='HP LWT'
                    )
                )
        if 'hp-ewt' in request.selected_channels and 'hp-ewt' in self.data[request]['channels']:
            plotting_temperatures = True
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['hp-ewt']['times'], 
                    y=[self.to_fahrenheit(x/1000) for x in self.data[request]['channels']['hp-ewt']['values']], 
                    mode=self.line_style, 
                    opacity=0.7,
                    line=dict(color='#1f77b4', dash='solid'),
                    name='HP EWT'
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
                    mode=self.line_style, 
                    opacity=0.7,
                    line=dict(color='#2ca02c', dash='solid'),
                    name='HP outdoor power',
                    yaxis=y_axis_power
                    )
                )
        if 'hp-idu-pwr' in request.selected_channels and 'hp-idu-pwr' in self.data[request]['channels']:
            plotting_power = True
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['hp-idu-pwr']['times'], 
                    y=[x/1000 for x in self.data[request]['channels']['hp-idu-pwr']['values']], 
                    mode=self.line_style, 
                    opacity=0.7,
                    line=dict(color='#ff7f0e', dash='solid'),
                    name='HP indoor power',
                    yaxis=y_axis_power
                    )
                ) 
        if 'oil-boiler-pwr' in request.selected_channels and 'oil-boiler-pwr' in self.data[request]['channels']:
            plotting_power = True
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['oil-boiler-pwr']['times'], 
                    y=[x/100 for x in self.data[request]['channels']['oil-boiler-pwr']['values']], 
                    mode=self.line_style, 
                    opacity=0.7,
                    line=dict(color=self.home_alone_line, dash='solid'),
                    name='Oil boiler power x10',
                    yaxis=y_axis_power
                    )
                ) 
        if 'primary-flow' in request.selected_channels and 'primary-flow' in self.data[request]['channels']:
            plotting_power = True
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['primary-flow']['times'],
                    y=[x/100 for x in self.data[request]['channels']['primary-flow']['values']], 
                    mode=self.line_style, 
                    opacity=0.4,
                    line=dict(color='purple', dash='solid'),
                    name='Primary pump flow',
                    yaxis=y_axis_power
                    )
                )
        if 'primary-pump-pwr' in request.selected_channels and 'primary-pump-pwr' in self.data[request]['channels']:
            plotting_power = True
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['primary-pump-pwr']['times'], 
                    y=[x/1000*100 for x in self.data[request]['channels']['primary-pump-pwr']['values']], 
                    mode=self.line_style, 
                    opacity=0.7,
                    line=dict(color='pink', dash='solid'),
                    name='Primary pump power x100',
                    yaxis=y_axis_power,
                    visible='legendonly',
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
            plot_bgcolor=self.plot_background_hex,
            paper_bgcolor=self.plot_background_hex,
            font_color=self.fontcolor_hex,
            title_font_color=self.fontcolor_hex,
            xaxis=dict(
                range=[self.data[request]['min_timestamp'], self.data[request]['max_timestamp']],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor=self.fontcolor_hex,
                showgrid=False
                ),
            yaxis=dict(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor=self.fontcolor_hex,
                zeroline=False,
                showgrid=True, 
                gridwidth=1, 
                gridcolor=self.gridcolor_hex
                ),
            yaxis2=dict(
                mirror=True,
                ticks='outside',
                zeroline=False,
                showline=True,
                linecolor=self.fontcolor_hex,
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
        fig.write_html(html_buffer)
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
                    mode=self.line_style, 
                    opacity=0.7,
                    line=dict(color='#d62728', dash='solid'),
                    name='Distribution SWT'
                    )
                )
        if 'dist-rwt' in request.selected_channels and 'dist-rwt' in self.data[request]['channels']:
            plotting_temperatures = True
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['dist-rwt']['times'], 
                    y=[self.to_fahrenheit(x/1000) for x in self.data[request]['channels']['dist-rwt']['values']], 
                    mode=self.line_style, 
                    opacity=0.7,
                    line=dict(color='#1f77b4', dash='solid'),
                    name='Distribution RWT'
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
                    mode=self.line_style, 
                    opacity=0.4,
                    line=dict(color='purple', dash='solid'),
                    name='Distribution flow',
                    yaxis = y_axis_power
                    )
                )
        if 'dist-pump-pwr' in request.selected_channels and 'dist-pump-pwr' in self.data[request]['channels']:
            plotting_power = True
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['dist-pump-pwr']['times'], 
                    y=[x/10 for x in self.data[request]['channels']['dist-pump-pwr']['values']], 
                    mode=self.line_style, 
                    opacity=0.7,
                    line=dict(color='pink', dash='solid'),
                    name='Distribution pump power /10',
                    yaxis = y_axis_power,
                    visible='legendonly', 
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
            plot_bgcolor=self.plot_background_hex,
            paper_bgcolor=self.plot_background_hex,
            font_color=self.fontcolor_hex,
            title_font_color=self.fontcolor_hex,
            margin=dict(t=30, b=30),
            xaxis=dict(
                range=[self.data[request]['min_timestamp'], self.data[request]['max_timestamp']],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor=self.fontcolor_hex,
                showgrid=False
                ),
            yaxis=dict(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor=self.fontcolor_hex,
                zeroline=False,
                showgrid=True, 
                gridwidth=1, 
                gridcolor=self.gridcolor_hex
                ),
            yaxis2=dict(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor=self.fontcolor_hex,
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
        fig.write_html(html_buffer)
        html_buffer.seek(0) 
        print(f"Distribution plot done in {round(time.time()-plot_start,1)} seconds")
        return html_buffer
    
    async def plot_heatcalls(self, request: DataRequest):
        plot_start = time.time()
        fig = go.Figure()
        if 'zone-heat-calls' in request.selected_channels:
            for zone in self.data[request]['channels_by_zone']:
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
                                    name=self.data[request]['channels_by_zone'][zone]['state'].replace('-state',''),
                                    showlegend=False,
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
                                    name=self.data[request]['channels_by_zone'][zone]['state'].replace('-state',''),
                                    showlegend=False,
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
                                    name=self.data[request]['channels_by_zone'][zone]['state'].replace('-state', ''),
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
                        name=self.data[request]['channels_by_zone'][zone]['state'].replace('-state','')
                    )
                )

        fig.update_layout(
            title=dict(text='Heat calls', x=0.5, xanchor='center'),
            plot_bgcolor=self.plot_background_hex,
            paper_bgcolor=self.plot_background_hex,
            font_color=self.fontcolor_hex,
            title_font_color=self.fontcolor_hex,
            margin=dict(t=30, b=30),
            xaxis=dict(
                range=[self.data[request]['min_timestamp'], self.data[request]['max_timestamp']],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor=self.fontcolor_hex,
                showgrid=False
                ),
            yaxis=dict(
                range = [-0.5, len(self.data[request]['channels_by_zone'].keys())*1.3],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor=self.fontcolor_hex,
                zeroline=False,
                showgrid=True, 
                gridwidth=1, 
                gridcolor=self.gridcolor_hex, 
                tickvals=list(range(len(self.data[request]['channels_by_zone'].keys())+1)),
                ),
            yaxis2=dict(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor=self.fontcolor_hex,
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
        fig.write_html(html_buffer)
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
                        mode=self.line_style, 
                        opacity=0.7,
                        line=dict(color=self.zone_color[int(zone[4])-1], dash='solid'),
                        name=self.data[request]['channels_by_zone'][zone]['temp'].replace('-temp','')
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
                        mode=self.line_style, 
                        opacity=0.7,
                        line=dict(color=self.zone_color[int(zone[4])-1], dash='dash'),
                        name=self.data[request]['channels_by_zone'][zone]['set'].replace('-set',''),
                        showlegend=False
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
                    mode=self.line_style, 
                    opacity=0.8,
                    line=dict(color=self.oat_color, dash='solid'),
                    name='Outside air',
                    yaxis='y2',
                    )
                )
            min_oat = self.to_fahrenheit(min(self.data[request]['channels']['oat']['values'])/1000)
            max_oat = self.to_fahrenheit(max(self.data[request]['channels']['oat']['values'])/1000)
            fig.update_layout(yaxis2=dict(title='Outside air temperature [F]'))

        fig.update_layout(yaxis=dict(title='Zone temperature [F]'))
        fig.update_layout(
            title=dict(text='Zones', x=0.5, xanchor='center'),
            plot_bgcolor=self.plot_background_hex,
            paper_bgcolor=self.plot_background_hex,
            font_color=self.fontcolor_hex,
            title_font_color=self.fontcolor_hex,
            margin=dict(t=30, b=30),
            xaxis=dict(
                range=[self.data[request]['min_timestamp'], self.data[request]['max_timestamp']],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor=self.fontcolor_hex,
                showgrid=False
                ),
            yaxis=dict(
                range = [min_zones-30,max_zones+20],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor=self.fontcolor_hex,
                zeroline=False,
                showgrid=True, 
                gridwidth=1, 
                gridcolor=self.gridcolor_hex
                ),
            yaxis2=dict(
                range = [min_oat-2, max_oat+20],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor=self.fontcolor_hex,
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
        fig.write_html(html_buffer)
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
                        mode=self.line_style, 
                        opacity=0.7,
                        name=buffer_channel.replace('buffer-',''),
                        line=dict(color=buffer_layer_colors[buffer_channel], dash='solid')
                        )
                    )  
        if 'buffer-hot-pipe' in request.selected_channels and 'buffer-hot-pipe' in self.data[request]['channels']:
            min_buffer_temp = min(min_buffer_temp, min([self.to_fahrenheit(x/1000) for x in self.data[request]['channels']['buffer-hot-pipe']['values']]))
            max_buffer_temp = max(max_buffer_temp, max([self.to_fahrenheit(x/1000) for x in self.data[request]['channels']['buffer-hot-pipe']['values']]))
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['buffer-hot-pipe']['times'], 
                    y=[self.to_fahrenheit(x/1000) for x in self.data[request]['channels']['buffer-hot-pipe']['values']], 
                    mode=self.line_style, 
                    opacity=0.7,
                    name='Hot pipe',
                    line=dict(color='#d62728', dash='solid')
                    )
                )
        if 'buffer-cold-pipe' in request.selected_channels and 'buffer-cold-pipe' in self.data[request]['channels']:
            min_buffer_temp = min(min_buffer_temp, min([self.to_fahrenheit(x/1000) for x in self.data[request]['channels']['buffer-cold-pipe']['values']]))
            max_buffer_temp = max(max_buffer_temp, max([self.to_fahrenheit(x/1000) for x in self.data[request]['channels']['buffer-cold-pipe']['values']]))
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['buffer-cold-pipe']['times'], 
                    y=[self.to_fahrenheit(x/1000) for x in self.data[request]['channels']['buffer-cold-pipe']['values']], 
                    mode=self.line_style, 
                    opacity=0.7,
                    name='Cold pipe',
                    line=dict(color='#1f77b4', dash='solid')
                    )
                )
               
        fig.update_layout(
            title=dict(text='Buffer', x=0.5, xanchor='center'),
            plot_bgcolor=self.plot_background_hex,
            paper_bgcolor=self.plot_background_hex,
            font_color=self.fontcolor_hex,
            title_font_color=self.fontcolor_hex,
            margin=dict(t=30, b=30),
            xaxis=dict(
                range=[self.data[request]['min_timestamp'], self.data[request]['max_timestamp']],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor=self.fontcolor_hex,
                showgrid=False,
                ),
            yaxis=dict(
                range = [min_buffer_temp-15, max_buffer_temp+30],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor=self.fontcolor_hex,
                title='Temperature [F]', 
                zeroline=False,
                showgrid=True, 
                gridwidth=1, 
                gridcolor=self.gridcolor_hex,
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
        fig.write_html(html_buffer)
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
                        mode=self.line_style, opacity=0.7,
                        name=tank_channel.replace('storage-',''),
                        line=dict(color=storage_layer_colors[tank_channel], dash='solid')
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
                    mode=self.line_style, 
                    opacity=0.7,
                    name='Hot pipe',
                    line=dict(color='#d62728', dash='solid')
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
                    mode=self.line_style, 
                    opacity=0.7,
                    name='Cold pipe',
                    line=dict(color='#1f77b4', dash='solid')
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
                    mode=self.line_style, 
                    opacity=0.7,
                    line=dict(color='pink', dash='solid'),
                    name='Storage pump power x1000',
                    yaxis=y_axis_power,
                    visible='legendonly'
                    )
                )
        if 'store-flow' in request.selected_channels and 'store-flow' in self.data[request]['channels']:
            plotting_power = True
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['store-flow']['times'], 
                    y=[x/100*10 for x in self.data[request]['channels']['store-flow']['values']], 
                    mode=self.line_style, 
                    opacity=0.4,
                    line=dict(color='purple', dash='solid'),
                    name='Storage pump flow x10',
                    yaxis=y_axis_power
                    )
                )
        max_power = 60
        if 'store-energy' in request.selected_channels and 'usable-energy' in self.data[request]['channels']:
            plotting_power = True
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['usable-energy']['times'], 
                    y=[x/1000 for x in self.data[request]['channels']['usable-energy']['values']], 
                    mode=self.line_style, 
                    opacity=0.4,
                    line=dict(color='#2ca02c', dash='solid'),
                    name='Usable',
                    yaxis=y_axis_power,
                    visible='legendonly'
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=self.data[request]['channels']['required-energy']['times'], 
                    y=[x/1000 for x in self.data[request]['channels']['required-energy']['values']], 
                    mode=self.line_style, 
                    opacity=0.4,
                    line=dict(color='#2ca02c', dash='dash'),
                    name='Required',
                    yaxis=y_axis_power,
                    visible='legendonly'
                    )
                )
            max_power = max([x/1000 for x in self.data[request]['channels']['required-energy']['values']])*4
            
        if plotting_temperatures and plotting_power:
            fig.update_layout(yaxis=dict(title='Temperature [F]', range=[min_store_temp-80, max_store_temp+60]))
            fig.update_layout(yaxis2=dict(title='GPM, kW, or kWh', range=[-1, max_power]))
        elif plotting_temperatures and not plotting_power:
            min_store_temp = 20 if min_store_temp<0 else min_store_temp
            fig.update_layout(yaxis=dict(title='Temperature [F]', range=[min_store_temp-20, max_store_temp+60]))
        elif plotting_power and not plotting_temperatures:
            fig.update_layout(yaxis=dict(title='GPM, kW, or kWh'))

        fig.update_layout(
            title=dict(text='Storage', x=0.5, xanchor='center'),
            plot_bgcolor=self.plot_background_hex,
            paper_bgcolor=self.plot_background_hex,
            font_color=self.fontcolor_hex,
            title_font_color=self.fontcolor_hex,
            margin=dict(t=30, b=30),
            xaxis=dict(
                range=[self.data[request]['min_timestamp'], self.data[request]['max_timestamp']],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor=self.fontcolor_hex,
                showgrid=False,
                ),
            yaxis=dict(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor=self.fontcolor_hex,
                zeroline=False,
                showgrid=True, 
                gridwidth=1, 
                gridcolor=self.gridcolor_hex
                ),
            yaxis2=dict(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor=self.fontcolor_hex,
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
        fig.write_html(html_buffer)
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
                    line=dict(color=self.home_alone_line, width=2),
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
                        )
                    )

        fig.update_layout(
            title=dict(text='Top State', x=0.5, xanchor='center'),
            plot_bgcolor=self.plot_background_hex,
            paper_bgcolor=self.plot_background_hex,
            font_color=self.fontcolor_hex,
            title_font_color=self.fontcolor_hex,
            margin=dict(t=30, b=30),
            xaxis=dict(
                range=[self.data[request]['min_timestamp'], self.data[request]['max_timestamp']],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor=self.fontcolor_hex,
                showgrid=False
                ),
            yaxis=dict(
                range = [-0.6, len(self.data[request]['top_states'])-1+0.2],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor=self.fontcolor_hex,
                zeroline=False,
                showgrid=True, 
                gridwidth=1, 
                gridcolor=self.gridcolor_hex, 
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
        fig.write_html(html_buffer)
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
                    line=dict(color=self.home_alone_line, width=2),
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
                        )
                    )

        fig.update_layout(
            title=dict(text='HomeAlone State', x=0.5, xanchor='center'),
            plot_bgcolor=self.plot_background_hex,
            paper_bgcolor=self.plot_background_hex,
            font_color=self.fontcolor_hex,
            title_font_color=self.fontcolor_hex,
            margin=dict(t=30, b=30),
            xaxis=dict(
                range=[self.data[request]['min_timestamp'], self.data[request]['max_timestamp']],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor=self.fontcolor_hex,
                showgrid=False
                ),
            yaxis=dict(
                range = [-0.6, 8-0.8],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor=self.fontcolor_hex,
                zeroline=False,
                showgrid=True, 
                gridwidth=1, 
                gridcolor=self.gridcolor_hex, 
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
        fig.write_html(html_buffer)
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
                    line=dict(color=self.home_alone_line, width=2),
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
                        )
                    )

        fig.update_layout(
            title=dict(text='AtomicAlly State', x=0.5, xanchor='center'),
            plot_bgcolor=self.plot_background_hex,
            paper_bgcolor=self.plot_background_hex,
            font_color=self.fontcolor_hex,
            title_font_color=self.fontcolor_hex,
            margin=dict(t=30, b=30),
            xaxis=dict(
                range=[self.data[request]['min_timestamp'], self.data[request]['max_timestamp']],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor=self.fontcolor_hex,
                showgrid=False
                ),
            yaxis=dict(
                range = [-0.6, 8-0.8],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor=self.fontcolor_hex,
                zeroline=False,
                showgrid=True, 
                gridwidth=1, 
                gridcolor=self.gridcolor_hex, 
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
        fig.write_html(html_buffer)
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
                )
            )

        fig.update_layout(
            title=dict(text='Weather Forecasts', x=0.5, xanchor='center'),
            plot_bgcolor=self.plot_background_hex,
            paper_bgcolor=self.plot_background_hex,
            font_color=self.fontcolor_hex,
            title_font_color=self.fontcolor_hex,
            margin=dict(t=30, b=30),
            xaxis=dict(
                range=[self.data[request]['min_timestamp'], self.data[request]['max_timestamp']],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor=self.fontcolor_hex,
                showgrid=False
                ),
            yaxis=dict(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor=self.fontcolor_hex,
                zeroline=False,
                showgrid=True, 
                gridwidth=1, 
                gridcolor=self.gridcolor_hex, 
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
        fig.write_html(html_buffer)
        html_buffer.seek(0)
        print(f"Weather plot done in {round(time.time()-plot_start,1)} seconds")
        return html_buffer
    
    async def plot_prices(self, request: DataRequest):
        plot_start = time.time()
        fig = go.Figure()

        # Open and read the price CSV file
        csv_times, csv_dist, csv_lmp = [], [], []
        with open('elec_prices.csv', newline='', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)
            for row in csvreader:
                csv_times.append(row[0])
                csv_dist.append(float(row[1]))
                csv_lmp.append(float(row[2])/10)
        csv_times = [pendulum.from_format(x, 'M/D/YY H:m', tz=self.timezone_str).timestamp() for x in csv_times]

        request_hours = int((request.end_ms - request.start_ms)/1000 / 3600)
        price_times_s = [request.start_ms/1000 + x*3600 for x in range(request_hours+2+48)]
        price_values = [lmp for time, dist, lmp in zip(csv_times, csv_dist, csv_lmp) if time in price_times_s]
        csv_times = [time for time in csv_times if time in price_times_s]
        price_times = [self.to_datetime(x*1000) for x in csv_times]

        # Plot LMP
        fig.add_trace(
            go.Scatter(
                x=price_times,
                y=price_values,
                mode='lines',
                line=dict(color=self.home_alone_line),
                opacity=0.8,
                showlegend=False,
                line_shape='hv',
            )
        )

        # Shading on-peak
        shapes_list = []
        for x in price_times:
            x1 = None
            if x==price_times[0] and x.hour in [8,9,10,11]:
                x1 = x+timedelta(hours=5-(x.hour-7))
            elif x.hour==7:
                x1 = x+timedelta(hours=5)
            elif x==price_times[0] and x.hour in [17,18,19]:
                x1 = x+timedelta(hours=4-(x.hour-16))
            elif x.hour==16:
                x1 = x+timedelta(hours=4)
            if x1 and x.hour in [7,8,9,10,11,16,17,18,19] and x.weekday()<5:
                shapes_list.append(
                        go.layout.Shape(
                            type='rect',
                            x0=x, x1=x1,
                            y0=0, y1=1,
                            xref="x", yref="paper",
                            fillcolor="rgba(0, 100, 255, 0.1)",
                            layer="below",
                            line=dict(width=0)
                        )
                    )
        
        fig.update_layout(yaxis=dict(title='LMP [cts/kWh]'))
        fig.update_layout(
            shapes = shapes_list,
            title=dict(text='Price Forecast', x=0.5, xanchor='center'),
            plot_bgcolor=self.plot_background_hex,
            paper_bgcolor=self.plot_background_hex,
            font_color=self.fontcolor_hex,
            title_font_color=self.fontcolor_hex,
            margin=dict(t=30, b=30),
            xaxis=dict(
                range=[self.data[request]['min_timestamp'], self.data[request]['max_timestamp']],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor=self.fontcolor_hex,
                showgrid=False
                ),
            yaxis=dict(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor=self.fontcolor_hex,
                zeroline=False,
                showgrid=True, 
                gridwidth=1, 
                gridcolor=self.gridcolor_hex, 
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
        fig.write_html(html_buffer)
        html_buffer.seek(0)
        print(f"Prices plot done in {round(time.time()-plot_start,1)} seconds")   
        return html_buffer             


if __name__ == "__main__":
    a = VisualizerApi(running_locally=True)
    a.start()
