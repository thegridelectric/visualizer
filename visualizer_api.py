import io
import zipfile
import numpy as np
import pandas as pd
from typing import List, Optional
import time
import asyncio
import async_timeout
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import dotenv
import pendulum
from datetime import timedelta
from pydantic import BaseModel
from sqlalchemy import create_engine, asc, or_, and_
from sqlalchemy.orm import sessionmaker
from fake_config import Settings
from fake_models import MessageSql
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from analysis import download_excel, get_bids
import os
from fastapi.responses import FileResponse
from typing import Union
import plotly.colors as pc
import csv
import uvicorn

class DataRequest(BaseModel):
    house_alias: str
    password: str
    start_ms: int
    end_ms: int
    selected_channels: List[str]
    ip_address: Optional[str] = ''
    user_agent: Optional[str] = ''
    timezone: Optional[str] = ''
    continue_option: Optional[bool] = False
    darkmode: Optional[bool] = False

class CsvRequest(BaseModel):
    house_alias: str
    password: str
    start_ms: int
    end_ms: int
    selected_channels: List[str]
    timestep: int
    continue_option: Optional[bool] = False

class DijkstraRequest(BaseModel):
    house_alias: str
    password: str
    time_ms: int

class MessagesRequest(BaseModel):
    password: str
    selected_message_types: List[str]
    house_alias: str = ""
    start_ms: int 
    end_ms: int
    darkmode: Optional[bool] = False



class VisualizerApi():
    def __init__(self, running_locally):
        self.running_locally = running_locally
        self.get_parameters()
        # Start sqlalchemy session
        self.settings = Settings(_env_file=dotenv.find_dotenv())
        engine = create_engine(self.settings.db_url.get_secret_value())
        self.Session = sessionmaker(bind=engine)
        self.admin_user_password = self.settings.visualizer_api_password.get_secret_value()
        # Start API
        self.app = FastAPI()
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"], # TODO: change to ["https://thegridelectric.github.io"] when ready
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.app.post("/plots")(self.get_plots)
        self.app.post("/csvs")(self.get_csv)
        self.app.post("/messages")(self.get_messages) 

    def run(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8000)       
        
    def get_parameters(self):
        self.timezone_str = 'America/New_York'
        self.timeout_seconds = 5*60
        self.max_days_warning = 3
        self.tank_temperatures = [
            'tank1-depth1', 'tank1-depth2', 'tank1-depth3', 'tank1-depth4', 
            'tank2-depth1', 'tank2-depth2', 'tank2-depth3', 'tank2-depth4', 
            'tank3-depth1', 'tank3-depth2', 'tank3-depth3', 'tank3-depth4'
            ]
        self.top_states_order = ['HomeAlone', 'Atn', 'Dormant']
        self.ha_states_order = ['HpOffStoreDischarge', 'HpOffStoreOff', 'HpOnStoreOff', 
                               'HpOnStoreCharge', 'StratBoss', 'Initializing', 'Dormant']
        self.aa_states_order = self.ha_states_order.copy()
        self.whitewire_threshold_watts = {
            'beech': 100,
            'other': 20,
        }
        gradient = plt.get_cmap('coolwarm', 4)
        buffer_colors = {
            'buffer-depth1': gradient(3),
            'buffer-depth2': gradient(2),
            'buffer-depth3': gradient(1),
            'buffer-depth4': gradient(0)
            }
        self.buffer_layer_colors = {key: self.to_hex(value) for key, value in buffer_colors.items()}
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
        self.storage_layer_colors = {key: self.to_hex(value) for key, value in storage_colors.items()}
        self.zone_color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']*3
        self.top_state_color = {
            'HomeAlone': '#EF553B',
            'Atn': '#00CC96',
            'Admin': '#636EFA'
        }
        self.ha_state_color = {
            'HpOffStoreDischarge': '#EF553B',
            'HpOffStoreOff': '#00CC96',
            'HpOnStoreOff': '#636EFA',
            'HpOnStoreCharge': '#feca52',
            'Initializing': '#a3a3a3',
            'StratBoss': '#ee93fa',
            'Dormant': '#4f4f4f'
        }

    def ms_to_datetime(self, time_ms):
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
        if isinstance(request, Union[DataRequest, CsvRequest]) and not request.continue_option:
            if (request.end_ms - request.start_ms)/1000/60/60/24 > self.max_days_warning:
                warning_message = f"That's a lot of data! Are you sure you want to proceed?"
                return {"success": False, "message": warning_message, "reload": False, "continue_option": True}
        if not self.running_locally: 
            if (request.end_ms-request.start_ms)/1000/60/60/24 > 5 and isinstance(request, isinstance(request, DataRequest, MessagesRequest)):
                warning_message = "Plotting data for this many days is not permitted. Please reduce the range and try again."
                return {"success": False, "message": warning_message, "reload": False}
            if (request.end_ms - request.start_ms)/1000/60/60/24 > 21 and isinstance(request, CsvRequest):
                warning_message = "Downloading data for this many days is not permitted. Please reduce the range and try again."
                return {"success": False, "message": warning_message, "reload": False}
        return {"success": True, "message": "", "reload": False}
    
    def get_data(self, request: Union[DataRequest, CsvRequest, DijkstraRequest]):
        success_status = self.check_request(request)
        if not success_status['success'] or request.selected_channels==['bids']:
            return success_status
        
        session = self.Session()
        self.all_raw_messages = session.query(MessageSql).filter(
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

        if not self.all_raw_messages:
            warning_message = f"No data found for house '{request.house_alias}' in the selected timeframe."
            return {"success": False, "message": warning_message, "reload": False}
        
        # Process reports
        reports: List[MessageSql] = sorted(
            [x for x in self.all_raw_messages if x.message_type_name in ['report', 'batched.readings']],
            key = lambda x: x.message_persisted_ms
            )
        self.channels = {}
        for message in reports:
            for channel in message.payload['ChannelReadingList']:
                if message.message_type_name == 'report':
                    channel_name = channel['ChannelName']
                elif message.message_type_name == 'batched.readings':
                    for dc in message.payload['DataChannelList']:
                        if dc['Id'] == channel['ChannelId']:
                            channel_name = dc['Name']
                if channel_name=='oat' and 'oak' in request.house_alias: #TODO: remove?
                    continue
                if not channel['ValueList'] or not channel['ScadaReadTimeUnixMsList']:
                    continue
                if len(channel['ValueList'])!=len(channel['ScadaReadTimeUnixMsList']):
                    continue
                if channel_name not in self.channels:
                    self.channels[channel_name] = {'values': [], 'times': []}
                self.channels[channel_name]['values'].extend(channel['ValueList'])
                self.channels[channel_name]['times'].extend(channel['ScadaReadTimeUnixMsList'])
            
        # Process snapshots
        max_timestamp = max(max(self.channels[channel_name]['times']) for channel_name in self.channels)
        snapshots = sorted(
                [x for x in self.all_raw_messages if x.message_type_name=='snapshot.spaceheat'
                and x.message_persisted_ms >= max_timestamp], 
                key = lambda x: x.message_persisted_ms
                )
        for snapshot in snapshots:
            for snap in snapshot.payload['LatestReadingList']:
                if snap['ChannelName'] in self.channels:
                    self.channels[snap['ChannelName']]['times'].append(snap['ScadaReadTimeUnixMs'])
                    self.channels[snap['ChannelName']]['values'].append(snap['Value'])

        # Get minimum and maximum timestamp for plots
        max_timestamp = max(max(self.channels[x]['times']) for x in self.channels)
        min_timestamp = min(min(self.channels[x]['times']) for x in self.channels)
        min_timestamp += -(max_timestamp-min_timestamp)*0.05
        max_timestamp += (max_timestamp-min_timestamp)*0.05
        self.min_timestamp = self.ms_to_datetime(min_timestamp)
        self.max_timestamp = self.ms_to_datetime(max_timestamp)

        # Sort values according to time and convert to datetime
        for channel_name in self.channels.keys():
            sorted_times_values = sorted(zip(self.channels[channel_name]['times'], self.channels[channel_name]['values']))
            sorted_times, sorted_values = zip(*sorted_times_values)
            self.channels[channel_name]['values'] = list(sorted_values)
            self.channels[channel_name]['times'] = pd.to_datetime(list(sorted_times), unit='ms', utc=True)
            self.channels[channel_name]['times'] = self.channels[channel_name]['times'].tz_convert(self.timezone_str)
            self.channels[channel_name]['times'] = [x.replace(tzinfo=None) for x in self.channels[channel_name]['times']]        

        # Find all zone channels
        self.channels_by_zone = {}
        for channel_name in self.channels.keys():
            if 'zone' in channel_name and 'gw-temp' not in channel_name:
                zone_name = channel_name.split('-')[0]
                if zone_name not in self.channels_by_zone:
                    self.channels_by_zone[zone_name] = [channel_name]
                else:
                    self.channels_by_zone[zone_name].append(channel_name)
                if 'state' not in channel_name: # TODO: delete and have a convert function, gets self.channels to the right units
                    self.channels[channel_name]['values'] = [x/1000 for x in self.channels[channel_name]['values']]

        # Relays
        relays = {}
        for message in reports:
            if 'StateList' not in message.payload:
                continue
            for state in message.payload['StateList']:
                if state['MachineHandle'] not in relays:
                    relays[state['MachineHandle']] = {'times': [], 'values': []}
                relays[state['MachineHandle']]['times'].extend([self.ms_to_datetime(x) for x in state['UnixMsList']])
                relays[state['MachineHandle']]['values'].extend(state['StateList'])

        # Top state
        self.top_states = {'all': {'times':[], 'values':[]}}
        if 'auto' in relays:
            for time, state in zip(relays['auto']['times'], relays['auto']['values']):
                if state not in self.top_states_order:
                    print(f"Warning: {state} is not a known top state")
                    continue
                if state not in self.top_states:
                    self.top_states[state] = {'time':[], 'values':[]}
                self.top_states['all']['time'].append(time)
                self.top_states['all']['values'].append(self.top_states_order.index(state))
                self.top_states[state]['time'].append(time)
                self.top_states[state]['values'].append(self.top_states_order.index(state))
        if "Dormant" in self.top_states:
            self.top_states['Admin'] = self.top_states['Dormant']
            del self.top_states['Dormant']
        
        # HomeAlone state
        self.ha_states = {'all': {'times':[], 'values':[]}}
        if 'auto.h.n' in relays or 'auto.h' in relays:
            ha_handle = 'auto.h.n' if 'auto.h.n' in relays else 'auto.h'
            for time, state in zip(relays[ha_handle]['times'], relays[ha_handle]['values']):
                if state not in self.ha_states_order:
                    print(f"Warning: {state} is not a known HA state")
                    continue
                if state not in self.ha_states:
                    self.ha_states[state] = {'time':[], 'values':[]}
                self.ha_states['all']['time'].append(time)
                self.ha_states['all']['values'].append(self.ha_states_order.index(state))
                self.ha_states[state]['time'].append(time)
                self.ha_states[state]['values'].append(self.ha_states_order.index(state))

        # AtomicAlly state
        self.aa_states = {'all': {'times':[], 'values':[]}}
        if 'a.aa' in relays:
            for time, state in zip(relays['a.aa']['times'], relays['a.aa']['values']):
                if state not in self.aa_states_order:
                    print(f"Warning: {state} is not a known AA state")
                    continue
                if state not in self.aa_states:
                    self.aa_states[state] = {'time':[], 'values':[]}
                self.aa_states['all']['time'].append(time)
                self.aa_states['all']['values'].append(self.aa_states_order.index(state))
                self.aa_states[state]['time'].append(time)
                self.aa_states[state]['values'].append(self.aa_states_order.index(state))

        # Weather forecasts
        self.weather_forecasts = None
        if isinstance(request, DataRequest):
            self.weather_forecasts = sorted(
                [x for x in self.all_raw_messages if x.message_type_name=='weather.forecast'], 
                key = lambda x: x.message_persisted_ms
                )
            
    async def get_messages(self, request: MessagesRequest):
        success_status = self.check_request(request)
        if not success_status['success']:
            return success_status
        try:
            async with async_timeout.timeout(self.timeout_seconds):
                print(request.selected_message_types)
                session = self.Session()
                messages: List[MessageSql] = session.query(MessageSql).filter(
                    MessageSql.from_alias.like(f'%.{request.house_alias}.%'),
                    MessageSql.message_type_name.in_(request.selected_message_types),
                    MessageSql.message_persisted_ms >= request.start_ms,
                    MessageSql.message_persisted_ms <=request.end_ms,
                ).order_by(asc(MessageSql.message_persisted_ms)).all()
                if not messages:
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
                    times_created.append(str(pendulum.from_timestamp(message.payload['TimeCreatedMs']/1000, tz='America/New_York').replace(microsecond=0)))
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
                    times_created.append(str(pendulum.from_timestamp(message.payload['CreatedMs']/1000, tz='America/New_York').replace(microsecond=0)))
                # Summary table: count of each message type
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
            warning_message = "The data request timed out. Please try loading a smaller amount of data at a time."
            print(warning_message)
            return {"success": False, "message": warning_message, "reload": False}
        except Exception as e:
            return {"success": False, "message": f"An error occurred: {str(e)}", "reload": False}
        
    async def get_plots(self):
        ...

    async def get_csv(self):
        ...

'''
# ------------------------------
# Export as CSV
# ------------------------------

@app.post('/csv')
async def get_csv(request: CsvRequest, apirequest: Request):
    request_start = time.time()
    try:
        async with async_timeout.timeout(self.timeout_seconds):
            
            error_msg, channels, _, __, ___, ____, _____, ______, _______ = await asyncio.to_thread(get_data, request)
            print(f"Time to fetch data: {round(time.time() - request_start,2)} sec")

            if time.time() - request_start > self.timeout_seconds:
                raise asyncio.TimeoutError('Timed out')

            if error_msg != '':
                return error_msg

            if 'all-data' in request.selected_channels:
                channels_to_export = channels.keys()
            else:
                channels_to_export = []
                for channel in request.selected_channels:
                    if channel in channels:
                        channels_to_export.append(channel)
                    elif channel == 'zone-heat-calls':
                        for c in channels.keys():
                            if 'zone' in c:
                                channels_to_export.append(c)
                    elif channel == 'buffer-depths':
                        for c in channels.keys():
                            if 'depth' in c and 'buffer' in c and 'micro' not in c:
                                channels_to_export.append(c)
                    elif channel == 'storage-depths':
                        for c in channels.keys():
                            if 'depth' in c and 'tank' in c and 'micro' not in c:
                                channels_to_export.append(c)
                    elif channel == 'relays':
                        for c in channels.keys():
                            if 'relay' in c:
                                channels_to_export.append(c)
                    elif channel == 'zone-heat-calls':
                        for c in channels.keys():
                            if 'zone' in c:
                                channels_to_export.append(c)
                    elif channel == 'store-energy':
                        for c in channels.keys():
                            if 'required-energy' in c or 'available-energy':
                                channels_to_export.append(c)

            num_points = int((request.end_ms - request.start_ms) / (request.timestep * 1000) + 1)
            
            if num_points * len(channels_to_export) > 3600 * 24 * 10 * len(channels):
                error_message = f"This request would generate {num_points} data points, which is too much data in one go."
                error_message += "\n\nSuggestions:\n- Increase the time step\n- Reduce the number of channels"
                error_message += "\n- Change the start and end times"
                return {"success": False, "message": error_message, "reload": False}

            csv_times = np.linspace(request.start_ms, request.end_ms, num_points)
            csv_times_dt = pd.to_datetime(csv_times, unit='ms', utc=True)
            csv_times_dt = [x.tz_convert('America/New_York').replace(tzinfo=None) for x in csv_times_dt]
            
            csv_values = {}
            for channel in channels_to_export:
                if time.time() - request_start > self.timeout_seconds:
                    raise asyncio.TimeoutError('Timed out')

                merged = await asyncio.to_thread(pd.merge_asof, 
                                                  pd.DataFrame({'times': csv_times_dt}),
                                                  pd.DataFrame(channels[channel]),
                                                  on='times',
                                                  direction='backward')
                csv_values[channel] = list(merged['values'])

            df = pd.DataFrame(csv_values)
            df['timestamps'] = csv_times_dt
            df = df[['timestamps'] + [col for col in df.columns if col != 'timestamps']]

            csv_buffer = io.StringIO()
            start_date = pendulum.from_timestamp(request.start_ms / 1000) 
            end_date = pendulum.from_timestamp(request.end_ms / 1000) 
            formatted_start_date = start_date.to_iso8601_string()[:16].replace('T', '-')
            formatted_end_date = end_date.to_iso8601_string()[:16].replace('T', '-')
            filename = f'{request.house_alias}_{request.timestep}s_{formatted_start_date}-{formatted_end_date}.csv'.replace(':','_')
            csv_buffer.write(filename+'\n')
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            response = StreamingResponse(
                iter([csv_buffer.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
            return response

    except asyncio.TimeoutError:
        print("Request timed out.")
        return {
            "success": False, 
            "message": "The data request timed out. Please try loading a smaller amount of data at a time.", 
            "reload": False
        }
    except Exception as e:
        return {
            "success": False, 
            "message": f"An error occurred: {str(e)}", 
            "reload": False
        }

# ------------------------------
# Generate interactive plots
# ------------------------------

@app.post('/plots')
async def get_plots(request: Union[DataRequest, DijkstraRequest], apirequest: Request):

    if isinstance(request, DijkstraRequest):

        download_excel(request.house_alias, request.time_ms)
        
        if os.path.exists('result.xlsx'):
            return FileResponse(
                'result.xlsx',
                media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                headers={"Content-Disposition": "attachment; filename=file.xlsx"}
            )
        else:
            return {"error": "File not found"}


    request_start = time.time()
    try:
        async with async_timeout.timeout(self.timeout_seconds):
            
            error_msg, channels, zones, self.ha_states, self.top_states, aa_modes, weather, self.min_timestamp, self.max_timestamp = await asyncio.to_thread(get_data, request)
            print(f"Time to fetch data: {round(time.time() - request_start,2)} sec")
            request_start = time.time()

            if request.selected_channels == ['bids']:
                zip_bids = get_bids(request.house_alias, request.start_ms, request.end_ms)
                return zip_bids
    
            if error_msg != '':
                return error_msg
            
            if request.darkmode:
                plot_background_hex = '#222222'
                gridcolor_hex = '#424242'
                fontcolor_hex = '#b5b5b5'
                home_alone_line = '#f0f0f0'
                oat_color = 'gray'
            else:
                plot_background_hex = 'white'
                gridcolor_hex = 'LightGray'
                fontcolor_hex = 'rgb(42,63,96)'
                home_alone_line = '#5e5e5e'
                oat_color = '#d6d6d6'

            line_style = 'lines+markers' if 'show-points'in request.selected_channels else 'lines'

            # --------------------------------------
            # PLOT 1: Heat pump
            # --------------------------------------

            if time.time() - request_start > self.timeout_seconds:
                raise asyncio.TimeoutError('Timed out')

            fig = go.Figure()

            # Temperature
            temp_plot = False
            if 'hp-lwt' in request.selected_channels and 'hp-lwt' in channels:
                temp_plot = True
                fig.add_trace(
                    go.Scatter(
                        x=channels['hp-lwt']['times'], 
                        y=[to_fahrenheit(x/1000) for x in channels['hp-lwt']['values']], 
                        mode=line_style,
                        opacity=0.7,
                        line=dict(color='#d62728', dash='solid'),
                        name='HP LWT'
                        )
                    )
            if 'hp-ewt' in request.selected_channels and 'hp-ewt' in channels:
                temp_plot = True
                fig.add_trace(
                    go.Scatter(
                        x=channels['hp-ewt']['times'], 
                        y=[to_fahrenheit(x/1000) for x in channels['hp-ewt']['values']], 
                        mode=line_style, 
                        opacity=0.7,
                        line=dict(color='#1f77b4', dash='solid'),
                        name='HP EWT'
                        )
                    )
                
            if time.time() - request_start > self.timeout_seconds:
                raise asyncio.TimeoutError('Timed out')

            # Secondary yaxis
            y_axis_power = 'y2' if temp_plot else 'y'
                
            # Power and flow
            power_plot = False
            if 'hp-odu-pwr' in request.selected_channels and 'hp-odu-pwr' in channels:
                power_plot = True
                fig.add_trace(
                    go.Scatter(
                        x=channels['hp-odu-pwr']['times'], 
                        y=[x/1000 for x in channels['hp-odu-pwr']['values']], 
                        mode=line_style, 
                        opacity=0.7,
                        line=dict(color='#2ca02c', dash='solid'),
                        name='HP outdoor power',
                        yaxis=y_axis_power
                        )
                    )
            if 'hp-idu-pwr' in request.selected_channels and 'hp-idu-pwr' in channels:
                power_plot = True
                fig.add_trace(
                    go.Scatter(
                        x=channels['hp-idu-pwr']['times'], 
                        y=[x/1000 for x in channels['hp-idu-pwr']['values']], 
                        mode=line_style, 
                        opacity=0.7,
                        line=dict(color='#ff7f0e', dash='solid'),
                        name='HP indoor power',
                        yaxis=y_axis_power
                        )
                    ) 
            if 'oil-boiler-pwr' in request.selected_channels and 'oil-boiler-pwr' in channels:
                power_plot = True
                fig.add_trace(
                    go.Scatter(
                        x=channels['oil-boiler-pwr']['times'], 
                        y=[x/100 for x in channels['oil-boiler-pwr']['values']], 
                        mode=line_style, 
                        opacity=0.7,
                        line=dict(color=home_alone_line, dash='solid'),
                        name='Oil boiler power x10',
                        yaxis=y_axis_power
                        )
                    ) 
            if time.time() - request_start > self.timeout_seconds:
                raise asyncio.TimeoutError('Timed out')

            if 'primary-flow' in request.selected_channels and 'primary-flow' in channels:
                power_plot = True
                fig.add_trace(
                    go.Scatter(
                        x=channels['primary-flow']['times'],
                        y=[x/100 for x in channels['primary-flow']['values']], 
                        mode=line_style, 
                        opacity=0.4,
                        line=dict(color='purple', dash='solid'),
                        name='Primary pump flow',
                        yaxis=y_axis_power
                        )
                    )
            if 'primary-pump-pwr' in request.selected_channels and 'primary-pump-pwr' in channels:
                power_plot = True
                fig.add_trace(
                    go.Scatter(
                        x=channels['primary-pump-pwr']['times'], 
                        y=[x/1000*100 for x in channels['primary-pump-pwr']['values']], 
                        mode=line_style, 
                        opacity=0.7,
                        line=dict(color='pink', dash='solid'),
                        name='Primary pump power x100',
                        yaxis=y_axis_power,
                        visible='legendonly',
                        )
                    )

            if power_plot and temp_plot:
                fig.update_layout(yaxis=dict(title='Temperature [F]', range=[0,260]))
                fig.update_layout(yaxis2=dict(title='Power [kW] or Flow [GPM]', range=[0,35]))
            elif temp_plot and not power_plot:
                fig.update_layout(yaxis=dict(title='Temperature [F]'))
            elif power_plot and not temp_plot:
                fig.update_layout(yaxis=dict(title='Power [kW] or Flow [GPM]', range=[0,10]))
            
            fig.update_layout(
                title=dict(text='Heat pump', x=0.5, xanchor='center'),
                margin=dict(t=30, b=30),
                plot_bgcolor=plot_background_hex,
                paper_bgcolor=plot_background_hex,
                font_color=fontcolor_hex,
                title_font_color=fontcolor_hex,
                xaxis=dict(
                    range=[self.min_timestamp, self.max_timestamp],
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor=fontcolor_hex,
                    showgrid=False
                    ),
                yaxis=dict(
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor=fontcolor_hex,
                    zeroline=False,
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor=gridcolor_hex
                    ),
                yaxis2=dict(
                    mirror=True,
                    ticks='outside',
                    zeroline=False,
                    showline=True,
                    linecolor=fontcolor_hex,
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

            html_buffer1 = io.StringIO()
            fig.write_html(html_buffer1)
            html_buffer1.seek(0)

            print(f"Plot 1 (heat pump) done in {round(time.time()-request_start,1)} seconds")
            request_start = time.time()

            # --------------------------------------
            # PLOT 2: Distribution
            # --------------------------------------

            if time.time() - request_start > self.timeout_seconds:
                raise asyncio.TimeoutError('Timed out')

            fig = go.Figure()

            # Temperature
            temp_plot = False
            if 'dist-swt' in request.selected_channels and 'dist-swt' in channels:
                temp_plot = True
                fig.add_trace(
                    go.Scatter(
                        x=channels['dist-swt']['times'], 
                        y=[to_fahrenheit(x/1000) for x in channels['dist-swt']['values']], 
                        mode=line_style, 
                        opacity=0.7,
                        line=dict(color='#d62728', dash='solid'),
                        name='Distribution SWT'
                        )
                    )
            if 'dist-rwt' in request.selected_channels and 'dist-rwt' in channels:
                temp_plot = True
                fig.add_trace(
                    go.Scatter(
                        x=channels['dist-rwt']['times'], 
                        y=[to_fahrenheit(x/1000) for x in channels['dist-rwt']['values']], 
                        mode=line_style, 
                        opacity=0.7,
                        line=dict(color='#1f77b4', dash='solid'),
                        name='Distribution RWT'
                        )
                    )
                
            # Secondary yaxis
            y_axis_power = 'y2' if temp_plot else 'y'

            # Power and flow
            power_plot = False   
            if 'dist-flow' in request.selected_channels and 'dist-flow' in channels:
                power_plot = True
                fig.add_trace(
                    go.Scatter(
                        x=channels['dist-flow']['times'], 
                        y=[x/100 for x in channels['dist-flow']['values']], 
                        mode=line_style, 
                        opacity=0.4,
                        line=dict(color='purple', dash='solid'),
                        name='Distribution flow',
                        yaxis = y_axis_power
                        )
                    )
            if 'dist-pump-pwr' in request.selected_channels and 'dist-pump-pwr' in channels:
                power_plot = True
                fig.add_trace(
                    go.Scatter(
                        x=channels['dist-pump-pwr']['times'], 
                        y=[x/10 for x in channels['dist-pump-pwr']['values']], 
                        mode=line_style, 
                        opacity=0.7,
                        line=dict(color='pink', dash='solid'),
                        name='Distribution pump power /10',
                        yaxis = y_axis_power,
                        visible='legendonly', 
                        )
                    )
                
            if temp_plot and power_plot:
                fig.update_layout(yaxis=dict(title='Temperature [F]', range=[0,260]))
                fig.update_layout(yaxis2=dict(title='Flow [GPM] or Power [W]', range=[0,20]))
            elif temp_plot and not power_plot:
                fig.update_layout(yaxis=dict(title='Temperature [F]'))
            elif power_plot and not temp_plot:
                fig.update_layout(yaxis=dict('Flow [GPM] or Power [W]'))

            fig.update_layout(
                title=dict(text='Distribution', x=0.5, xanchor='center'),
                plot_bgcolor=plot_background_hex,
                paper_bgcolor=plot_background_hex,
                font_color=fontcolor_hex,
                title_font_color=fontcolor_hex,
                margin=dict(t=30, b=30),
                xaxis=dict(
                    range=[self.min_timestamp, self.max_timestamp],
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor=fontcolor_hex,
                    showgrid=False
                    ),
                yaxis=dict(
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor=fontcolor_hex,
                    zeroline=False,
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor=gridcolor_hex
                    ),
                yaxis2=dict(
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor=fontcolor_hex,
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

            html_buffer2 = io.StringIO()
            fig.write_html(html_buffer2)
            html_buffer2.seek(0) 

            print(f"Plot 2 (distribution) done in {round(time.time()-request_start,1)} seconds")
            request_start = time.time()

            # --------------------------------------
            # PLOT 3: Heat calls
            # --------------------------------------

            if time.time() - request_start > self.timeout_seconds:
                raise asyncio.TimeoutError('Timed out')

            fig = go.Figure()

            if 'zone-heat-calls' in request.selected_channels:
                for zone in zones:
                    shape_start = None
                    for key in [x for x in zones[zone] if 'whitewire' in x]:
                        if request.house_alias not in self.whitewire_threshold_watts:
                            house_threshold = self.whitewire_threshold_watts['other']
                        else:
                            house_threshold = self.whitewire_threshold_watts[request.house_alias]
                        channels[key]['values'] = [1 if abs(x*1000)>house_threshold else 0 for x in channels[key]['values']]
                        
                    for key2 in [x for x in zones[zone] if 'state' in x]:
                        # find the corresponding key in whitewire
                        key = key2
                        for cn in [x for x in zones[zone] if 'whitewire' in x]:
                            if cn.split('-')[0] == key2.split('-')[0]:
                                key = cn

                        zone_color = self.zone_color[int(key[4])-1]
                        last_was_1 = False
                        fig.add_trace(
                            go.Scatter(
                                x=[channels[key]['times'][0], channels[key]['times'][0]],
                                y=[int(key[4])-1, int(key[4])],
                                mode='lines',
                                line=dict(color=zone_color, width=2),
                                opacity=0,
                                name=key.replace('-state',''),
                                showlegend=False,
                            )
                        )
                        for i in range(len(channels[key]['values'])):
                            if channels[key]['values'][i] == 1:
                                if not last_was_1 or 'show-points' in request.selected_channels:
                                    if i>0: 
                                        fig.add_trace(
                                            go.Scatter(
                                                x=[channels[key]['times'][i], channels[key]['times'][i]],
                                                y=[int(key[4])-1, int(key[4])],
                                                mode='lines',
                                                line=dict(color=zone_color, width=2),
                                                opacity=0.7,
                                                name=key.replace('-state',''),
                                                showlegend=False,
                                            )
                                        )
                                if i<len(channels[key]['values'])-1:
                                    last_was_1 = True
                                    if not shape_start:
                                        shape_start = channels[key]['times'][i]
                                    if channels[key]['values'][i+1] != 1 or i+1==len(channels[key]['values'])-1:
                                        if channels[key]['values'][i+1] != 1:
                                            fig.add_trace(
                                                go.Scatter(
                                                    x=[channels[key]['times'][i+1], channels[key]['times'][i+1]],
                                                    y=[int(key[4])-1, int(key[4])],
                                                    mode='lines',
                                                    line=dict(color=zone_color, width=2),
                                                    opacity=0.7,
                                                    name=key.replace('-state',''),
                                                    showlegend=False,
                                                )
                                            )
                                        if shape_start:
                                            fig.add_shape(
                                                type='rect',
                                                x0=shape_start,
                                                y0=int(key[4]) - 1,
                                                x1=channels[key]['times'][i+1],
                                                y1=int(key[4]),
                                                line=dict(color=zone_color, width=0),
                                                fillcolor=zone_color,
                                                opacity=0.2,
                                                name=key.replace('-state', ''),
                                            )
                                            shape_start = None
                                        
                            
                            else:
                                last_was_1 = False
                        fig.add_trace(
                            go.Scatter(
                                x=[None], 
                                y=[None],
                                mode='lines',
                                line=dict(color=zone_color, width=2),
                                name=key.replace('-state','')
                            )
                        )

            fig.update_layout(
                title=dict(text='Heat calls', x=0.5, xanchor='center'),
                plot_bgcolor=plot_background_hex,
                paper_bgcolor=plot_background_hex,
                font_color=fontcolor_hex,
                title_font_color=fontcolor_hex,
                margin=dict(t=30, b=30),
                xaxis=dict(
                    range=[self.min_timestamp, self.max_timestamp],
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor=fontcolor_hex,
                    showgrid=False
                    ),
                yaxis=dict(
                    range = [-0.5, len(zones.keys())*1.3],
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor=fontcolor_hex,
                    zeroline=False,
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor=gridcolor_hex, 
                    tickvals=list(range(len(zones.keys())+1)),
                    ),
                yaxis2=dict(
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor=fontcolor_hex,
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

            html_buffer3 = io.StringIO()
            fig.write_html(html_buffer3)
            html_buffer3.seek(0)

            print(f"Plot 3 (heat calls) done in {round(time.time()-request_start,1)} seconds")
            request_start = time.time()

            # --------------------------------------
            # PLOT 4: Zones
            # --------------------------------------

            if time.time() - request_start > self.timeout_seconds:
                raise asyncio.TimeoutError('Timed out')

            fig = go.Figure()

            min_zones, max_zones = 45, 80
            for zone in zones:
                for key in zones[zone]:
                    if 'temp' in key:
                        fig.add_trace(
                            go.Scatter(
                                x=channels[key]['times'], 
                                y=channels[key]['values'], 
                                mode=line_style, 
                                opacity=0.7,
                                line=dict(color=self.zone_color[int(key[4])-1], dash='solid'),
                                name=key.replace('-temp','')
                                )
                            )
                        min_temp = min(channels[key]['values'])
                        max_temp = max(channels[key]['values'])
                        if min_temp < min_zones:
                            min_zones = min_temp
                        if max_temp > max_zones:
                            max_zones = max_temp
                    elif 'set' in key:
                        fig.add_trace(
                            go.Scatter(
                                x=channels[key]['times'], 
                                y=channels[key]['values'], 
                                mode=line_style, 
                                opacity=0.7,
                                line=dict(color=self.zone_color[int(key[4])-1], dash='dash'),
                                name=key.replace('-set',''),
                                showlegend=False
                                )
                            )
                        min_temp = min(channels[key]['values'])
                        max_temp = max(channels[key]['values'])
                        if min_temp < min_zones:
                            min_zones = min_temp
                        if max_temp > max_zones:
                            max_zones = max_temp

            min_oat, max_oat = 70, 80    
            if 'oat' in request.selected_channels and 'oat' in channels:
                fig.add_trace(
                    go.Scatter(
                        x=channels['oat']['times'], 
                        y=[to_fahrenheit(x/1000) for x in channels['oat']['values']], 
                        mode=line_style, 
                        opacity=0.8,
                        line=dict(color=oat_color, dash='solid'),
                        name='Outside air',
                        yaxis='y2',
                        )
                    )
                min_oat = to_fahrenheit(min(channels['oat']['values'])/1000)
                max_oat = to_fahrenheit(max(channels['oat']['values'])/1000)
                fig.update_layout(yaxis2=dict(title='Outside air temperature [F]'))
            
            fig.update_layout(yaxis=dict(title='Zone temperature [F]'))

            fig.update_layout(
                title=dict(text='Zones', x=0.5, xanchor='center'),
                plot_bgcolor=plot_background_hex,
                paper_bgcolor=plot_background_hex,
                font_color=fontcolor_hex,
                title_font_color=fontcolor_hex,
                margin=dict(t=30, b=30),
                xaxis=dict(
                    range=[self.min_timestamp, self.max_timestamp],
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor=fontcolor_hex,
                    showgrid=False
                    ),
                yaxis=dict(
                    range = [min_zones-30,max_zones+20],
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor=fontcolor_hex,
                    zeroline=False,
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor=gridcolor_hex
                    ),
                yaxis2=dict(
                    range = [min_oat-2, max_oat+20],
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor=fontcolor_hex,
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

            html_buffer4 = io.StringIO()
            fig.write_html(html_buffer4)
            html_buffer4.seek(0)

            print(f"Plot 4 (zones) done in {round(time.time()-request_start,1)} seconds")
            request_start = time.time()

            # --------------------------------------
            # PLOT 5: Buffer
            # --------------------------------------

            if time.time() - request_start > self.timeout_seconds:
                raise asyncio.TimeoutError('Timed out')

            fig = go.Figure()

            min_buffer_temp = 1e5
            max_buffer_temp = 0
            buffer_channels = []

            if 'buffer-depths' in request.selected_channels:
                buffer_channels = sorted([key for key in channels.keys() if 'buffer-depth' in key and 'micro-v' not in key])
                for buffer_channel in buffer_channels:
                    yf = [to_fahrenheit(x/1000) for x in channels[buffer_channel]['values']]
                    min_buffer_temp = min(min_buffer_temp, min(yf))
                    max_buffer_temp = max(max_buffer_temp, max(yf))
                    fig.add_trace(
                        go.Scatter(
                            x=channels[buffer_channel]['times'], 
                            y=yf, 
                            mode=line_style, 
                            opacity=0.7,
                            name=buffer_channel.replace('buffer-',''),
                            line=dict(color=self.buffer_layer_colors[buffer_channel], dash='solid')
                            )
                        )
            
            if 'buffer-hot-pipe' in request.selected_channels and 'buffer-hot-pipe' in channels:
                yf = [to_fahrenheit(x/1000) for x in channels['buffer-hot-pipe']['values']]
                min_buffer_temp = min(min_buffer_temp, min(yf))
                max_buffer_temp = max(max_buffer_temp, max(yf))
                fig.add_trace(
                    go.Scatter(
                        x=channels['buffer-hot-pipe']['times'], 
                        y=yf, 
                        mode=line_style, 
                        opacity=0.7,
                        name='Hot pipe',
                        line=dict(color='#d62728', dash='solid')
                        )
                    )
            if 'buffer-cold-pipe' in request.selected_channels and 'buffer-cold-pipe' in channels:
                yf = [to_fahrenheit(x/1000) for x in channels['buffer-cold-pipe']['values']]
                min_buffer_temp = min(min_buffer_temp, min(yf))
                max_buffer_temp = max(max_buffer_temp, max(yf))
                fig.add_trace(
                    go.Scatter(
                        x=channels['buffer-cold-pipe']['times'], 
                        y=yf, 
                        mode=line_style, 
                        opacity=0.7,
                        name='Cold pipe',
                        line=dict(color='#1f77b4', dash='solid')
                        )
                    )
                    
            fig.update_layout(
                title=dict(text='Buffer', x=0.5, xanchor='center'),
                plot_bgcolor=plot_background_hex,
                paper_bgcolor=plot_background_hex,
                font_color=fontcolor_hex,
                title_font_color=fontcolor_hex,
                margin=dict(t=30, b=30),
                xaxis=dict(
                    range=[self.min_timestamp, self.max_timestamp],
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor=fontcolor_hex,
                    showgrid=False,
                    ),
                yaxis=dict(
                    range = [min_buffer_temp-15, max_buffer_temp+30],
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor=fontcolor_hex,
                    title='Temperature [F]', 
                    zeroline=False,
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor=gridcolor_hex,
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

            html_buffer5 = io.StringIO()
            fig.write_html(html_buffer5)
            html_buffer5.seek(0)

            print(f"Plot 5 (buffer) done in {round(time.time()-request_start,1)} seconds")
            request_start = time.time()

            # --------------------------------------
            # PLOT 6: Storage
            # --------------------------------------

            if time.time() - request_start > self.timeout_seconds:
                raise asyncio.TimeoutError('Timed out')

            fig = go.Figure()
            
            # Temperature
            temp_plot = False
            min_store_temp = 1e5
            max_store_temp = 0
            tank_channels = []

            if 'storage-depths' in request.selected_channels:
                temp_plot = True
                tank_channels = sorted([key for key in channels.keys() if 'tank' in key and 'micro-v' not in key])
                for tank_channel in tank_channels:
                    yf = [to_fahrenheit(x/1000) for x in channels[tank_channel]['values']]
                    min_store_temp = min(min_store_temp, min(yf))
                    max_store_temp = max(max_store_temp, max(yf))
                    fig.add_trace(
                        go.Scatter(x=channels[tank_channel]['times'], y=yf, 
                        mode=line_style, opacity=0.7,
                        name=tank_channel.replace('storage-',''),
                        line=dict(color=self.storage_layer_colors[tank_channel], dash='solid'))
                        )

            if ('thermocline' in request.selected_channels 
                and 'thermocline-position' in channels
                and 'top-centroid' in channels
                and 'bottom-centroid' in channels):
                fig.add_trace(
                    go.Scatter(
                        x=channels['top-centroid']['times'],
                        y=[x/1000 for x in channels['top-centroid']['values']], 
                        mode='markers', opacity=0.7,
                        name='centroids',
                        line=dict(color='yellow', dash='solid'),
                        showlegend=False),
                    )
                fig.add_trace(
                    go.Scatter(
                        x=channels['bottom-centroid']['times'],
                        y=[x/1000 for x in channels['bottom-centroid']['values']], 
                        mode='markers', opacity=0.7,
                        name='centroids',
                        line=dict(color='yellow', dash='solid'),
                        showlegend=False),
                    ) 
                thermocline_temps = []
                for i in range(len(channels['thermocline-position']['times'])):
                    x = channels['thermocline-position']['values'][i] - 1
                    thermoc_time = channels['thermocline-position']['times'][i]
                    times = channels[tank_temperatures[x]]['times']
                    values = [to_fahrenheit(y/1000) for y in channels[tank_temperatures[x]]['values']]
                    idx = min(range(len(times)), key=lambda i: abs(times[i] - thermoc_time))
                    thermocline_temps.append(values[idx])
                fig.add_trace(
                    go.Scatter(
                        x=channels['thermocline-position']['times'],
                        y=thermocline_temps,
                        mode='markers', opacity=0.7,
                        name=f'thermocline',
                        line=dict(color='green', dash='solid'),
                        showlegend=False),
                    )
            
            if 'store-hot-pipe' in request.selected_channels and 'store-hot-pipe' in channels:
                temp_plot = True
                yf = [to_fahrenheit(x/1000) for x in channels['store-hot-pipe']['values']]
                min_store_temp = min(min_store_temp, min(yf))
                max_store_temp = max(max_store_temp, max(yf))
                fig.add_trace(
                    go.Scatter(
                        x=channels['store-hot-pipe']['times'], 
                        y=yf, 
                        mode=line_style, 
                        opacity=0.7,
                        name='Hot pipe',
                        line=dict(color='#d62728', dash='solid'))
                    )
            if 'store-cold-pipe' in request.selected_channels and 'store-cold-pipe' in channels:
                temp_plot = True
                yf = [to_fahrenheit(x/1000) for x in channels['store-cold-pipe']['values']]
                min_store_temp = min(min_store_temp, min(yf))
                max_store_temp = max(max_store_temp, max(yf))
                fig.add_trace(
                    go.Scatter(
                        x=channels['store-cold-pipe']['times'], 
                        y=yf, 
                        mode=line_style, 
                        opacity=0.7,
                        name='Cold pipe',
                        line=dict(color='#1f77b4', dash='solid'))
                    )

            # Secondary yaxis
            y_axis_power = 'y2' if temp_plot else 'y'

            # Power
            power_plot = False
            max_power = 60
            if 'store-pump-pwr' in request.selected_channels and 'store-pump-pwr' in channels:
                power_plot = True
                fig.add_trace(
                    go.Scatter(
                        x=channels['store-pump-pwr']['times'], 
                        y=[x for x in channels['store-pump-pwr']['values']], 
                        mode=line_style, 
                        opacity=0.7,
                        line=dict(color='pink', dash='solid'),
                        name='Storage pump power x1000',
                        yaxis=y_axis_power,
                        visible='legendonly'
                        )
                    )
            if 'store-flow' in request.selected_channels and 'store-flow' in channels:
                power_plot = True
                fig.add_trace(
                    go.Scatter(
                        x=channels['store-flow']['times'], 
                        y=[x/100*10 for x in channels['store-flow']['values']], 
                        mode=line_style, 
                        opacity=0.4,
                        line=dict(color='purple', dash='solid'),
                        name='Storage pump flow x10',
                        yaxis=y_axis_power
                        )
                    )
            if 'store-energy' in request.selected_channels and 'usable-energy' in channels:
                power_plot = True
                fig.add_trace(
                    go.Scatter(
                        x=channels['usable-energy']['times'], 
                        y=[x/1000 for x in channels['usable-energy']['values']], 
                        mode=line_style, 
                        opacity=0.4,
                        line=dict(color='#2ca02c', dash='solid'),
                        name='Usable',
                        yaxis=y_axis_power
                        )
                    )
                fig.add_trace(
                    go.Scatter(
                        x=channels['required-energy']['times'], 
                        y=[x/1000 for x in channels['required-energy']['values']], 
                        mode=line_style, 
                        opacity=0.4,
                        line=dict(color='#2ca02c', dash='dash'),
                        name='Required',
                        yaxis=y_axis_power
                        )
                    )
                max_power = max([x/1000 for x in channels['required-energy']['values']])*4
                
            if temp_plot and power_plot:
                fig.update_layout(yaxis=dict(title='Temperature [F]', range=[min_store_temp-80, max_store_temp+60]))
                fig.update_layout(yaxis2=dict(title='GPM, kW, or kWh', range=[-1, max_power]))
            elif temp_plot and not power_plot:
                min_store_temp = 20 if min_store_temp<0 else min_store_temp
                fig.update_layout(yaxis=dict(title='Temperature [F]', range=[min_store_temp-20, max_store_temp+60]))
            elif power_plot and not temp_plot:
                fig.update_layout(yaxis=dict(title='GPM, kW, or kWh'))

            fig.update_layout(
                title=dict(text='Storage', x=0.5, xanchor='center'),
                plot_bgcolor=plot_background_hex,
                paper_bgcolor=plot_background_hex,
                font_color=fontcolor_hex,
                title_font_color=fontcolor_hex,
                margin=dict(t=30, b=30),
                xaxis=dict(
                    range=[self.min_timestamp, self.max_timestamp],
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor=fontcolor_hex,
                    showgrid=False,
                    ),
                yaxis=dict(
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor=fontcolor_hex,
                    zeroline=False,
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor=gridcolor_hex
                    ),
                yaxis2=dict(
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor=fontcolor_hex,
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

            html_buffer6 = io.StringIO()
            fig.write_html(html_buffer6)
            html_buffer6.seek(0)

            print(f"Plot 6 (storage) done in {round(time.time()-request_start,1)} seconds")
            request_start = time.time()

            # --------------------------------------
            # PLOT 7: Top State
            # --------------------------------------

            if time.time() - request_start > self.timeout_seconds:
                raise asyncio.TimeoutError('Timed out')

            fig = go.Figure()

            if self.top_states!={}:

                fig.add_trace(
                    go.Scatter(
                        x=self.top_states['all']['times'],
                        y=self.top_states['all']['values'],
                        mode='lines',
                        line=dict(color=home_alone_line, width=2),
                        opacity=0.3,
                        showlegend=False,
                        line_shape='hv'
                    )
                )

                for state in self.top_states.keys():
                    if state != 'all' and state in self.top_state_color:
                        fig.add_trace(
                            go.Scatter(
                                x=self.top_states[state]['times'],
                                y=self.top_states[state]['values'],
                                mode='markers',
                                marker=dict(color=self.top_state_color[state], size=10),
                                opacity=0.8,
                                name=state,
                            )
                        )

            fig.update_layout(
                title=dict(text='Top State', x=0.5, xanchor='center'),
                plot_bgcolor=plot_background_hex,
                paper_bgcolor=plot_background_hex,
                font_color=fontcolor_hex,
                title_font_color=fontcolor_hex,
                margin=dict(t=30, b=30),
                xaxis=dict(
                    range=[self.min_timestamp, self.max_timestamp],
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor=fontcolor_hex,
                    showgrid=False
                    ),
                yaxis=dict(
                    range = [-0.6, len(self.top_states)-1+0.2],
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor=fontcolor_hex,
                    zeroline=False,
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor=gridcolor_hex, 
                    tickvals=list(range(len(self.top_states)-1)),
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

            html_buffer7 = io.StringIO()
            fig.write_html(html_buffer7)
            html_buffer7.seek(0)

            print(f"Plot 7 (top state) done in {round(time.time()-request_start,1)} seconds")
            request_start = time.time()

            # --------------------------------------
            # PLOT 8: HomeAlone
            # --------------------------------------

            if time.time() - request_start > self.timeout_seconds:
                raise asyncio.TimeoutError('Timed out')

            fig = go.Figure()

            if self.ha_states!={}:

                fig.add_trace(
                    go.Scatter(
                        x=self.ha_states['all']['times'],
                        y=self.ha_states['all']['values'],
                        mode='lines',
                        line=dict(color=home_alone_line, width=2),
                        opacity=0.3,
                        showlegend=False,
                        line_shape='hv'
                    )
                )

                for state in self.ha_states.keys():
                    if state != 'all' and state in self.ha_state_color:
                        fig.add_trace(
                            go.Scatter(
                                x=self.ha_states[state]['times'],
                                y=self.ha_states[state]['values'],
                                mode='markers',
                                marker=dict(color=self.ha_state_color[state], size=10),
                                opacity=0.8,
                                name=state,
                            )
                        )

            fig.update_layout(
                title=dict(text='HomeAlone State', x=0.5, xanchor='center'),
                plot_bgcolor=plot_background_hex,
                paper_bgcolor=plot_background_hex,
                font_color=fontcolor_hex,
                title_font_color=fontcolor_hex,
                margin=dict(t=30, b=30),
                xaxis=dict(
                    range=[self.min_timestamp, self.max_timestamp],
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor=fontcolor_hex,
                    showgrid=False
                    ),
                yaxis=dict(
                    range = [-0.6, 8-0.8],
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor=fontcolor_hex,
                    zeroline=False,
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor=gridcolor_hex, 
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

            html_buffer8 = io.StringIO()
            fig.write_html(html_buffer8)
            html_buffer8.seek(0)

            print(f"Plot 8 (HA state) done in {round(time.time()-request_start,1)} seconds")
            request_start = time.time()

            # --------------------------------------
            # PLOT 9: Atomic Ally
            # --------------------------------------

            if time.time() - request_start > self.timeout_seconds:
                raise asyncio.TimeoutError('Timed out')

            fig = go.Figure()

            if aa_modes!={}:

                fig.add_trace(
                    go.Scatter(
                        x=aa_modes['all']['times'],
                        y=aa_modes['all']['values'],
                        mode='lines',
                        line=dict(color=home_alone_line, width=2),
                        opacity=0.3,
                        showlegend=False,
                        line_shape='hv'
                    )
                )

                for state in aa_modes.keys():
                    if state != 'all' and state in self.ha_state_color:
                        fig.add_trace(
                            go.Scatter(
                                x=aa_modes[state]['times'],
                                y=aa_modes[state]['values'],
                                mode='markers',
                                marker=dict(color=self.ha_state_color[state], size=10),
                                opacity=0.8,
                                name=state,
                            )
                        )

            fig.update_layout(
                title=dict(text='AtomicAlly State', x=0.5, xanchor='center'),
                plot_bgcolor=plot_background_hex,
                paper_bgcolor=plot_background_hex,
                font_color=fontcolor_hex,
                title_font_color=fontcolor_hex,
                margin=dict(t=30, b=30),
                xaxis=dict(
                    range=[self.min_timestamp, self.max_timestamp],
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor=fontcolor_hex,
                    showgrid=False
                    ),
                yaxis=dict(
                    range = [-0.6, 8-0.8],
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor=fontcolor_hex,
                    zeroline=False,
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor=gridcolor_hex, 
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

            html_buffer9 = io.StringIO()
            fig.write_html(html_buffer9)
            html_buffer9.seek(0)

            print(f"Plot 9 (AA state) done in {round(time.time()-request_start,1)} seconds")
            request_start = time.time()

            # --------------------------------------
            # PLOT 10: Weather forecasts
            # --------------------------------------

            if time.time() - request_start > self.timeout_seconds:
                raise asyncio.TimeoutError('Timed out')

            fig = go.Figure()
            color_scale = pc.diverging.RdBu[::-1]

            oat_forecasts, ws_forecasts = {}, {}
            for message in weather:
                forecast_start_time = int((message.message_persisted_ms/1000 // 3600) * 3600)
                oat_forecasts[forecast_start_time] = message.payload['OatF']
                ws_forecasts[forecast_start_time] = message.payload['WindSpeedMph']

            for idx, weather_time in enumerate(oat_forecasts):
                forecast_times = [int(weather_time) + 3600 * i for i in range(len(oat_forecasts[weather_time]))]
                forecast_times = [pendulum.from_timestamp(x, tz="America/New_York") for x in forecast_times]
                color = color_scale[int((idx / len(oat_forecasts)) * (len(color_scale) - 1))]
                opcty = 0.2
                showme = False
                if idx == len(oat_forecasts) - 1:
                    color = 'red'
                    opcty = 1
                    showme = True

                fig.add_trace(
                    go.Scatter(
                        x=forecast_times,
                        y=oat_forecasts[weather_time],
                        mode='lines',
                        line=dict(color=color, width=2),
                        opacity=opcty,
                        showlegend=showme,
                        line_shape='hv',
                        name=f"{pendulum.from_timestamp(weather_time, tz='America/New_York').hour}" 
                    )
                )

            fig.update_layout(
                title=dict(text='Weather Forecasts', x=0.5, xanchor='center'),
                plot_bgcolor=plot_background_hex,
                paper_bgcolor=plot_background_hex,
                font_color=fontcolor_hex,
                title_font_color=fontcolor_hex,
                margin=dict(t=30, b=30),
                xaxis=dict(
                    range=[self.min_timestamp, self.max_timestamp],
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor=fontcolor_hex,
                    showgrid=False
                    ),
                yaxis=dict(
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor=fontcolor_hex,
                    zeroline=False,
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor=gridcolor_hex, 
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

            html_buffer10 = io.StringIO()
            fig.write_html(html_buffer10)
            html_buffer10.seek(0)

            print(f"Plot 10 (weather) done in {round(time.time()-request_start,1)} seconds")
            request_start = time.time()

            # --------------------------------------
            # PLOT 11: Price
            # --------------------------------------

            if time.time() - request_start > self.timeout_seconds:
                raise asyncio.TimeoutError('Timed out')

            fig = go.Figure()

            request_hours = int((request.end_ms - request.start_ms)/1000 / 3600)
            price_times_s = [request.start_ms/1000 + x*3600 for x in range(request_hours+2+48)]
            price_times = [pendulum.from_timestamp(x, tz='America/New_York') for x in price_times_s]

            # Open and read the CSV file
            csv_times, csv_dist, csv_lmp = [], [], []
            with open('elec_prices.csv', newline='', encoding='utf-8') as csvfile:
                csvreader = csv.reader(csvfile)
                next(csvreader)
                for row in csvreader:
                    csv_times.append(row[0])
                    csv_dist.append(float(row[1]))
                    csv_lmp.append(float(row[2])/10)
            csv_times = [pendulum.from_format(x, 'M/D/YY H:m', tz='America/New_York').timestamp() for x in csv_times]

            price_values = [
                lmp
                for time, dist, lmp in zip(csv_times, csv_dist, csv_lmp)
                if time in price_times_s
                ]
            csv_times = [
                time for time in csv_times
                if time in price_times_s
            ]
            price_times2 = [pendulum.from_timestamp(x, tz='America/New_York') for x in csv_times]

            fig.add_trace(
                go.Scatter(
                    x=price_times2,
                    y=price_values,
                    mode='lines',
                    line=dict(color=home_alone_line),
                    opacity=0.8,
                    showlegend=False,
                    line_shape='hv',
                )
            )

            shapes_list = []
            for x in price_times2:
                if x.weekday() in [5,6]:
                    continue
                # Morning onpeak
                if x==price_times2[0] and x.hour in [8,9,10,11]:
                    shapes_list.append(
                        go.layout.Shape(
                            type='rect',
                            x0=x,
                            x1=x + timedelta(hours=5-(x.hour-7)),
                            y0=0,
                            y1=1,
                            xref="x",
                            yref="paper",
                            fillcolor="rgba(0, 100, 255, 0.1)",
                            layer="below",
                            line=dict(width=0)
                        )
                    )
                elif x.hour==7:
                    shapes_list.append(
                        go.layout.Shape(
                            type='rect',
                            x0=x,
                            x1=x + timedelta(hours=5),
                            y0=0,
                            y1=1,
                            xref="x",
                            yref="paper",
                            fillcolor="rgba(0, 100, 255, 0.1)",
                            layer="below",
                            line=dict(width=0)
                        )
                    )
                # Afternoon onpeak
                elif x==price_times2[0] and x.hour in [17,18,19]:
                    shapes_list.append(
                        go.layout.Shape(
                            type='rect',
                            x0=x,
                            x1=x + timedelta(hours=4-(x.hour-16)),
                            y0=0,
                            y1=1,
                            xref="x",
                            yref="paper",
                            fillcolor="rgba(0, 100, 255, 0.1)",
                            layer="below",
                            line=dict(width=0)
                        )
                    )
                elif x.hour==16:
                    shapes_list.append(
                        go.layout.Shape(
                            type='rect',
                            x0=x,
                            x1=x + timedelta(hours=4),
                            y0=0,
                            y1=1,
                            xref="x",
                            yref="paper",
                            fillcolor="rgba(0, 100, 255, 0.1)",
                            layer="below",
                            line=dict(width=0)
                        )
                    )
                
            fig.update_layout(yaxis=dict(title='LMP [cts/kWh]'))

            fig.update_layout(
                shapes = shapes_list,
                title=dict(text='Price Forecast', x=0.5, xanchor='center'),
                plot_bgcolor=plot_background_hex,
                paper_bgcolor=plot_background_hex,
                font_color=fontcolor_hex,
                title_font_color=fontcolor_hex,
                margin=dict(t=30, b=30),
                xaxis=dict(
                    range=[self.min_timestamp, self.max_timestamp],
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor=fontcolor_hex,
                    showgrid=False
                    ),
                yaxis=dict(
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor=fontcolor_hex,
                    zeroline=False,
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor=gridcolor_hex, 
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

            html_buffer11 = io.StringIO()
            fig.write_html(html_buffer11)
            html_buffer11.seek(0)

            print(f"Plot 11 (prices) done in {round(time.time()-request_start,1)} seconds")                

    except asyncio.TimeoutError:
        print("Request timed out!")
        return {
                "success": False, 
                "message": f"The data request timed out. Please try loading a smaller amount of data at a time.", 
                "reload": False
                }
    except Exception as e:
        return {
            "success": False, 
            "message": f"An error occurred: {str(e)}", 
            "reload": False
            }
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        zip_file.writestr('plot1.html', html_buffer1.read())
        zip_file.writestr('plot2.html', html_buffer2.read())
        zip_file.writestr('plot3.html', html_buffer3.read())
        zip_file.writestr('plot4.html', html_buffer4.read())
        zip_file.writestr('plot5.html', html_buffer5.read())
        zip_file.writestr('plot6.html', html_buffer6.read())
        zip_file.writestr('plot7.html', html_buffer7.read())
        zip_file.writestr('plot8.html', html_buffer8.read())
        zip_file.writestr('plot9.html', html_buffer9.read())
        zip_file.writestr('plot10.html', html_buffer10.read())
        zip_file.writestr('plot11.html', html_buffer11.read())
    zip_buffer.seek(0)

    return StreamingResponse(zip_buffer, 
                             media_type='application/zip', 
                             headers={"Content-Disposition": "attachment; filename=plots.zip"})
'''

if __name__ == "__main__":
    a = VisualizerApi(running_locally=True)
    a.run()
