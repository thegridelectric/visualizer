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
from sqlalchemy import create_engine, asc, or_
from sqlalchemy.orm import sessionmaker
from fake_config import Settings
from fake_models import MessageSql
# from gjk.models import ReadingSql, DataChannelSql
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from analysis import download_excel, get_bids
import os
from fastapi.responses import FileResponse
from typing import Union
import plotly.colors as pc
import json

RUNNING_LOCALLY = True

PYPLOT_PLOT = True
MATPLOTLIB_PLOT = False
MESSAGE_SQL = True
TIMEOUT_SECONDS = 5*60
MAX_DAYS_WARNING = 3

settings = Settings(_env_file=dotenv.find_dotenv())
admin_user_password = settings.visualizer_api_password.get_secret_value()
engine = create_engine(settings.db_url.get_secret_value())
Session = sessionmaker(bind=engine)

def valid_password(house_alias, password):
    if password == admin_user_password:
        return True
    house_owner_password = getattr(settings, f"{house_alias}_owner_password").get_secret_value()
    if password == house_owner_password:
        return True
    return False

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # allow_origins=["https://thegridelectric.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Useful conversions
# ------------------------------

tank_temperatures = [
        'tank1-depth1', 'tank1-depth2', 'tank1-depth3', 'tank1-depth4', 
        'tank2-depth1', 'tank2-depth2', 'tank2-depth3', 'tank2-depth4', 
        'tank3-depth1', 'tank3-depth2', 'tank3-depth3', 'tank3-depth4'
        ]

def to_fahrenheit(t):
    return t*9/5+32

# ------------------------------
# Request types
# ------------------------------

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

# ------------------------------
# Plot colors
# ------------------------------

def to_hex(rgba):
    r, g, b, a = (int(c * 255) for c in rgba)
    return f'#{r:02x}{g:02x}{b:02x}'

gradient = plt.get_cmap('coolwarm', 4)
buffer_colors = {
    'buffer-depth1': gradient(3),
    'buffer-depth2': gradient(2),
    'buffer-depth3': gradient(1),
    'buffer-depth4': gradient(0)
    }
buffer_colors_hex = {key: to_hex(value) for key, value in buffer_colors.items()}

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
storage_colors_hex = {key: to_hex(value) for key, value in storage_colors.items()}

zone_colors_hex = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

modes_colors_hex = {
    'HpOffStoreDischarge': '#EF553B',
    'HpOffStoreOff': '#00CC96',
    'HpOnStoreOff': '#636EFA',
    'HpOnStoreCharge': '#feca52',
    'Initializing': '#a3a3a3',
    'WaitingForTemperaturesOnPeak': '#a3a3a3',
    'WaitingForTemperaturesOffPeak': '#4f4f4f',
    'Dormant': '#4f4f4f'
}
modes_order = [
    'HpOffStoreDischarge', 'HpOffStoreOff', 'HpOnStoreOff', 'HpOnStoreCharge', 'Initializing', 'Dormant']

top_modes_colors_hex = {
    'HomeAlone': '#EF553B',
    'Atn': '#00CC96',
    'Admin': '#636EFA'
}
top_modes_order = ['HomeAlone', 'Atn', 'Dormant']

aa_modes_colors_hex = {
    'HpOffStoreDischarge': '#EF553B',
    'HpOffStoreOff': '#00CC96',
    'HpOnStoreOff': '#636EFA',
    'HpOnStoreCharge': '#feca52',
    'WaitingNoElec': '#a3a3a3',
    'WaitingElec': '#4f4f4f',
    'Dormant': '#4f4f4f'
}
aa_modes_order = [
        'HpOffStoreDischarge', 'HpOffStoreOff', 'HpOnStoreOff', 'HpOnStoreCharge', 'WaitingNoElec', 'WaitingElec', 'Dormant'
        ]

# ------------------------------
# Pull data from journaldb
# ------------------------------

def get_data(request: Union[DataRequest, CsvRequest, DijkstraRequest]):

    import time
    request_start = time.time()

    if not valid_password(request.house_alias, request.password):
        with open('failed_logins.log', 'a') as log_file:
            log_entry = f"{pendulum.now()} - Failed login from {request.ip_address} with password: {request.password}\n"
            log_entry += f"Timezone '{request.timezone}', device: {request.user_agent}\n\n"
            log_file.write(log_entry)
        return {
            "success": False, 
            "message": "Wrong password.", 
            "reload":True
            }, 0, 0, 0, 0, 0, 0, 0, 0
    
    if request.house_alias == '':
        return {
            "success": False, 
            "message": "Please enter a house alias.", 
            "reload": True
            }, 0, 0, 0, 0, 0, 0, 0, 0
    
    if not request.continue_option:
        if (request.end_ms - request.start_ms)/1000/60/60/24 > MAX_DAYS_WARNING:
            warning_message = f"That's a lot of data! This could take a while, "
            warning_message += f"and eventually trigger a timeout (after {int(TIMEOUT_SECONDS/60)} minutes). "
            warning_message += f"It might be best to get this data in several smaller requests.\n\nAre you sure you would like to continue?"
            return {
                "success": False,
                "message": warning_message, 
                "reload":False,
                "continue_option": True,
                }, 0, 0, 0, 0, 0, 0, 0, 0

    if not RUNNING_LOCALLY: 
        if (request.end_ms - request.start_ms)/1000/60/60/24 > 5 and isinstance(request, DataRequest):
            return {
                "success": False,
                "message": "That's too many days to plot.", 
                "reload": False,
                }, 0, 0, 0, 0, 0, 0, 0, 0
        
        if (request.end_ms - request.start_ms)/1000/60/60/24 > 21 and isinstance(request, CsvRequest):
            return {
                "success": False,
                "message": "That's too many days of data to download.", 
                "reload": False,
                }, 0, 0, 0, 0, 0, 0, 0, 0
        
    if request.selected_channels == ['bids']:
        print("Looking for bids only")
        return {
                "success": True,
                "message": "Getting bids, not plots", 
                "reload": False,
                }, 0, 0, 0, 0, 0, 0, 0, 0
        # call a new function that gets the bids
        
    if MESSAGE_SQL:

        session = Session()

        messages = session.query(MessageSql).filter(
            MessageSql.from_alias.like(f'%{request.house_alias}%'),
            or_(
                MessageSql.message_type_name == "batched.readings",
                MessageSql.message_type_name == "report"
                ),
            MessageSql.message_persisted_ms >= request.start_ms,
            MessageSql.message_persisted_ms <=request.end_ms,
        ).order_by(asc(MessageSql.message_persisted_ms)).all()

        if not messages:
            return {
                "success": False, 
                "message": f"No data found for house '{request.house_alias}' in the selected timeframe.", 
                "reload":False
                }, 0, 0, 0, 0, 0, 0, 0, 0
        
        channels = {}
        for message in messages:
            if time.time() - request_start > TIMEOUT_SECONDS:
                return {
                    "success": False, 
                    "message": f"Timeout: getting the data took too much time.", 
                    "reload":False
                    }, 0, 0, 0, 0, 0, 0, 0, 0
            for channel in message.payload['ChannelReadingList']:
                # Find the channel name
                if message.message_type_name == 'report':
                    channel_name = channel['ChannelName']
                elif message.message_type_name == 'batched.readings':
                    for dc in message.payload['DataChannelList']:
                        if dc['Id'] == channel['ChannelId']:
                            channel_name = dc['Name']
                # Store the values and times for the channel
                if not (channel_name=='oat' and 'oak' in request.house_alias):
                    if channel_name not in channels:
                        channels[channel_name] = {
                            'values': channel['ValueList'],
                            'times': channel['ScadaReadTimeUnixMsList']
                        }
                    else:
                        channels[channel_name]['values'].extend(channel['ValueList'])
                        channels[channel_name]['times'].extend(channel['ScadaReadTimeUnixMsList'])

    # Sort values according to time and find min/max
    min_time_ms, max_time_ms = 1e20, 0
    keys_to_delete = []
    for key in channels.keys():
        # Check the length
        if (len(channels[key]['times']) != len(channels[key]['values']) 
            or not channels[key]['times']):
            print(f"Warning: channel data is empty or has length mismatch: {key}")
            keys_to_delete.append(key)
            continue
        sorted_times_values = sorted(zip(channels[key]['times'], channels[key]['values']))
        sorted_times, sorted_values = zip(*sorted_times_values)
        if list(sorted_times)[0] < min_time_ms:
            min_time_ms = list(sorted_times)[0]
        if list(sorted_times)[-1] > max_time_ms:
            max_time_ms = list(sorted_times)[-1]
        channels[key]['values'] = list(sorted_values)
        channels[key]['times'] = list(sorted_times)
        # channels[key]['times'] = pd.to_datetime(list(sorted_times), unit='ms', utc=True)
        # channels[key]['times'] = channels[key]['times'].tz_convert('America/New_York')
        # channels[key]['times'] = [x.replace(tzinfo=None) for x in channels[key]['times']]
    for key in keys_to_delete:
        del channels[key]

    # Add snapshots
    snapshots = session.query(MessageSql).filter(
        MessageSql.from_alias.like(f'%{request.house_alias}%'),
        MessageSql.message_type_name == "snapshot.spaceheat",
        MessageSql.message_persisted_ms >= max_time_ms,
        MessageSql.message_persisted_ms <= request.end_ms,
    ).order_by(asc(MessageSql.message_persisted_ms)).all()
    for snapshot in snapshots:
        for snap in snapshot.payload['LatestReadingList']:
            if snap['ChannelName'] in channels:
                channels[snap['ChannelName']]['times'].append(snap['ScadaReadTimeUnixMs'])
                channels[snap['ChannelName']]['values'].append(snap['Value'])

    # Sort values according to time and find new max
    max_time_ms = 0
    for key in channels:
        sorted_times_values = sorted(zip(channels[key]['times'], channels[key]['values']))
        sorted_times, sorted_values = zip(*sorted_times_values)
        if list(sorted_times)[-1] > max_time_ms:
            max_time_ms = list(sorted_times)[-1]
        channels[key]['values'] = list(sorted_values)
        channels[key]['times'] = pd.to_datetime(list(sorted_times), unit='ms', utc=True)
        channels[key]['times'] = channels[key]['times'].tz_convert('America/New_York')
        channels[key]['times'] = [x.replace(tzinfo=None) for x in channels[key]['times']]

    # Find all zone channels
    zones = {}
    for channel_name in channels.keys():
        if 'zone' in channel_name and 'gw-temp' not in channel_name:
            if 'state' not in channel_name:
                channels[channel_name]['values'] = [x/1000 for x in channels[channel_name]['values']]
            zone_name = channel_name.split('-')[0]
            if zone_name not in zones:
                zones[zone_name] = [channel_name]
            else:
                zones[zone_name].append(channel_name)

    # HomeAlone state
    relays = {}
    for message in messages:
        if 'StateList' in message.payload:
            for state in message.payload['StateList']:
                if state['MachineHandle'] not in relays:
                    relays[state['MachineHandle']] = {}
                    relays[state['MachineHandle']]['times'] = []
                    relays[state['MachineHandle']]['values'] = []
                relays[state['MachineHandle']]['times'].extend(state['UnixMsList'])
                relays[state['MachineHandle']]['values'].extend(state['StateList'])
    modes = {}
    if 'auto.h.n' in relays:   
        modes['all'] = {}
        modes['all']['times'] = []
        modes['all']['values'] = []
        formatted_times = [pendulum.from_timestamp(x/1000, tz='America/New_York') for x in relays['auto.h.n']['times']]
        # print(set(relays['auto.h.n']['values']))
        for state in [x for x in modes_order if x in list(set(relays['auto.h.n']['values']))]:
            modes[state] = {}
            modes[state]['times'] = []
            modes[state]['values'] = []

        final_states = []
        for time, state in zip(formatted_times, relays['auto.h.n']['values']):
            if state not in modes_order:
                final_states.append(state)
            else:
                modes['all']['times'].append(time)
                modes['all']['values'].append(4 if 'Waiting' in state else modes_order.index(state))
                modes[state]['times'].append(time)
                modes[state]['values'].append(4 if 'Waiting' in state else modes_order.index(state))
        idx = len(modes_order)+1
        final_states = list(set(final_states))
        for time, state in zip(formatted_times, relays['auto.h.n']['values']):
            if state in final_states:
                modes['all']['times'].append(time)
                modes['all']['values'].append(idx)
                modes[state]['times'].append(time)
                modes[state]['values'].append(idx)
    if not modes:
        modes = {}
        if 'auto.h' in relays:   
            modes['all'] = {}
            modes['all']['times'] = []
            modes['all']['values'] = []
            formatted_times = [pendulum.from_timestamp(x/1000, tz='America/New_York') for x in relays['auto.h']['times']]
            # print(set(relays['auto.h']['values']))
            for state in list(set(relays['auto.h']['values'])):
                modes[state] = {}
                modes[state]['times'] = []
                modes[state]['values'] = []
            for time, state in zip(formatted_times, relays['auto.h']['values']):
                position = len(modes_order) if state not in modes_order else modes_order.index(state)
                modes['all']['times'].append(time)
                modes['all']['values'].append(position)
                modes[state]['times'].append(time)
                modes[state]['values'].append(position)
            idx = len(modes_order)+1
    
    top_modes = {}
    if 'auto' in relays:         
        top_modes['all'] = {}
        top_modes['all']['times'] = []
        top_modes['all']['values'] = []
        formatted_times = [pendulum.from_timestamp(x/1000, tz='America/New_York') for x in relays['auto']['times']]
        # print(set(relays['auto']['values']))
        for state in list(set(relays['auto']['values'])):
            top_modes[state] = {}
            top_modes[state]['times'] = []
            top_modes[state]['values'] = []

        for time, state in zip(formatted_times, relays['auto']['values']):
            if state in top_modes_order:
                top_modes['all']['times'].append(time)
                top_modes['all']['values'].append(top_modes_order.index(state))
                top_modes[state]['times'].append(time)
                top_modes[state]['values'].append(top_modes_order.index(state))
    aa_modes = {}
    if 'a.aa' in relays:         
        aa_modes['all'] = {}
        aa_modes['all']['times'] = []
        aa_modes['all']['values'] = []
        formatted_times = [pendulum.from_timestamp(x/1000, tz='America/New_York') for x in relays['a.aa']['times']]
        for state in [x for x in aa_modes_order if x in list(set(relays['a.aa']['values']))]:
            aa_modes[state] = {}
            aa_modes[state]['times'] = []
            aa_modes[state]['values'] = []

        for time, state in zip(formatted_times, relays['a.aa']['values']):
            if state in aa_modes_order:
                aa_modes['all']['times'].append(time)
                aa_modes['all']['values'].append(aa_modes_order.index(state))
                aa_modes[state]['times'].append(time)
                aa_modes[state]['values'].append(aa_modes_order.index(state))
    
    if "Dormant" in top_modes:
        top_modes['Admin'] = top_modes['Dormant']
        del top_modes['Dormant']
    
    # Start and end times on plots
    min_time_ms += -(max_time_ms-min_time_ms)*0.05
    max_time_ms += (max_time_ms-min_time_ms)*0.05
    min_time_ms_dt = pd.to_datetime(min_time_ms, unit='ms', utc=True)
    max_time_ms_dt = pd.to_datetime(max_time_ms, unit='ms', utc=True)
    min_time_ms_dt = min_time_ms_dt.tz_convert('America/New_York').replace(tzinfo=None)
    max_time_ms_dt = max_time_ms_dt.tz_convert('America/New_York').replace(tzinfo=None)

    # Weather forecasts
    weather_messages = None
    if isinstance(request, DataRequest):
        try:
            weather_messages = session.query(MessageSql).filter(
                MessageSql.from_alias.like(f'%{request.house_alias}%'),
                MessageSql.message_type_name == "weather.forecast",
                MessageSql.message_persisted_ms >= request.start_ms - 24*60*60*1000,
                MessageSql.message_persisted_ms <= request.end_ms,
            ).order_by(asc(MessageSql.message_persisted_ms)).all()
        except:
            print("Could not get weather messages")

    return "", channels, zones, modes, top_modes, aa_modes, weather_messages, min_time_ms_dt, max_time_ms_dt

# ------------------------------
# Get messages for message tracker
# ------------------------------

def get_requested_messages(request: MessagesRequest, running_locally:bool=False):

    total_errors, total_warnings = 0, 0

    if not running_locally and (request.end_ms - request.start_ms)/1000/60/60/24 > 5:
        return {
            "success": False,
            "message": "That's too many days of messages to load.", 
            "reload": False,
            }
    
    session = Session()

    messages: List[MessageSql] = session.query(MessageSql).filter(
        MessageSql.from_alias.like(f'%{request.house_alias}%'),
        MessageSql.message_type_name.in_(request.selected_message_types),
        MessageSql.message_persisted_ms >= request.start_ms,
        MessageSql.message_persisted_ms <=request.end_ms,
    ).order_by(asc(MessageSql.message_persisted_ms)).all()

    if not messages:
        return {
            "success": False, 
            "message": f"No data found.", 
            "reload":False
            }
    
    levels = {
        'critical': 1,
        'error': 2,
        'warning': 3,
        'info': 4,
        'debug': 5,
        'trace': 6
    }
    
    sources = []
    pb_types = []
    summaries = []
    details = []
    times_created = []
    sorted_problem_types = sorted(
        [m for m in messages if m.message_type_name == 'gridworks.event.problem'],
        key=lambda x: (levels[x.payload['ProblemType']], x.payload['TimeCreatedMs'])
    )
    sorted_glitches = sorted(
        [m for m in messages if m.message_type_name == 'glitch'],
        key=lambda x: (levels[str(x.payload['Type']).lower()], x.payload['CreatedMs'])
    )

    for message in sorted_problem_types:
        source = message.payload['Src']
        if ".scada" in source:
            source = source.split('.scada')[0].split('.')[-1]
        sources.append(source)
        pb_types.append(message.payload['ProblemType'])
        summaries.append(message.payload['Summary'])
        details.append(message.payload['Details'].replace('<','').replace('>','').replace('\n','<br>'))
        times_created.append(str(pendulum.from_timestamp(message.payload['TimeCreatedMs']/1000, tz='America/New_York').replace(microsecond=0)))
    
    for message in sorted_glitches:
        sources.append(message.payload['FromGNodeAlias'])
        pb_types.append(str(message.payload['Type']).lower())
        summaries.append(message.payload['Summary'])
        details.append(message.payload['Details'].replace('<','').replace('>','').replace('\n','<br>'))
        times_created.append(str(pendulum.from_timestamp(message.payload['CreatedMs']/1000, tz='America/New_York').replace(microsecond=0)))

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

    table_data_columns = {
        "Log level": pb_types,
        "From node": sources,
        "Summary": summaries,
        "Details": details,
        "Time created": times_created,
        "SummaryTable": summary_table
    }
    return table_data_columns

@app.post('/messages')
async def get_messages(request: MessagesRequest):
    if not valid_password(request.house_alias, request.password):
        return {
            "success": False, 
            "message": "Wrong password.", 
            "reload":True
            }
    try:
        async with async_timeout.timeout(TIMEOUT_SECONDS):
            response = await asyncio.to_thread(get_requested_messages, request, RUNNING_LOCALLY)
            return response
    except asyncio.TimeoutError:
        print("Request timed out.")
        return {
            "success": False, 
            "message": "The data request timed out. Please try loading a smaller amount of data at a time.", 
            "reload": False
        }
    except asyncio.CancelledError:
        print("Request cancelled or client disconnected.")
        return {
            "success": False, 
            "message": "The request was cancelled because the client disconnected.", 
            "reload": False
        }
    except Exception as e:
        return {
            "success": False, 
            "message": f"An error occurred: {str(e)}", 
            "reload": False
        }

# ------------------------------
# Export as CSV
# ------------------------------

@app.post('/csv')
async def get_csv(request: CsvRequest, apirequest: Request):
    request_start = time.time()
    try:
        async with async_timeout.timeout(TIMEOUT_SECONDS):
            
            error_msg, channels, _, __, ___, ____, _____, ______, _______ = await asyncio.to_thread(get_data, request)
            print(f"Time to fetch data: {round(time.time() - request_start,2)} sec")

            if time.time() - request_start > TIMEOUT_SECONDS:
                raise asyncio.TimeoutError('Timed out')
            if await apirequest.is_disconnected():
                raise asyncio.CancelledError("Client disconnected.")

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
                if time.time() - request_start > TIMEOUT_SECONDS:
                    raise asyncio.TimeoutError('Timed out')
                if await apirequest.is_disconnected():
                    raise asyncio.CancelledError("Client disconnected.")
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
    except asyncio.CancelledError:
        print("Request cancelled or client disconnected.")
        return {
            "success": False, 
            "message": "The request was cancelled because the client disconnected.", 
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
        async with async_timeout.timeout(TIMEOUT_SECONDS):
            
            error_msg, channels, zones, modes, top_modes, aa_modes, weather, min_time_ms_dt, max_time_ms_dt = await asyncio.to_thread(get_data, request)
            print(f"Time to fetch data: {round(time.time() - request_start,2)} sec")
            if request.selected_channels == ['bids']:

                zip_bids = get_bids(request.house_alias, request.start_ms, request.end_ms)
                print("Made it here!")
                return zip_bids
    
            if error_msg != '':
                return error_msg
            
            zone_colors_hex = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']*200

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

            if PYPLOT_PLOT:

                line_style = 'lines+markers' if 'show-points'in request.selected_channels else 'lines'

                # --------------------------------------
                # PLOT 1: Heat pump
                # --------------------------------------

                if time.time() - request_start > TIMEOUT_SECONDS:
                    raise asyncio.TimeoutError('Timed out')
                if await apirequest.is_disconnected():
                    raise asyncio.CancelledError("Client disconnected.")

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
                    
                if time.time() - request_start > TIMEOUT_SECONDS:
                    raise asyncio.TimeoutError('Timed out')
                if await apirequest.is_disconnected():
                    raise asyncio.CancelledError("Client disconnected.")

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
                if time.time() - request_start > TIMEOUT_SECONDS:
                    raise asyncio.TimeoutError('Timed out')
                if await apirequest.is_disconnected():
                    raise asyncio.CancelledError("Client disconnected.")
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
                        range=[min_time_ms_dt, max_time_ms_dt],
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

                # --------------------------------------
                # PLOT 2: Distribution
                # --------------------------------------

                if time.time() - request_start > TIMEOUT_SECONDS:
                    raise asyncio.TimeoutError('Timed out')
                if await apirequest.is_disconnected():
                    raise asyncio.CancelledError("Client disconnected.")

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
                        range=[min_time_ms_dt, max_time_ms_dt],
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

                # --------------------------------------
                # PLOT 3: Heat calls
                # --------------------------------------

                if time.time() - request_start > TIMEOUT_SECONDS:
                    raise asyncio.TimeoutError('Timed out')
                if await apirequest.is_disconnected():
                    raise asyncio.CancelledError("Client disconnected.")

                fig = go.Figure()

                if 'zone-heat-calls' in request.selected_channels:
                    for zone in zones:
                        for key in [x for x in zones[zone] if 'state' in x]:
                            zone_color = zone_colors_hex[int(key[4])-1]
                            last_was_1 = False
                            # TODO: fig.add_trace here to debug fir
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
                                    # if last_was_1: 
                                    if i<len(channels[key]['values'])-1:
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
                                        # if channels[key]['values'][i+1] == 1:
                                        last_was_1 = True
                                        fig.add_shape(
                                            type='rect',
                                            x0=channels[key]['times'][i],
                                            y0=int(key[4]) - 1,
                                            x1=channels[key]['times'][i+1],
                                            y1=int(key[4]),
                                            line=dict(color=zone_color, width=0),
                                            fillcolor=zone_color,
                                            opacity=0.2,
                                            name=key.replace('-state', ''),
                                        )

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
                        range=[min_time_ms_dt, max_time_ms_dt],
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

                # --------------------------------------
                # PLOT 4: Zones
                # --------------------------------------

                if time.time() - request_start > TIMEOUT_SECONDS:
                    raise asyncio.TimeoutError('Timed out')
                if await apirequest.is_disconnected():
                    raise asyncio.CancelledError("Client disconnected.")

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
                                    line=dict(color=zone_colors_hex[int(key[4])-1], dash='solid'),
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
                                    line=dict(color=zone_colors_hex[int(key[4])-1], dash='dash'),
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
                        range=[min_time_ms_dt, max_time_ms_dt],
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

                # --------------------------------------
                # PLOT 5: Buffer
                # --------------------------------------

                if time.time() - request_start > TIMEOUT_SECONDS:
                    raise asyncio.TimeoutError('Timed out')
                if await apirequest.is_disconnected():
                    raise asyncio.CancelledError("Client disconnected.")

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
                                line=dict(color=buffer_colors_hex[buffer_channel], dash='solid')
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
                        range=[min_time_ms_dt, max_time_ms_dt],
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

                # --------------------------------------
                # PLOT 6: Storage
                # --------------------------------------

                if time.time() - request_start > TIMEOUT_SECONDS:
                    raise asyncio.TimeoutError('Timed out')
                if await apirequest.is_disconnected():
                    raise asyncio.CancelledError("Client disconnected.")

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
                            line=dict(color=storage_colors_hex[tank_channel], dash='solid'))
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
                        range=[min_time_ms_dt, max_time_ms_dt],
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

                # --------------------------------------
                # PLOT 7: Top State
                # --------------------------------------

                if time.time() - request_start > TIMEOUT_SECONDS:
                    raise asyncio.TimeoutError('Timed out')
                if await apirequest.is_disconnected():
                    raise asyncio.CancelledError("Client disconnected.")

                fig = go.Figure()

                if top_modes!={}:

                    fig.add_trace(
                        go.Scatter(
                            x=top_modes['all']['times'],
                            y=top_modes['all']['values'],
                            mode='lines',
                            line=dict(color=home_alone_line, width=2),
                            opacity=0.3,
                            showlegend=False,
                            line_shape='hv'
                        )
                    )

                    for state in top_modes.keys():
                        if state != 'all' and state in top_modes_colors_hex:
                            fig.add_trace(
                                go.Scatter(
                                    x=top_modes[state]['times'],
                                    y=top_modes[state]['values'],
                                    mode='markers',
                                    marker=dict(color=top_modes_colors_hex[state], size=10),
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
                        range=[min_time_ms_dt, max_time_ms_dt],
                        mirror=True,
                        ticks='outside',
                        showline=True,
                        linecolor=fontcolor_hex,
                        showgrid=False
                        ),
                    yaxis=dict(
                        range = [-0.6, len(top_modes)-1+0.2],
                        mirror=True,
                        ticks='outside',
                        showline=True,
                        linecolor=fontcolor_hex,
                        zeroline=False,
                        showgrid=True, 
                        gridwidth=1, 
                        gridcolor=gridcolor_hex, 
                        tickvals=list(range(len(top_modes)-1)),
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

                # --------------------------------------
                # PLOT 8: HomeAlone
                # --------------------------------------

                if time.time() - request_start > TIMEOUT_SECONDS:
                    raise asyncio.TimeoutError('Timed out')
                if await apirequest.is_disconnected():
                    raise asyncio.CancelledError("Client disconnected.")

                fig = go.Figure()

                if modes!={}:

                    fig.add_trace(
                        go.Scatter(
                            x=modes['all']['times'],
                            y=modes['all']['values'],
                            mode='lines',
                            line=dict(color=home_alone_line, width=2),
                            opacity=0.3,
                            showlegend=False,
                            line_shape='hv'
                        )
                    )

                    for state in modes.keys():
                        if state != 'all' and state in modes_colors_hex:
                            fig.add_trace(
                                go.Scatter(
                                    x=modes[state]['times'],
                                    y=modes[state]['values'],
                                    mode='markers',
                                    marker=dict(color=modes_colors_hex[state], size=10),
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
                        range=[min_time_ms_dt, max_time_ms_dt],
                        mirror=True,
                        ticks='outside',
                        showline=True,
                        linecolor=fontcolor_hex,
                        showgrid=False
                        ),
                    yaxis=dict(
                        range = [-0.6, 7-0.8],
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

                # --------------------------------------
                # PLOT 9: Atomic Ally
                # --------------------------------------

                if time.time() - request_start > TIMEOUT_SECONDS:
                    raise asyncio.TimeoutError('Timed out')
                if await apirequest.is_disconnected():
                    raise asyncio.CancelledError("Client disconnected.")

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
                        if state != 'all' and state in aa_modes_colors_hex:
                            fig.add_trace(
                                go.Scatter(
                                    x=aa_modes[state]['times'],
                                    y=aa_modes[state]['values'],
                                    mode='markers',
                                    marker=dict(color=aa_modes_colors_hex[state], size=10),
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
                        range=[min_time_ms_dt, max_time_ms_dt],
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

                # --------------------------------------
                # PLOT 10: Weather forecasts
                # --------------------------------------

                if time.time() - request_start > TIMEOUT_SECONDS:
                    raise asyncio.TimeoutError('Timed out')
                if await apirequest.is_disconnected():
                    raise asyncio.CancelledError("Client disconnected.")

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
                        range=[min_time_ms_dt, max_time_ms_dt],
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
                

    except asyncio.TimeoutError:
        print("Request timed out.")
        return {
                "success": False, 
                "message": f"The data request timed out. Please try loading a smaller amount of data at a time.", 
                "reload": False
                }
    except asyncio.CancelledError:
        print("Request cancelled or client disconnected.")
        return {
            "success": False, 
            "message": "The request was cancelled because the client disconnected.", 
            "reload": False
        }
    except Exception as e:
        return {
            "success": False, 
            "message": f"An error occurred: {str(e)}", 
            "reload": False
            }


    # if MATPLOTLIB_PLOT:
    #     fig, ax = plt.subplots(5,1, figsize=(12,22), sharex=True)
    #     line_style = '-x' if 'show-points'in request.selected_channels else '-'

    #     # --------------------------------------
    #     # PLOT 1
    #     # --------------------------------------

    #     ax[0].set_title('Heat pump')

    #     # Temperature
    #     temp_plot = False
    #     if 'hp-lwt' in request.selected_channels:
    #         temp_plot = True
    #         channels['hp-lwt']['values'] = [to_fahrenheit(x/1000) for x in channels['hp-lwt']['values']]
    #         ax[0].plot(channels['hp-lwt']['times'], channels['hp-lwt']['values'], line_style, color='tab:red', alpha=0.7, label='HP LWT')
    #     if 'hp-ewt' in request.selected_channels:
    #         temp_plot = True
    #         channels['hp-ewt']['values'] = [to_fahrenheit(x/1000) for x in channels['hp-ewt']['values']]
    #         ax[0].plot(channels['hp-ewt']['times'], channels['hp-ewt']['values'], line_style, color='tab:blue', alpha=0.7, label='HP EWT')
    #     if temp_plot:
    #         if 'hp-odu-pwr' in request.selected_channels or 'hp-idu-pwr' in request.selected_channels or 'primary-pump-pwr' in request.selected_channels:
    #             ax[0].set_ylim([0,230])
    #         else:
    #             lower_bound = ax[0].get_ylim()[0] - 5
    #             upper_bound = ax[0].get_ylim()[1] + 25
    #             ax[0].set_ylim([lower_bound, upper_bound])
    #         ax[0].set_ylabel('Temperature [F]')
    #         legend = ax[0].legend(loc='upper left', fontsize=9)
    #         legend.get_frame().set_facecolor('none')
    #         ax20 = ax[0].twinx()
    #     else:
    #         ax20 = ax[0]

    #     # Power
    #     power_plot = False
    #     if 'hp-odu-pwr' in request.selected_channels:
    #         power_plot = True
    #         channels['hp-odu-pwr']['values'] = [x/1000 for x in channels['hp-odu-pwr']['values']]
    #         ax20.plot(channels['hp-odu-pwr']['times'], channels['hp-odu-pwr']['values'], line_style, color='tab:green', alpha=0.7, label='HP outdoor')
    #     if 'hp-idu-pwr' in request.selected_channels:
    #         power_plot = True
    #         channels['hp-idu-pwr']['values'] = [x/1000 for x in channels['hp-idu-pwr']['values']]
    #         ax20.plot(channels['hp-idu-pwr']['times'], channels['hp-idu-pwr']['values'], line_style, color='#ff7f0e', alpha=0.7, label='HP indoor')
    #     if 'primary-pump-pwr' in request.selected_channels:
    #         power_plot = True
    #         channels['primary-pump-pwr']['values'] = [x/10 for x in channels['primary-pump-pwr']['values']]
    #         ax20.plot(channels['primary-pump-pwr']['times'], channels['primary-pump-pwr']['values'], line_style, 
    #                 color='purple', alpha=0.7, label='Primary pump x100')
    #     if power_plot:
    #         if temp_plot:
    #             ax20.set_ylim([0,30])
    #         else:
    #             upper_bound = ax[0].get_ylim()[1] + 2.5
    #             ax[0].set_ylim([-1, upper_bound])
    #         ax20.set_ylabel('Power [kW]')
    #         legend = ax20.legend(loc='upper right', fontsize=9)
    #         legend.get_frame().set_facecolor('none')
    #     else:
    #         ax20.set_yticks([])

    #     # --------------------------------------
    #     # PLOT 2
    #     # --------------------------------------

    #     ax[1].set_title('Distribution')

    #     # Temperature
    #     temp_plot = False
    #     if 'dist-swt' in request.selected_channels:  
    #         temp_plot = True    
    #         channels['dist-swt']['values'] = [to_fahrenheit(x/1000) for x in channels['dist-swt']['values']]
    #         ax[1].plot(channels['dist-swt']['times'], channels['dist-swt']['values'], line_style, color='tab:red', alpha=0.7, label='Distribution SWT')
    #     if 'dist-rwt' in request.selected_channels:  
    #         temp_plot = True    
    #         channels['dist-rwt']['values'] = [to_fahrenheit(x/1000) for x in channels['dist-rwt']['values']]
    #         ax[1].plot(channels['dist-rwt']['times'], channels['dist-rwt']['values'], line_style, color='tab:blue', alpha=0.7, label='Distribution RWT')
    #     if temp_plot:
    #         ax[1].set_ylabel('Temperature [F]')
    #         if 'zone-heat-calls' in request.selected_channels:
    #             ax[1].set_ylim([0,260])
    #         else:
    #             lower_bound = ax[1].get_ylim()[0] - 5
    #             upper_bound = ax[1].get_ylim()[1] + 25
    #             ax[1].set_ylim([lower_bound, upper_bound])
    #         legend = ax[1].legend(loc='upper left', fontsize=9)
    #         legend.get_frame().set_facecolor('none')
    #         ax21 = ax[1].twinx()
    #     else:
    #         ax21 = ax[1]

    #     # Distribution pump power
    #     power_plot = False   
    #     if 'dist-pump-pwr'in request.selected_channels:
    #         power_plot = True
    #         ax21.plot(channels['dist-pump-pwr']['times'], [x/10 for x in channels['dist-pump-pwr']['values']], alpha=0.8, 
    #                 color='pink', label='Distribution pump power /10') 
    #     if 'dist-flow' in request.selected_channels and 'dist-flow'in channels:
    #         power_plot = True
    #         ax21.plot(channels['dist-flow']['times'], [x/100 for x in channels['dist-flow']['values']], alpha=0.4, 
    #                 color='tab:purple', label='Distribution flow') 

    #     # Zone heat calls
    #     num_zones = len(zones.keys())
    #     height_of_stack = 0
    #     stacked_values = None
    #     scale = 1
    #     if 'zone-heat-calls' in request.selected_channels:
    #         for zone in zones:
    #             for key in [x for x in zones[zone] if 'state' in x]:
    #                 if stacked_values is None:
    #                     stacked_values = np.zeros(len(channels[key]['times']))
    #                 if len(stacked_values) != len(channels[key]['values']):
    #                     height_of_stack += scale
    #                     stacked_values = np.ones(len(channels[key]['times'])) * height_of_stack
    #                 ax21.bar(channels[key]['times'], [x*scale for x in channels[key]['values']], alpha=0.7, bottom=stacked_values, 
    #                             label=key.replace('-state',''), width=0.003)
    #                 stacked_values += [x*scale for x in channels[key]['values']]   
    #                 # Print the value of the last 1 in the list
    #                 # ones_times = [
    #                 #     channels[key]['times'][i] 
    #                 #     for i in range(len(channels[key]['times']))
    #                 #     if channels[key]['values'][i]==1]
    #                 # if ones_times:
    #                 #     print(f"{key}: {ones_times[-1]}")

    #     if temp_plot and power_plot:
    #         if 'dist-flow' in request.selected_channels:
    #             upper_bound = max(channels['dist-flow']['values'])/100 * 2.5
    #         else:
    #             upper_bound = max(channels['dist-pump-pwr']['values'])/100 * 2.5
    #         ax21.set_ylim([0,upper_bound])
    #         ax21.set_ylabel('Flow rate [GPM] or Power [W]')
    #     elif temp_plot and not power_plot:
    #         upper_bound = num_zones * scale / 0.3
    #         ax21.set_ylim([0,upper_bound])
    #         ax21.set_ylabel('Heat calls')
    #     elif not temp_plot and power_plot:
    #         upper_bound = (max(channels['dist-pump-pwr']['values']) + 10)/10
    #         ax21.set_ylim([0,upper_bound])
    #         ax21.set_ylabel('Flow rate [GPM] or Power [W]')
    #     elif not temp_plot and not power_plot:
    #         upper_bound = num_zones * scale
    #         ax21.set_ylim([0,upper_bound])
    #         ax21.set_ylabel('Heat calls')
    #         ax21.set_yticks([])

    #     legend = ax21.legend(loc='upper right', fontsize=9)
    #     legend.get_frame().set_facecolor('none')


    #     # --------------------------------------
    #     # PLOT 3
    #     # --------------------------------------
            
    #     ax[2].set_title('Zones')
    #     ax22 = ax[2].twinx()

    #     colors = {}
    #     for zone in zones:
    #         for temp in zones[zone]:
    #             if 'temp' in temp:
    #                 color = ax[2].plot(channels[temp]['times'], channels[temp]['values'], line_style, label=temp, alpha=0.7)[0].get_color()
    #                 colors[temp] = color
    #             elif 'set' in temp:
    #                 base_temp = temp.replace('-set', '-temp')
    #                 if base_temp in colors:
    #                     ax22.plot(channels[temp]['times'], channels[temp]['values'], '-'+line_style, label=temp, 
    #                                 color=colors[base_temp], alpha=0.7)
                        
    #     ax[2].set_ylabel('Temperature [F]')
    #     ax22.set_yticks([])
    #     lower_bound = min(ax[2].get_ylim()[0], ax22.get_ylim()[0]) - 5
    #     upper_bound = max(ax[2].get_ylim()[1], ax22.get_ylim()[1]) + 15
    #     ax[2].set_ylim([lower_bound, upper_bound])
    #     ax22.set_ylim([lower_bound, upper_bound])
    #     legend = ax[2].legend(loc='upper left', fontsize=9)
    #     legend.get_frame().set_facecolor('none')
    #     legend = ax22.legend(loc='upper right', fontsize=9)
    #     legend.get_frame().set_facecolor('none')

    #     # --------------------------------------
    #     # PLOT 4
    #     # --------------------------------------

    #     ax[3].set_title('Buffer')

    #     buffer_channels = []
    #     if 'buffer-depths' in request.selected_channels:
    #         buffer_channels = sorted([key for key in channels.keys() if 'buffer-depth' in key and 'micro-v' not in key])
    #         for buffer_channel in buffer_channels:
    #             channels[buffer_channel]['values'] = [to_fahrenheit(x/1000) for x in channels[buffer_channel]['values']]
    #             ax[3].plot(channels[buffer_channel]['times'], channels[buffer_channel]['values'], line_style, 
    #                     color=buffer_colors[buffer_channel], alpha=0.7, label=buffer_channel)

    #     if not buffer_channels:
    #         if 'buffer-hot-pipe' in request.selected_channels:
    #             channels['buffer-hot-pipe']['values'] = [to_fahrenheit(x/1000) for x in channels['buffer-hot-pipe']['values']]
    #             ax[3].plot(channels['buffer-hot-pipe']['times'], channels['buffer-hot-pipe']['values'], line_style, 
    #                     color='tab:red', alpha=0.7, label='Buffer hot pipe')
    #         if 'buffer-cold-pipe' in request.selected_channels:
    #             channels['buffer-cold-pipe']['values'] = [to_fahrenheit(x/1000) for x in channels['buffer-cold-pipe']['values']]
    #             ax[3].plot(channels['buffer-cold-pipe']['times'], channels['buffer-cold-pipe']['values'], line_style, 
    #                     color='tab:blue', alpha=0.7, label='Buffer cold pipe')

    #     ax[3].set_ylabel('Temperature [F]')
    #     legend = ax[3].legend(loc='upper left', fontsize=9)
    #     legend.get_frame().set_facecolor('none')
    #     lower_bound = ax[3].get_ylim()[0] - 5
    #     upper_bound = ax[3].get_ylim()[1] + 25
    #     ax[3].set_ylim([lower_bound, upper_bound])

    #     # --------------------------------------
    #     # PLOT 5
    #     # --------------------------------------

    #     ax[4].set_title('Storage')

    #     # Temperature
    #     temp_plot = False
    #     tank_channels = []

    #     if 'storage-depths' in request.selected_channels:
    #         temp_plot = True
    #         tank_channels = sorted([key for key in channels.keys() if 'tank' in key and 'micro-v' not in key])
    #         for tank_channel in tank_channels:
    #             channels[tank_channel]['values'] = [to_fahrenheit(x/1000) for x in channels[tank_channel]['values']]
    #             ax[4].plot(channels[tank_channel]['times'], channels[tank_channel]['values'], line_style, 
    #                     color=storage_colors[tank_channel], alpha=0.7, label=tank_channel)

    #     if not tank_channels:
    #         if 'store-hot-pipe' in request.selected_channels:
    #             temp_plot = True
    #             channels['store-hot-pipe']['values'] = [to_fahrenheit(x/1000) for x in channels['store-hot-pipe']['values']]
    #             ax[4].plot(channels['store-hot-pipe']['times'], channels['store-hot-pipe']['values'], line_style, 
    #                     color='tab:red', alpha=0.7, label='Storage hot pipe')
    #         if 'store-cold-pipe' in request.selected_channels:
    #             temp_plot = True
    #             channels['store-cold-pipe']['values'] = [to_fahrenheit(x/1000) for x in channels['store-cold-pipe']['values']]
    #             ax[4].plot(channels['store-cold-pipe']['times'], channels['store-cold-pipe']['values'], line_style, 
    #                     color='tab:blue', alpha=0.7, label='Storage cold pipe')
                
    #     if temp_plot:
    #         ax24 = ax[4].twinx()
    #     else:
    #         ax24 = ax[4]

    #     # Power
    #     power_plot = False
    #     if 'store-pump-pwr' in request.selected_channels:
    #         power_plot = True
    #         channels['store-pump-pwr']['values'] = [x/10 for x in channels['store-pump-pwr']['values']]
    #         ax24.plot(channels['store-pump-pwr']['times'], channels['store-pump-pwr']['values'], line_style, 
    #                 color='tab:green', alpha=0.7, label='Storage pump x100')

    #     if power_plot:
    #         if temp_plot:
    #             ax24.set_ylim([-1,40])
    #         ax24.set_ylabel('Power [kW]')
    #         legend = ax24.legend(loc='upper right', fontsize=9)
    #         legend.get_frame().set_facecolor('none')
    #     else:
    #         ax24.set_yticks([])

    #     if temp_plot:
    #         if 'store-pump-pwr' in request.selected_channels:
    #             lower_bound = ax[4].get_ylim()[0] - 5 - max(channels['store-pump-pwr']['values'])
    #         else:
    #             lower_bound = ax[4].get_ylim()[0] - 5
    #         upper_bound = ax[4].get_ylim()[1] + 0.5*(ax[4].get_ylim()[1] - ax[4].get_ylim()[0])
    #         ax[4].set_ylim([lower_bound, upper_bound])
    #         ax[4].set_ylabel('Temperature [F]')
    #         legend = ax[4].legend(loc='upper left', fontsize=9)
    #         legend.get_frame().set_facecolor('none')

    #     # --------------------------------------
    #     # All plots
    #     # --------------------------------------

    #     for axis in ax:
    #         axis.grid(axis='y', alpha=0.5)
    #         xlim = axis.get_xlim()
    #         if (mdates.num2date(xlim[1]) - mdates.num2date(xlim[0]) >= timedelta(hours=4) and 
    #             mdates.num2date(xlim[1]) - mdates.num2date(xlim[0]) <= timedelta(hours=30)):
    #             axis.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    #         elif (mdates.num2date(xlim[1]) - mdates.num2date(xlim[0]) >= timedelta(hours=31) and 
    #             mdates.num2date(xlim[1]) - mdates.num2date(xlim[0]) <= timedelta(hours=65)):
    #             axis.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    #         axis.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    #         axis.tick_params(axis='x', which='both', labelbottom=True, labelsize=8)
    #         plt.setp(axis.xaxis.get_majorticklabels(), rotation=45, ha='right')

    #     plt.tight_layout(pad=5.0)
    #     img_buf = io.BytesIO()
    #     plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=200, transparent=True)
    #     img_buf.seek(0)
    #     plt.close()

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        if PYPLOT_PLOT:
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
        # if MATPLOTLIB_PLOT:
        #     zip_file.writestr(f'plot.png', img_buf.getvalue())
    zip_buffer.seek(0)

    return StreamingResponse(zip_buffer, 
                             media_type='application/zip', 
                             headers={"Content-Disposition": "attachment; filename=plots.zip"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
