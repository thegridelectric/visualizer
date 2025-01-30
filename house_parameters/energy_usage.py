from fake_models import MessageSql
from fake_config import Settings
from sqlalchemy import create_engine, asc, or_
from sqlalchemy.orm import sessionmaker
import dotenv
import pendulum
import requests
import datetime
import pytz
from datetime import datetime
import numpy as np
import json

ALPHA = 9.8
BETA = -ALPHA/55
GAMMA = 0.05

# --------------------------------
# Get the data
# --------------------------------

settings = Settings(_env_file=dotenv.find_dotenv())
engine = create_engine(settings.db_url.get_secret_value())
Session = sessionmaker(bind=engine)
session = Session()

# CHOOSE DATES FOR TRAINING DATA
# start_ms_train = pendulum.datetime(2024,12,3).timestamp()*1000
# end_ms_train = pendulum.datetime(2025,1,18).timestamp()*1000
# print("Finding data...")
# MESSAGES = session.query(MessageSql).filter(
#     MessageSql.message_type_name == "report",
#     MessageSql.message_persisted_ms >= start_ms_train,
#     MessageSql.message_persisted_ms <= end_ms_train,
# ).order_by(asc(MessageSql.message_persisted_ms)).all()
# print("Done!")

# # Save as json
# messages_dict = [m.to_dict() for m in MESSAGES]
# with open('messages.json', 'w') as file:
#     json.dump(messages_dict, file, indent=4)
# print("Saved messages to json")

def from_dict_msg(data):
    message = MessageSql(
            message_id=data["MessageId"],
            from_alias=data["FromAlias"],
            message_type_name=data["MessageTypeName"],
            message_persisted_ms=data["MessagePersistedMs"],
            payload=data["Payload"],
            message_created_ms=data.get("MessageCreatedMs")  # This is optional
        )
    return message

# Load the list of messages from the JSON file
print('Reading json file...')
with open('messages.json', 'r') as file:
    messages_dict = json.load(file)
messages_loaded = [from_dict_msg(message_data) for message_data in messages_dict]
print("Opened messages from json")

def required_heating_power(oat, ws):
    r = ALPHA + BETA*oat + GAMMA*ws
    return r if r>0 else 0

def to_fahrenheit(t):
    return round((t * 9/5) + 32,3)

def to_mph(ws):
    return ws * 0.621371

def get_weather_data(latitude, longitude, start_time, end_time):    
    # Get the gridpoint info
    url = f"https://api.weather.gov/points/{latitude},{longitude}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching grid data: {response.status_code}")
        return None
    # Get the nearest observation station URL
    grid_data = response.json()
    station_url = grid_data['properties']['observationStations']
    station_response = requests.get(station_url)
    if station_response.status_code != 200:
        print(f"Error fetching station data: {station_response.status_code}")
        return None
    # Get the station ID (first station in the list)
    stations = station_response.json()
    station_id = stations['features'][0]['properties']['stationIdentifier']
    # Get hourly observations from the station
    observations_url = f"https://api.weather.gov/stations/{station_id}/observations"
    params = {
        'start': datetime.utcfromtimestamp(start_time).isoformat() + "Z",
        'end': datetime.utcfromtimestamp(end_time).isoformat() + "Z"
    }
    observations_response = requests.get(observations_url, params=params)
    if observations_response.status_code != 200:
        print(f"Error fetching observations data: {observations_response.status_code}")
        return None
    # Extract the relevant data (temperature, windSpeed)
    observations = observations_response.json()
    weather_data = {
        'time': [],
        'oat': [],
        'ws': []
    }
    if not observations['features']:
        print("No past weather is available at this time. Cannot compute predicted energy use.")
        return None
    for observation in observations['features']:
        weather_data['time'].append(datetime.fromisoformat(observation['properties']['timestamp']).astimezone(pytz.timezone('America/New_York')).strftime('%Y-%m-%d %H:%M:%S'))
        weather_data['oat'].append(observation['properties']['temperature']['value'])
        weather_data['ws'].append(observation['properties']['windSpeed']['value'])
    return weather_data


def energy_used(house_alias, year, month, day, hour=None, onpeak_period=None):

    selected_messages = messages_loaded.copy()

    # --------------------------------
    # Get the start and end time
    # --------------------------------

    if hour is not None:
        hour_start = hour
        hour_end = hour+1
        if hour_start not in [7,8,9,10,11,16,17,18,19]:
            print("Hour start should be in onpeak hours")
            return np.nan
    elif onpeak_period == 'morning':
        hour_start = 7
        hour_end = 12
    elif onpeak_period == 'afternoon':
        hour_start = 16
        hour_end = 20
    else:
        print("onpeak_period must be 'morning' or 'afternoon'")
        return np.nan
    start_ms = pendulum.datetime(year, month, day, hour_start, tz="America/New_York").timestamp() * 1000 
    end_ms = pendulum.datetime(year, month, day, hour_end, 5, tz="America/New_York").timestamp() * 1000 

    print(f"\nCalculatig energy used on {year}/{month}/{day}, from {hour_start}:00 to {hour_end}:00")

    channels = {}
    for message in [x for x in selected_messages 
                    if f'{house_alias}' in x.from_alias
                    and x.message_persisted_ms >= start_ms-5*60*1000
                    and x.message_persisted_ms <= end_ms+5*60*1000]:
        for channel in message.payload['ChannelReadingList']:
            # Find the channel name
            if message.message_type_name == 'report':
                channel_name = channel['ChannelName']
            elif message.message_type_name == 'batched.readings':
                for dc in message.payload['DataChannelList']:
                    if dc['Id'] == channel['ChannelId']:
                        channel_name = dc['Name']
            # Store the values and times for the channel
            if (('buffer-depth' in channel_name or ('tank' in channel_name and 'depth' in channel_name)) and 'micro' not in channel_name):
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

    # --------------------------------
    # Remove cases where the zone is not at setpoint
    # --------------------------------

    all_channels = {}
    for message in [x for x in selected_messages 
                    if f'{house_alias}' in x.from_alias
                    and x.message_persisted_ms >= start_ms-5*60*1000
                    and x.message_persisted_ms <= end_ms+5*60*1000]:
        for channel in message.payload['ChannelReadingList']:
            # Find the channel name
            if message.message_type_name == 'report':
                channel_name = channel['ChannelName']
            elif message.message_type_name == 'batched.readings':
                for dc in message.payload['DataChannelList']:
                    if dc['Id'] == channel['ChannelId']:
                        channel_name = dc['Name']
            # Store the values and times for the channel
            if channel_name not in all_channels:
                all_channels[channel_name] = {
                    'values': channel['ValueList'],
                    'times': channel['ScadaReadTimeUnixMsList']
                }
            else:
                all_channels[channel_name]['values'].extend(channel['ValueList'])
                all_channels[channel_name]['times'].extend(channel['ScadaReadTimeUnixMsList'])

    for key in all_channels.keys():
        sorted_times_values = sorted(zip(all_channels[key]['times'], all_channels[key]['values']))
        sorted_times, sorted_values = zip(*sorted_times_values)
        all_channels[key]['values'] = list(sorted_values)
        all_channels[key]['times'] = list(sorted_times)

    zones = {}
    for channel_name in all_channels.keys():
        if 'zone' in channel_name and 'gw-temp' not in channel_name:
            if 'state' not in channel_name:
                all_channels[channel_name]['values'] = [x/1000 for x in all_channels[channel_name]['values']]
            zone_name = channel_name.split('-')[0]
            if zone_name not in zones:
                zones[zone_name] = [channel_name]
            else:
                zones[zone_name].append(channel_name)

    temp_start, temp_end = {}, {}
    set_start, set_end = {}, {}
    for zone in zones:
        for key in [x for x in zones[zone] if 'temp' in x]:
            chn = all_channels[key]
            # Temperature at the start of the hour
            differences = [abs(time - start_ms) for time in chn['times'] if time < start_ms + 5*60*1000]
            if not differences:
                print("No data found")
                return np.nan
            closest_index = differences.index(min(differences))
            temp_start[zone] = chn['values'][closest_index]
            # Temperature at the end of the hour
            chn = all_channels[key]
            differences = [abs(time - end_ms) for time in chn['times'] if time < start_ms + 5*60*1000]
            if not differences:
                print("No data found")
                return np.nan
            closest_index = differences.index(min(differences))
            temp_end[zone] = chn['values'][closest_index]

        for key in [x for x in zones[zone] if 'set' in x]:
            chn = all_channels[key]
            # Temperature at the start of the hour
            differences = [abs(time - start_ms) for time in chn['times'] if time < start_ms + 5*60*1000]
            if not differences:
                print("No data found")
                return np.nan
            closest_index = differences.index(min(differences))
            set_start[zone] = chn['values'][closest_index]
            # Temperature at the end of the hour
            chn = all_channels[key]
            differences = [abs(time - end_ms) for time in chn['times'] if time < start_ms + 5*60*1000]
            if not differences:
                print("No data found")
                return np.nan
            closest_index = differences.index(min(differences))
            set_end[zone] = chn['values'][closest_index]

    threshold = 2
    for zone in zones:
        if zone not in set_start or zone not in set_end or zone not in temp_start or zone not in temp_end:
            print("ERROR")
        else:
            if ((np.abs(set_start[zone]-temp_start[zone])>threshold) or np.abs(set_end[zone]-temp_end[zone])>threshold):
                print("Setpoint is not the same as temperature either at start or end of hour")
                print(f"Set: {set_start[zone]}, Temp: {temp_start[zone]}")
                return np.nan

    # --------------------------------
    # Buffer energy use
    # --------------------------------

    first_values_buffer, first_times_buffer = [], []
    last_values_buffer, last_times_buffer = [], []
    for buffer_key in [x for x in channels if 'buffer' in x]:
        # Find the value closest to the start of on-peak
        differences = [abs(time - start_ms) for time in channels[buffer_key]['times']]
        if not differences:
            print("No data found")
            return np.nan
        closest_index = differences.index(min(differences))
        first_values_buffer.append(channels[buffer_key]['values'][closest_index])
        first_times_buffer.append(channels[buffer_key]['times'][closest_index])
        # Find the value closest to the end of on-peak
        differences = [abs(time - end_ms) for time in channels[buffer_key]['times']]
        if not differences:
            print("No data found")
            return np.nan
        closest_index = differences.index(min(differences))
        last_values_buffer.append(channels[buffer_key]['values'][closest_index])
        last_times_buffer.append(channels[buffer_key]['times'][closest_index])
    if last_times_buffer and first_times_buffer:
        if last_times_buffer[-1] - first_times_buffer[-1] < 50*60*1000:
            print("Not enough time between first and last value:")
            print(f"-First: {pendulum.from_timestamp(first_times_buffer[-1]/1000, tz='America/New_York')}")
            print(f"-Last: {pendulum.from_timestamp(last_times_buffer[-1]/1000, tz='America/New_York')}")
            return np.nan
    if len(first_values_buffer) != 4 or len(last_values_buffer) != 4:
        print("Some buffer temperatures are missing, try another day/period")
        return np.nan
    else:
        first_values_buffer = [x/1000 for x in first_values_buffer]
        last_values_buffer = [x/1000 for x in last_values_buffer]
        buffer_avg_before = sum(first_values_buffer)/4
        buffer_avg_after = sum(last_values_buffer)/4
        buffer_energy_used = 120 * 3.785 * 4.187/3600 * (buffer_avg_before - buffer_avg_after)
        # print(f"Buffer before: {[to_fahrenheit(x) for x in first_values_buffer]}")
        # print(f"Buffer after: {[to_fahrenheit(x) for x in last_values_buffer]}")
        # print(f"Buffer used: {buffer_energy_used}")

    # --------------------------------
    # Storage energy use
    # --------------------------------

    first_values_store, first_times_store = [], []
    last_values_store, last_times_store = [], []
    for store_key in [x for x in channels if 'tank' in x]:
        # Get the closest value to start on onpeak
        differences = [abs(time - start_ms) for time in channels[store_key]['times']]
        if not differences:
            print("No data found")
            return np.nan
        closest_index = differences.index(min(differences))
        first_values_store.append(channels[store_key]['values'][closest_index])
        first_times_store.append(channels[store_key]['times'][closest_index])
        # Get the closest value to end of onpeak
        differences = [abs(time - end_ms) for time in channels[store_key]['times']]
        if not differences:
            print("No data found")
            return np.nan
        closest_index = differences.index(min(differences))
        last_values_store.append(channels[store_key]['values'][closest_index])
        last_times_store.append(channels[store_key]['times'][closest_index])

    if last_times_store and first_times_store:
        if last_times_store[-1] - first_times_store[-1] < 30*60*1000:
            print("Not enough time between first and last value (store):")
            print(f"-First: {pendulum.from_timestamp(first_times_store[-1]/1000, tz='America/New_York')}")
            print(f"-Last: {pendulum.from_timestamp(last_times_store[-1]/1000, tz='America/New_York')}")
            return np.nan
    if len(first_values_store) != 12 or len(last_values_store) != 12:
        print("Some storage temperatures are missing, try another day/period")
        return np.nan
    else:
        first_values_store = [x/1000 for x in first_values_store]
        last_values_store = [x/1000 for x in last_values_store]
        store_avg_before = sum(first_values_store)/12
        store_avg_after = sum(last_values_store)/12
        store_energy_used = 3 * 120 * 3.785 * 4.187/3600 * (store_avg_before - store_avg_after)
        # print(f"Store before: {[to_fahrenheit(x) for x in first_values_store]}")
        # print(f"Store after: {[to_fahrenheit(x) for x in last_values_store]}")
        # print(f"Store used: {store_energy_used}")

    total_energy_used = store_energy_used+buffer_energy_used
    if hour is None:
        print(f"Energy used: {round(total_energy_used,1)} kWh")
    return round(total_energy_used,2)
    
    # # --------------------------------
    # # PREDICTED energy use
    # # --------------------------------

    # try:
    #     data = get_weather_data(45.6573, -68.7098, start_ms/1000, end_ms/1000)
    #     sorted_items = sorted(zip(data['time'], data['oat'], data['ws']), key=lambda x: datetime.strptime(x[0], '%Y-%m-%d %H:%M:%S'))
    #     weather_data = {'time': [], 'oat': [], 'ws': []}
    #     for time, oat, ws in sorted_items:
    #         weather_data['time'].append(time)
    #         weather_data['oat'].append(to_fahrenheit(oat))
    #         weather_data['ws'].append(to_mph(ws))
    #     required_heat = 0
    #     for i in range(5):
    #         required_heat += required_heating_power(weather_data['oat'][i], weather_data['ws'][i])
    #     print(f"Predicted energy use: {round(required_heat,2)} kWh")
    # except:
    #     print("Could not find predicted energy use.")