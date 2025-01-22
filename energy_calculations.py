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

ALPHA = 9.8
BETA = -ALPHA/55
GAMMA = 0.05

def required_heating_power(oat, ws):
    r = ALPHA + BETA*oat + GAMMA*ws
    return r if r>0 else 0

def to_fahrenheit(t):
    return (t * 9/5) + 32

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


def energy_used_vs_predicted(house_alias, year, month, day, onpeak_period):

    # --------------------------------
    # Get the start and end time
    # --------------------------------

    if onpeak_period == 'morning':
        hour_start = 7
        hour_end = 12
    elif onpeak_period == 'afternoon':
        hour_start = 16
        hour_end = 20
    else:
        print("onpeak_period must be 'morning' or 'afternoon'")
        return
    start_ms = pendulum.datetime(year, month, day, hour_start, tz="America/New_York").timestamp() * 1000 
    end_ms = pendulum.datetime(year, month, day, hour_end, 5, tz="America/New_York").timestamp() * 1000 

    print(f"Calculatig energy used on {year}/{month}/{day}, from {hour_start}:00 to {hour_end}:00")

    # --------------------------------
    # Get the data
    # --------------------------------

    settings = Settings(_env_file=dotenv.find_dotenv())
    engine = create_engine(settings.db_url.get_secret_value())
    Session = sessionmaker(bind=engine)
    session = Session()

    messages = session.query(MessageSql).filter(
        MessageSql.from_alias.like(f'%{house_alias}%'),
        or_(
            MessageSql.message_type_name == "batched.readings",
            MessageSql.message_type_name == "report"
            ),
        MessageSql.message_persisted_ms >= start_ms,
        MessageSql.message_persisted_ms <= end_ms,
    ).order_by(asc(MessageSql.message_persisted_ms)).all()

    channels = {}
    for message in messages:
        for channel in message.payload['ChannelReadingList']:
            # Find the channel name
            if message.message_type_name == 'report':
                channel_name = channel['ChannelName']
            elif message.message_type_name == 'batched.readings':
                for dc in message.payload['DataChannelList']:
                    if dc['Id'] == channel['ChannelId']:
                        channel_name = dc['Name']
            # Store the values and times for the channel
            if (('buffer-depth' in channel_name or ('tank' in channel_name and 'depth' in channel_name)) 
                and 'micro' not in channel_name):
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
    # Buffer energy use
    # --------------------------------

    first_values_buffer = []
    last_values_buffer = []
    for buffer_key in [x for x in channels if 'buffer' in x]:
        # Find the value closest to the start of on-peak
        differences = [abs(time - start_ms) for time in channels[buffer_key]['times'] if time < start_ms + 5*60*1000]
        closest_index = differences.index(min(differences))
        first_values_buffer.append(channels[buffer_key]['values'][closest_index])
        # Find the value closest to the end of on-peak
        differences = [abs(time - end_ms) for time in channels[buffer_key]['times']]
        closest_index = differences.index(min(differences))
        last_values_buffer.append(channels[buffer_key]['values'][closest_index])
    if len(first_values_buffer) != 4 or len(last_values_buffer) != 4:
        print("Some buffer temperatures are missing, try another day/period")
        return
    else:
        first_values_buffer = [x/1000 for x in first_values_buffer]
        last_values_buffer = [x/1000 for x in last_values_buffer]
        buffer_avg_before = sum(first_values_buffer)/4
        buffer_avg_after = sum(last_values_buffer)/4
        buffer_energy_used = 120 * 3.785 * 4.187/3600 * (buffer_avg_before - buffer_avg_after)

    # --------------------------------
    # Storage energy use
    # --------------------------------

    first_values_store = []
    last_values_store = []
    for store_key in [x for x in channels if 'tank' in x]:
        # Get the closest value to start on onpeak
        differences = [abs(time - start_ms) for time in channels[store_key]['times']]
        closest_index = differences.index(min(differences))
        first_values_store.append(channels[store_key]['values'][closest_index])
        # Get the closest value to end of onpeak
        differences = [abs(time - end_ms) for time in channels[store_key]['times']]
        closest_index = differences.index(min(differences))
        last_values_store.append(channels[store_key]['values'][closest_index])

    if len(first_values_store) != 12 or len(last_values_store) != 12:
        print("Some storage temperatures are missing, try another day/period")
        return
    else:
        first_values_store = [x/1000 for x in first_values_store]
        last_values_store = [x/1000 for x in last_values_store]
        store_avg_before = sum(first_values_store)/12
        store_avg_after = sum(last_values_store)/12
        store_energy_used = 3 * 120 * 3.785 * 4.187/3600 * (store_avg_before - store_avg_after)

    total_energy_used = store_energy_used+buffer_energy_used
    print(f"Energy used: {round(total_energy_used,1)} kWh")
    
    # --------------------------------
    # PREDICTED energy use
    # --------------------------------

    try:
        data = get_weather_data(45.6573, -68.7098, start_ms/1000, end_ms/1000)
        sorted_items = sorted(zip(data['time'], data['oat'], data['ws']), key=lambda x: datetime.strptime(x[0], '%Y-%m-%d %H:%M:%S'))
        weather_data = {'time': [], 'oat': [], 'ws': []}
        for time, oat, ws in sorted_items:
            weather_data['time'].append(time)
            weather_data['oat'].append(to_fahrenheit(oat))
            weather_data['ws'].append(to_mph(ws))
        required_heat = 0
        for i in range(5):
            required_heat += required_heating_power(weather_data['oat'][i], weather_data['ws'][i])
        print(f"Predicted energy use: {round(required_heat,2)} kWh")
    except:
        print("Could not find predicted energy use.")