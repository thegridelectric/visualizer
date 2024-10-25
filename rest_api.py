from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import dotenv
import pendulum
from sqlalchemy import create_engine, asc, or_
from sqlalchemy.orm import sessionmaker
from config import Settings
from models import MessageSql
import matplotlib.pyplot as plt
import io
import zipfile
from fastapi.responses import StreamingResponse
import pandas as pd
import matplotlib.dates as mdates
from datetime import timedelta
import numpy as np
from typing import List

settings = Settings(_env_file=dotenv.find_dotenv())
valid_password = settings.thermostat_api_key.get_secret_value()
engine = create_engine(settings.db_url.get_secret_value())
Session = sessionmaker(bind=engine)

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

class DataRequest(BaseModel):
    house_alias: str
    password: str
    start_ms: int
    end_ms: int
    selected_plot_keys: List[str]
    ip_address: str
    user_agent: str
    timezone: str

def to_fahrenheit(t):
    return t*9/5+32

buffer_colors = {
    'buffer-depth1': 'tab:red',
    'buffer-depth2': 'firebrick',
    'buffer-depth3': 'tab:purple',
    'buffer-depth4': 'tab:blue'
    }

@app.post('/plots')
async def get_plots(request: DataRequest):

    if request.password != valid_password:
        with open('failed_logins.log', 'a') as log_file:
            log_entry = f"{pendulum.now()} - Failed login from {request.ip_address} with password: {request.password}\n"
            log_entry += f"Timezone '{request.timezone}', device: {request.user_agent}\n\n"
            log_file.write(log_entry)
        return {
            "success": False, 
            "message": "Wrong password.", 
            "reload":True
            }
    
    if request.house_alias == '':
        return {
            "success": False, 
            "message": "Please enter a house alias.", 
            "reload": True
            }
    
    if (request.end_ms - request.start_ms)/1000/60/60/24 > 31:
        return {
            "success": False,
            "message": "The time difference between the start and end date exceeds the authorized limit (31 days).", 
            "reload":False
            }

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
            }
    
    selected_plot_keys = request.selected_plot_keys

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
            if channel_name not in channels:
                channels[channel_name] = {
                    'values': channel['ValueList'],
                    'times': channel['ScadaReadTimeUnixMsList']
                }
            else:
                channels[channel_name]['values'].extend(channel['ValueList'])
                channels[channel_name]['times'].extend(channel['ScadaReadTimeUnixMsList'])

    # Sort values according to time
    for key in channels.keys():
        sorted_times_values = sorted(zip(channels[key]['times'], channels[key]['values']))
        sorted_times, sorted_values = zip(*sorted_times_values)
        channels[key]['values'] = list(sorted_values)
        channels[key]['times'] = pd.to_datetime(list(sorted_times), unit='ms', utc=True)
        channels[key]['times'] = channels[key]['times'].tz_convert('America/New_York')
        channels[key]['times'] = [x.replace(tzinfo=None) for x in channels[key]['times']]

        # Check the length
        if len(channels[key]['times']) != len(channels[key]['values']):
            print(f"Length mismatch in channel: {key}")
            selected_plot_keys.remove(key)
                
    # Find all zone channels
    zones = {}
    first_times, process_heatcalls = None, False
    for channel_name in channels.keys():
        if 'zone' in channel_name and 'gw-temp' not in channel_name:
            if 'state' not in channel_name:
                channels[channel_name]['values'] = [x/1000 for x in channels[channel_name]['values']]
            else:
                # Round times to the minute
                channels[channel_name]['times'] = pd.Series(channels[channel_name]['times']).dt.round('s').tolist()
                if first_times is None:
                    first_times = channels[channel_name]['times']
                if channels[channel_name]['times'] != first_times:
                    process_heatcalls = True
            zone_name = channel_name.split('-')[0]
            if zone_name not in zones:
                zones[zone_name] = [channel_name]
            else:
                zones[zone_name].append(channel_name)

    if process_heatcalls:

        def interpolate_value(state, given_time, channels_copy):
            prev_time = None
            next_time = None
            for existing_time in channels_copy[state]['times']:
                if existing_time < given_time:
                    prev_time = channels_copy[state]['times'].index(existing_time)
                elif existing_time > given_time and next_time is None:
                    next_time = channels_copy[state]['times'].index(existing_time)
            if prev_time is None or next_time is None:
                return 0
            if channels_copy[state]['values'][prev_time]==1 and channels_copy[state]['values'][next_time]==1:
                return 1
            else:
                return 0

        # Get all timestamps in the zone states
        all_times = []
        for zone in zones:
            for state in [x for x in zones[zone] if 'state' in x]:
                all_times.extend(channels[state]['times'])
        all_times = sorted(list(set(all_times)))

        # Fill in the blanks
        channels_copy = channels.copy()
        for zone in zones:
            for state in [x for x in zones[zone] if 'state' in x]:
                if channels[state]['times'] != all_times:
                    values_to_insert = []
                    times_to_insert = []
                    # Add missing times
                    for time in all_times:
                        if time not in channels[state]['times']:
                            values_to_insert.append(interpolate_value(state, time, channels_copy))
                            times_to_insert.append(time)
                    channels[state]['times'].extend(times_to_insert)
                    channels[state]['values'].extend(values_to_insert)
                    # Sort by time again
                    sorted_times_values = sorted(zip(channels[state]['times'], channels[state]['values']))
                    sorted_times, sorted_values = zip(*sorted_times_values)
                    channels[state]['values'] = list(sorted_values)
                    channels[state]['times'] = list(sorted_times)

    # Create a BytesIO object for the zip file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:

        fig, ax = plt.subplots(5,1, figsize=(12,22), sharex=True)

        # --------------------------------------
        # PLOT 1
        # --------------------------------------

        ax[0].set_title('Heat pump')

        # Temperature
        temp_plot = False
        if 'hp-lwt' in selected_plot_keys:
            temp_plot = True
            print(f'We have {len(channels['hp-lwt']['values'])} data points for HP LWT')
            channels['hp-lwt']['values'] = [to_fahrenheit(x/1000) for x in channels['hp-lwt']['values']]
            ax[0].scatter(channels['hp-lwc']['times'], channels['hp-lwt']['values'], color='tab:red', alpha=0.7, label='HP LWT')
        if 'hp-ewt' in selected_plot_keys:
            temp_plot = True
            channels['hp-ewt']['values'] = [to_fahrenheit(x/1000) for x in channels['hp-ewt']['values']]
            ax[0].plot(channels['hp-ewt']['times'], channels['hp-ewt']['values'], color='tab:blue', alpha=0.7, label='HP EWT')
        if temp_plot:
            if 'hp-odu-pwr' in selected_plot_keys or 'hp-idu-pwr' in selected_plot_keys or 'primary-pump-pwr' in selected_plot_keys:
                ax[0].set_ylim([0,230])
            else:
                lower_bound = ax[0].get_ylim()[0] - 5
                upper_bound = ax[0].get_ylim()[1] + 25
                ax[0].set_ylim([lower_bound, upper_bound])
            ax[0].set_ylabel('Temperature [F]')
            ax[0].legend(loc='upper left', fontsize=9)
            ax20 = ax[0].twinx()
        else:
            ax20 = ax[0]

        # Power
        power_plot = False
        if 'hp-odu-pwr' in selected_plot_keys:
            power_plot = True
            channels['hp-odu-pwr']['values'] = [x/1000 for x in channels['hp-odu-pwr']['values']]
            ax20.plot(channels['hp-odu-pwr']['times'], channels['hp-odu-pwr']['values'], color='tab:green', alpha=0.7, label='HP outdoor')
        if 'hp-idu-pwr' in selected_plot_keys:
            power_plot = True
            channels['hp-idu-pwr']['values'] = [x/1000 for x in channels['hp-idu-pwr']['values']]
            ax20.plot(channels['hp-idu-pwr']['times'], channels['hp-idu-pwr']['values'], color='orange', alpha=0.7, label='HP indoor')
        if 'primary-pump-pwr' in selected_plot_keys:
            power_plot = True
            channels['primary-pump-pwr']['values'] = [x/10 for x in channels['primary-pump-pwr']['values']]
            ax20.plot(channels['primary-pump-pwr']['times'], channels['primary-pump-pwr']['values'], 
                    color='purple', alpha=0.7, label='Primary pump x100')
        if power_plot:
            if temp_plot:
                ax20.set_ylim([0,30])
            else:
                upper_bound = ax[0].get_ylim()[1] + 2.5
                ax[0].set_ylim([-1, upper_bound])
            ax20.set_ylabel('Power [kW]')
            ax20.legend(loc='upper right', fontsize=9)
        else:
            ax20.set_yticks([])

        # --------------------------------------
        # PLOT 2
        # --------------------------------------

        ax[1].set_title('Distribution')

        # Temperature
        temp_plot = False
        if 'dist-swt' in selected_plot_keys:  
            temp_plot = True    
            channels['dist-swt']['values'] = [to_fahrenheit(x/1000) for x in channels['dist-swt']['values']]
            ax[1].plot(channels['dist-swt']['times'], channels['dist-swt']['values'], color='tab:red', alpha=0.7, label='Distribution SWT')
        if 'dist-rwt' in selected_plot_keys:  
            temp_plot = True    
            channels['dist-rwt']['values'] = [to_fahrenheit(x/1000) for x in channels['dist-rwt']['values']]
            ax[1].plot(channels['dist-rwt']['times'], channels['dist-rwt']['values'], color='tab:blue', alpha=0.7, label='Distribution RWT')
        if temp_plot:
            ax[1].set_ylabel('Temperature [F]')
            if 'zone_heat_calls' in selected_plot_keys:
                ax[1].set_ylim([0,230])
            else:
                lower_bound = ax[1].get_ylim()[0] - 5
                upper_bound = ax[1].get_ylim()[1] + 25
                ax[1].set_ylim([lower_bound, upper_bound])
            ax[1].legend(loc='upper left', fontsize=9)
            ax21 = ax[1].twinx()
        else:
            ax21 = ax[1]

        # Zone heat calls
        num_zones = len(zones.keys())
        height_of_stack = 0
        stacked_values = None
        if 'zone_heat_calls' in selected_plot_keys:
            for zone in zones:
                for key in [x for x in zones[zone] if 'state' in x]:
                    if stacked_values is None:
                        stacked_values = np.zeros(len(channels[key]['times']))
                    if len(stacked_values) != len(channels[key]['values']):
                        height_of_stack += 1
                        stacked_values = np.ones(len(channels[key]['times'])) * height_of_stack
                    ax21.bar(channels[key]['times'], channels[key]['values'], alpha=0.7, bottom=stacked_values, 
                                label=key.replace('-state',''), width=0.003)
                    stacked_values += channels[key]['values']                    

            if temp_plot:
                upper_bound = num_zones / 0.3
            else:
                upper_bound = num_zones
                ax21.set_yticks(range(num_zones+1))
            ax21.set_ylim([0,upper_bound])
            ax21.set_ylabel('Heat calls')
            ax21.legend(loc='upper right', fontsize=9)
        else:
            ax21.set_yticks([])

        # --------------------------------------
        # PLOT 3
        # --------------------------------------
            
        ax[2].set_title('Zones')
        ax22 = ax[2].twinx()

        colors = {}
        for zone in zones:
            for temp in zones[zone]:
                if 'temp' in temp:
                    color = ax[2].plot(channels[temp]['times'], channels[temp]['values'], label=temp, alpha=0.7)[0].get_color()
                    colors[temp] = color
                elif 'set' in temp:
                    base_temp = temp.replace('-set', '-temp')
                    if base_temp in colors:
                        ax22.plot(channels[temp]['times'], channels[temp]['values'], label=temp, 
                                linestyle='dashed', color=colors[base_temp], alpha=0.7)
                        
        ax[2].set_ylabel('Temperature [F]')
        ax22.set_yticks([])
        lower_bound = min(ax[2].get_ylim()[0], ax22.get_ylim()[0]) - 5
        upper_bound = max(ax[2].get_ylim()[1], ax22.get_ylim()[1]) + 15
        ax[2].set_ylim([lower_bound, upper_bound])
        ax22.set_ylim([lower_bound, upper_bound])
        ax[2].legend(loc='upper left', fontsize=9)
        ax22.legend(loc='upper right', fontsize=9)

        # --------------------------------------
        # PLOT 4
        # --------------------------------------

        ax[3].set_title('Buffer')

        buffer_channels = []
        if 'buffer-depths' in selected_plot_keys:
            buffer_channels = sorted([key for key in channels.keys() if 'buffer-depth' in key and 'micro-v' not in key])
            for buffer_channel in buffer_channels:
                channels[buffer_channel]['values'] = [to_fahrenheit(x/1000) for x in channels[buffer_channel]['values']]
                ax[3].plot(channels[buffer_channel]['times'], channels[buffer_channel]['values'], 
                       color=buffer_colors[buffer_channel], alpha=0.7, label=buffer_channel)

        if not buffer_channels:
            if 'buffer-hot-pipe' in selected_plot_keys:
                channels['buffer-hot-pipe']['values'] = [to_fahrenheit(x/1000) for x in channels['buffer-hot-pipe']['values']]
                ax[3].plot(channels['buffer-hot-pipe']['times'], channels['buffer-hot-pipe']['values'], 
                        color='tab:red', alpha=0.7, label='Buffer hot pipe')
            if 'buffer-cold-pipe' in selected_plot_keys:
                channels['buffer-cold-pipe']['values'] = [to_fahrenheit(x/1000) for x in channels['buffer-cold-pipe']['values']]
                ax[3].plot(channels['buffer-cold-pipe']['times'], channels['buffer-cold-pipe']['values'], 
                        color='tab:blue', alpha=0.7, label='Buffer cold pipe')

        ax[3].set_ylabel('Temperature [F]')
        ax[3].legend(loc='upper left', fontsize=9)
        lower_bound = ax[3].get_ylim()[0] - 5
        upper_bound = ax[3].get_ylim()[1] + 25
        ax[3].set_ylim([lower_bound, upper_bound])

        # --------------------------------------
        # PLOT 5
        # --------------------------------------

        ax[4].set_title('Storage')

        # Temperature
        temp_plot = False
        tank_channels = []

        if 'storage-depths' in selected_plot_keys:
            temp_plot = True
            alpha_down = 0.7
            tank_channels = sorted([key for key in channels.keys() if 'tank' in key and 'micro-v' not in key])
            for tank_channel in tank_channels:
                channels[tank_channel]['values'] = [to_fahrenheit(x/1000) for x in channels[tank_channel]['values']]
                ax[4].plot(channels[tank_channel]['times'], channels[tank_channel]['values'], 
                       color='purple', alpha=alpha_down, label=tank_channel)
                alpha_down += -0.6/(len(tank_channels))

        if not tank_channels:
            if 'store-hot-pipe' in selected_plot_keys:
                temp_plot = True
                channels['store-hot-pipe']['values'] = [to_fahrenheit(x/1000) for x in channels['store-hot-pipe']['values']]
                ax[4].plot(channels['store-hot-pipe']['times'], channels['store-hot-pipe']['values'], 
                        color='tab:red', alpha=0.7, label='Storage hot pipe')
            if 'store-cold-pipe' in selected_plot_keys:
                temp_plot = True
                channels['store-cold-pipe']['values'] = [to_fahrenheit(x/1000) for x in channels['store-cold-pipe']['values']]
                ax[4].plot(channels['store-cold-pipe']['times'], channels['store-cold-pipe']['values'], 
                        color='tab:blue', alpha=0.7, label='Storage cold pipe')
                
        if temp_plot:
            ax24 = ax[4].twinx()
        else:
            ax24 = ax[4]

        # Power
        power_plot = False
        if 'store-pump-pwr' in selected_plot_keys:
            power_plot = True
            channels['store-pump-pwr']['values'] = [x/10 for x in channels['store-pump-pwr']['values']]
            ax24.plot(channels['store-pump-pwr']['times'], channels['store-pump-pwr']['values'], 
                    color='tab:green', alpha=0.7, label='Storage pump x100')
        
        if power_plot:
            if temp_plot:
                ax24.set_ylim([0,40])
            ax24.set_ylabel('Power [kW]')
            ax24.legend(loc='upper right', fontsize=9)
        else:
            ax24.set_yticks([])

        if temp_plot:
            if 'store-pump-pwr' in selected_plot_keys:
                lower_bound = ax[4].get_ylim()[0] - 5 - max(channels['store-pump-pwr']['values'])
            else:
                lower_bound = ax[4].get_ylim()[0] - 5
            upper_bound = ax[4].get_ylim()[1] + 0.5*(ax[4].get_ylim()[1] - ax[4].get_ylim()[0])
            ax[4].set_ylim([lower_bound, upper_bound])
            ax[4].set_ylabel('Temperature [F]')
            ax[4].legend(loc='upper left', fontsize=9)

        # --------------------------------------
        # All plots
        # --------------------------------------

        for axis in ax:
            axis.grid(axis='y', alpha=0.5)
            xlim = axis.get_xlim()
            if (mdates.num2date(xlim[1]) - mdates.num2date(xlim[0]) >= timedelta(hours=4) and 
                mdates.num2date(xlim[1]) - mdates.num2date(xlim[0]) <= timedelta(hours=30)):
                axis.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            elif (mdates.num2date(xlim[1]) - mdates.num2date(xlim[0]) >= timedelta(hours=31) and 
                mdates.num2date(xlim[1]) - mdates.num2date(xlim[0]) <= timedelta(hours=65)):
                axis.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            axis.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            axis.tick_params(axis='x', which='both', labelbottom=True, labelsize=8)
            plt.setp(axis.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout(pad=5.0)
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=200)
        img_buf.seek(0)
        zip_file.writestr(f'plot.png', img_buf.getvalue())
        plt.close()

    zip_buffer.seek(0)
    return StreamingResponse(zip_buffer, media_type='application/zip', headers={"Content-Disposition": "attachment; filename=plots.zip"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
