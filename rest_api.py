from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import dotenv
import pendulum
from sqlalchemy import create_engine, desc
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
    password: str
    start_ms: int
    end_ms: int
    selected_plot_keys: List[str]

def to_fahrenheit(t):
    return t*9/5+32

# @app.post("/{house_alias}/thermostats")
# async def get_latest_temperature(house_alias: str, request: DataRequest):

#     if request.password != valid_password:
#         raise HTTPException(status_code=403, detail="Unauthorized")

#     session = Session()
#     timezone = "America/New_York"
#     start = pendulum.datetime(2024, 1, 1, 0, 0, tz=timezone)
#     start_ms = int(start.timestamp() * 1000)

#     last_message = session.query(MessageSql).filter(
#         MessageSql.from_alias.like(f'%{house_alias}%'),
#         MessageSql.message_persisted_ms >= start_ms
#     ).order_by(desc(MessageSql.message_persisted_ms)).first()

#     if not last_message:
#         raise HTTPException(status_code=404, detail="No messages found.")

#     temperature_data = []
#     for channel in last_message.payload['ChannelReadingList']:
#         if ('zone' in channel['ChannelName'] and 'gw' not in channel['ChannelName'] 
#             and ('temp' in channel['ChannelName'] or 'set' in channel['ChannelName'])):
#                 temperature_data.append({
#                     "zone": channel['ChannelName'],
#                     "temperature": channel['ValueList'][-1] / 1000,
#                     "time": last_message.message_persisted_ms
#                 })

#     return temperature_data


@app.post('/{house_alias}/plots')
async def get_plots(house_alias: str, request: DataRequest):

    if request.password != valid_password:
        return {"success": False, "message": "Invalid password. This page will reload.", "reload":True}

    session = Session()

    messages = session.query(MessageSql).filter(
        MessageSql.from_alias.like(f'%{house_alias}%'),
        MessageSql.message_persisted_ms >= request.start_ms,
        MessageSql.message_persisted_ms <=request.end_ms,
    ).order_by(desc(MessageSql.message_persisted_ms)).all()

    if not messages:
        return {"success": False, "message": f"No data found for house '{house_alias}' in the selected timeframe.", "reload":False}
    
    selected_plot_keys = request.selected_plot_keys

    channels = {}
    for message in messages:
        for channel in message.payload['ChannelReadingList']:
            if channel['ChannelName'] not in channels:
                channels[channel['ChannelName']] = {
                    'values': channel['ValueList'],
                    'times': channel['ScadaReadTimeUnixMsList']
                }
            else:
                channels[channel['ChannelName']]['values'].extend(channel['ValueList'])
                channels[channel['ChannelName']]['times'].extend(channel['ScadaReadTimeUnixMsList'])

    # Sort values according to time
    for key in channels.keys():
        sorted_times_values = sorted(zip(channels[key]['times'], channels[key]['values']))
        sorted_times, sorted_values = zip(*sorted_times_values)
        channels[key]['values'] = list(sorted_values)
        channels[key]['times'] = pd.to_datetime(list(sorted_times), unit='ms', utc=True)
        channels[key]['times'] = channels[key]['times'].tz_convert('America/New_York')
        channels[key]['times'] = [x.replace(tzinfo=None) for x in channels[key]['times']]
                
    # Find all zone channels
    zones = {}
    for key in channels.keys():
        if 'zone' in key:
            if 'state' not in key:
                channels[key]['values'] = [x/1000 for x in channels[key]['values']]
            if key.split('-')[0] not in zones:
                zones[key.split('-')[0]] = [key]
            else:
                zones[key.split('-')[0]].append(key)

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
            channels['hp-lwt']['values'] = [to_fahrenheit(x/1000) for x in channels['hp-lwt']['values']]
            ax[0].plot(channels['hp-lwt']['times'], channels['hp-lwt']['values'], color='tab:red', alpha=0.7, label='HP LWT')
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
        num_zones = 0
        stacked_values = None
        if 'zone_heat_calls' in selected_plot_keys:
            for key in channels.keys():
                if 'zone' in key and 'state' in key:
                    num_zones += 1
                    if stacked_values is None:
                        stacked_values = np.zeros(len(channels[key]['times']))
                    ax21.bar(channels[key]['times'], channels[key]['values'], alpha=0.7, bottom=stacked_values, 
                            label=key.replace('-state',''), width=0.004)
                    stacked_values += channels[key]['values']

            upper_bound = num_zones / 0.3 if temp_plot else num_zones
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
        ax22.set_ylabel('Setpoint [F]')
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
            if 'store-pump-pwr' in selected_plot_keys:
                ax[4].set_ylim([40,230])
            else:
                lower_bound = ax[4].get_ylim()[0] - 5
                upper_bound = ax[4].get_ylim()[1] + 25
                ax[4].set_ylim([lower_bound, upper_bound])
            ax[4].set_ylabel('Temperature [F]')
            ax[4].legend(loc='upper left', fontsize=9)
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
