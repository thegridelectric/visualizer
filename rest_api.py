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
import plotly.graph_objects as go
import time

channels = {}
request_global = None
MATPLOTLIB_PLOT = True
PYPLOT_PLOT = False

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
    selected_channels: List[str]
    ip_address: str
    user_agent: str
    timezone: str

class CsvRequest(BaseModel):
    selected_channels: List[str]
    timestep: int

def to_fahrenheit(t):
    return t*9/5+32

gradient = plt.get_cmap('coolwarm', 4)
buffer_colors = {
    'buffer-depth1': gradient(3),
    'buffer-depth2': gradient(2),
    'buffer-depth3': gradient(1),
    'buffer-depth4': gradient(0)
    }

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

@app.post('/csv')
async def get_csv(request: CsvRequest):

    global channels, request_global

    num_points = int((request_global.end_ms-request_global.start_ms) / (request.timestep*1000) + 1)
    
    if num_points*len(channels) > 3600/5*24*7*len(channels):
        error_message = f"This request would generate {num_points} data points, which is too much data in one go."
        error_message += "\n\nSuggestions:\n- Increase the time step\n- Reduce the number of channels"
        error_message += "\n- Change the start and end times"
        return {
                "success": False, 
                "message": error_message, 
                "reload": False
                }

    csv_times = np.linspace(request_global.start_ms, request_global.end_ms, num_points)
    csv_times_dt = [pd.to_datetime(x, unit='ms', utc=True) for x in csv_times]
    csv_times_dt = [x.tz_convert('America/New_York').replace(tzinfo=None) for x in csv_times_dt]
    csv_values = {}

    for channel in channels:
        merged = pd.merge_asof(
            pd.DataFrame({'times': csv_times_dt}),
            pd.DataFrame(channels[channel]),
            on='times',
            direction='backward'
        )
        csv_values[channel] = list(merged['values'])

    df = pd.DataFrame(csv_values)
    df['timestamps'] = csv_times_dt
    df = df[['timestamps'] + [col for col in df.columns if col != 'timestamps']]

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    response = StreamingResponse(
        iter([csv_buffer.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=output.csv"
        }
    )
    return response


@app.post('/plots')
async def get_plots(request: DataRequest):

    global channels, request_global
    request_global = request

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

    # Sort values according to time and find min/max
    min_time_ms, max_time_ms = 1e20, 0
    for key in channels.keys():
        sorted_times_values = sorted(zip(channels[key]['times'], channels[key]['values']))
        sorted_times, sorted_values = zip(*sorted_times_values)
        if list(sorted_times)[0] < min_time_ms:
            min_time_ms = list(sorted_times)[0]
        if list(sorted_times)[-1] > max_time_ms:
            max_time_ms = list(sorted_times)[-1]
        channels[key]['values'] = list(sorted_values)
        channels[key]['times'] = pd.to_datetime(list(sorted_times), unit='ms', utc=True)
        channels[key]['times'] = channels[key]['times'].tz_convert('America/New_York')
        channels[key]['times'] = [x.replace(tzinfo=None) for x in channels[key]['times']]

        # Check the length
        if len(channels[key]['times']) != len(channels[key]['values']):
            print(f"Length mismatch in channel: {key}")
            request.selected_channels.remove(key)
                
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

        # Get all timestamps in the zone states
        all_times = []
        for zone in zones:
            for state in [x for x in zones[zone] if 'state' in x]:
                all_times.extend(channels[state]['times'])
        all_times = sorted(list(set(all_times)))

        channels_copy = channels.copy()
        for zone in zones:
            for state in [x for x in zones[zone] if 'state' in x]:
                if channels[state]['times'] != all_times:
                    # Remove duplicates
                    unique_dict = {}
                    for time, value in zip(channels[state]['times'], channels[state]['values']):
                        if time not in unique_dict:
                            unique_dict[time] = value
                    channels[state]['times'] = list(unique_dict.keys())
                    channels[state]['values'] = list(unique_dict.values())
                    # Add missing times
                    for time in all_times:
                        if time not in channels[state]['times']:
                            channels[state]['times'].append(time)
                            channels[state]['values'].append(0)
                    # Sort by time again
                    sorted_times_values = sorted(zip(channels[state]['times'], channels[state]['values']))
                    sorted_times, sorted_values = zip(*sorted_times_values)
                    channels[state]['times'] = list(sorted_times)
                    channels[state]['values'] = list(sorted_values)
    
    # Start and end times on plots
    diff = max_time_ms - min_time_ms
    min_time_ms += -diff*0.05
    max_time_ms += diff*0.05
    min_time_ms_dt = pd.to_datetime(min_time_ms, unit='ms', utc=True)
    min_time_ms_dt = min_time_ms_dt.tz_convert('America/New_York').replace(tzinfo=None)
    max_time_ms_dt = pd.to_datetime(max_time_ms, unit='ms', utc=True)
    max_time_ms_dt = max_time_ms_dt.tz_convert('America/New_York').replace(tzinfo=None)

    # Get rid of selected_channels that are not in the data
    # for rc in request.selected_channels:
    #     if rc not in channels:
    #         request.selected_channels.remove(rc)

    if PYPLOT_PLOT:
        # --------------------------------------
        # PLOT 1
        # --------------------------------------

        fig = go.Figure()
        fig.update_xaxes(showgrid=False)

        # Temperature
        temp_plot = False
        if 'hp-lwt' in request.selected_channels:
            temp_plot = True
            yf = [to_fahrenheit(x/1000) for x in channels['hp-lwt']['values']]
            fig.add_trace(go.Scatter(x=channels['hp-lwt']['times'], y=yf, 
                                    mode='lines', opacity=0.7,
                                    line=dict(color='#d62728', dash='solid'),
                                    name='HP LWT'))
        if 'hp-ewt' in request.selected_channels:
            temp_plot = True
            yf = [to_fahrenheit(x/1000) for x in channels['hp-ewt']['values']]

            fig.add_trace(go.Scatter(x=channels['hp-ewt']['times'], y=yf, 
                                    mode='lines', opacity=0.7,
                                    line=dict(color='#1f77b4', dash='solid'),
                                    name='HP EWT'))

        if temp_plot:
            if 'hp-odu-pwr' in request.selected_channels or 'hp-idu-pwr' in request.selected_channels or 'primary-pump-pwr' in request.selected_channels:
                fig.update_yaxes(range=[0, 260])
            fig.update_layout(yaxis=dict(title='Temperature [F]', zeroline=False))
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            y_axis_power = 'y2'
        else:
            y_axis_power = 'y'
            fig.update_layout(yaxis=dict(title='Power [kW]', zeroline=False))
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

        # Power
        power_plot = False
        if 'hp-odu-pwr' in request.selected_channels:
            power_plot = True
            yf = [x/1000 for x in channels['hp-odu-pwr']['values']]
            fig.add_trace(go.Scatter(x=channels['hp-odu-pwr']['times'], y=yf, 
                                    mode='lines', opacity=0.7,
                                    line=dict(color='#2ca02c', dash='solid'),
                                    name='HP outdoor',
                                    yaxis=y_axis_power))

        if 'hp-idu-pwr' in request.selected_channels:
            power_plot = True
            yf = [x/1000 for x in channels['hp-idu-pwr']['values']]
            fig.add_trace(go.Scatter(x=channels['hp-idu-pwr']['times'], y=yf, 
                                    mode='lines', opacity=0.7,
                                    line=dict(color='#ff7f0e', dash='solid'),
                                    name='HP indoor',
                                    yaxis=y_axis_power))
            
        if 'primary-pump-pwr' in request.selected_channels:
            power_plot = True
            yf = [x/10 for x in channels['primary-pump-pwr']['values']]
            fig.add_trace(go.Scatter(x=channels['primary-pump-pwr']['times'], y=yf, 
                                    mode='lines', opacity=0.7,
                                    line=dict(color='purple', dash='solid'),
                                    name='Primary pump x100',
                                    yaxis=y_axis_power))

        if power_plot:
            fig.update_layout(yaxis2=dict(title='Power [kW]', overlaying='y', side='right', showgrid=False, zeroline=False, range=[0, 30]))

        fig.update_layout(
            title=dict(text='Heat pump', x=0.5, xanchor='center'),
            margin=dict(t=30, b=30),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                range=[min_time_ms_dt, max_time_ms_dt],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='rgb(42,63,96)',
                ),
            yaxis=dict(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='rgb(42,63,96)',
                ),
            yaxis2=dict(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='rgb(42,63,96)',
                ),
            legend=dict(
                x=0,
                y=1,
                xanchor='left',
                yanchor='top'
            )
        )

        fig.write_html(f"heatpump_{request.house_alias}.html")

        # --------------------------------------
        # PLOT 2
        # --------------------------------------

        fig = go.Figure()
        fig.update_xaxes(showgrid=False)

        fig.update_layout(
            title=dict(text='Distribution', x=0.5, xanchor='center'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=30, b=30),)

        # Temperature
        temp_plot = False
        if 'dist-swt' in request.selected_channels:
            temp_plot = True
            yf = [to_fahrenheit(x/1000) for x in channels['dist-swt']['values']]
            fig.add_trace(go.Scatter(x=channels['dist-swt']['times'], y=yf, 
                                    mode='lines', opacity=0.7,
                                    line=dict(color='#d62728', dash='solid'),
                                    name='Distribution SWT'))
        if 'dist-rwt' in request.selected_channels:
            temp_plot = True
            yf = [to_fahrenheit(x/1000) for x in channels['dist-rwt']['values']]
            fig.add_trace(go.Scatter(x=channels['dist-rwt']['times'], y=yf, 
                                    mode='lines', opacity=0.7,
                                    line=dict(color='#1f77b4', dash='solid'),
                                    name='Distribution RWT'))
            
        if temp_plot:
            if 'zone_heat_calls' in request.selected_channels:
                fig.update_yaxes(range=[0, 260])
            fig.update_layout(yaxis=dict(title='Temperature [F]', zeroline=False))
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            y_axis_power = 'y2'
        else:
            y_axis_power = 'y'
            fig.update_layout(yaxis=dict('Power [kW]', zeroline=False))
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

        # Distribution pump power
        power_plot = False   
        if 'dist-pump-pwr'in request.selected_channels:
            power_plot = True
            yf = [x/10 for x in channels['dist-pump-pwr']['values']]
            fig.add_trace(go.Scatter(x=channels['dist-pump-pwr']['times'], y=yf, 
                                    mode='lines', opacity=0.7,
                                    line=dict(color='pink', dash='solid'),
                                    name='Distribution pump power /10',
                                    yaxis = y_axis_power))
        if 'dist-flow' in request.selected_channels and 'dist-flow' in channels:
            power_plot = True
            yf = [x/100 for x in channels['dist-flow']['values']]
            fig.add_trace(go.Scatter(x=channels['dist-flow']['times'], y=yf, 
                                    mode='lines', opacity=0.4,
                                    line=dict(color='purple', dash='solid'),
                                    name='Distribution flow',
                                    yaxis = y_axis_power))
            
        if power_plot:
            fig.update_layout(yaxis2=dict(title='Flow [GPM] or Power [W]', overlaying='y', side='right', showgrid=False, zeroline=False, range=[0,20]))

        fig.update_layout(
            legend=dict(
                x=0,
                y=1,
                xanchor='left',
                yanchor='top'
            )
        )

        fig.update_layout(
            xaxis=dict(
                range=[min_time_ms_dt, max_time_ms_dt],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='rgb(42,63,96)',
                ),
            yaxis=dict(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='rgb(42,63,96)',
                ),
            yaxis2=dict(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='rgb(42,63,96)',
                ),
            legend=dict(
                x=0,
                y=1,
                xanchor='left',
                yanchor='top'
            )
        )

        fig.write_html(f"distribution_{request.house_alias}.html")

        # --------------------------------------
        # PLOT 3
        # --------------------------------------

        fig = go.Figure()
        fig.update_xaxes(showgrid=False)

        fig.update_layout(
            title=dict(text='Heat calls', x=0.5, xanchor='center'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=30, b=30),)

        # Zone heat calls
        num_zones = len(zones.keys())
        if 'zone_heat_calls' in request.selected_channels:

            y_labels = []

            for zone in zones:
                for key in [x for x in zones[zone] if 'state' in x]:

                    y_labels.append(key)
                    zone_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                    zone_color = zone_colors[int(key[4])-1]
            
                    for i in range(len(channels[key]['values'])):
                        if channels[key]['values'][i] == 1:
                            y_start = int(key[4]) - 1
                            y_end = int(key[4])
                            x_value = channels[key]['times'][i]

                            fig.add_trace(go.Scatter(
                                x=[x_value, x_value],
                                y=[y_start, y_end],
                                mode='lines',
                                line=dict(color=zone_color, width=2),
                                showlegend=False,
                            ))
                    
                    fig.add_trace(go.Scatter(
                        x=[None], y=[None],
                        mode='lines',
                        line=dict(color=zone_color, width=2),
                        name=key.replace('-state','')
                    ))

                    # fig.add_annotation(
                    #     x=min_time_ms_dt,
                    #     y=y_end - 0.5,
                    #     text=key,
                    #     showarrow=False,
                    #     font=dict(color=zone_color)  # Use the same color for the text
                    # )

        # fig.update_yaxes(
        #     tickvals=[i+0.5 for i in range(len(y_labels))],  # Tick positions
        #     ticktext=list(y_labels),  # Labels
        # )

        fig.update_layout(yaxis=dict(zeroline=False))
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', tickvals=list(range(num_zones+1)),)

        fig.update_layout(
            xaxis=dict(
                range=[min_time_ms_dt, max_time_ms_dt],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='rgb(42,63,96)',
                ),
            yaxis=dict(
                range = [-0.5, num_zones*2],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='rgb(42,63,96)',
                ),
            yaxis2=dict(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='rgb(42,63,96)',
                ),
            legend=dict(
                x=0,
                y=1,
                xanchor='left',
                yanchor='top',
            )
        )

        fig.write_html(f'heatcalls_{request.house_alias}.html')

        # --------------------------------------
        # PLOT 4
        # --------------------------------------

        fig = go.Figure()
        fig.update_xaxes(showgrid=False)

        fig.update_layout(
            title=dict(text='Zones', x=0.5, xanchor='center'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=30, b=30),)
        
        # zone_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        # zone_color_by_number = {}
        # for key in [x for x in zones[zone] if 'state' in x]:   
        #     zone_color_by_number[int(key[4])] = zone_colors[int(key[4])-1]
        
        for zone in zones:
            for temp in zones[zone]:
                if 'temp' in temp:
                    fig.add_trace(go.Scatter(x=channels[temp]['times'], y=channels[temp]['values'], 
                                    mode='lines', opacity=0.7,
                                    line=dict(color=zone_colors[int(temp[4])-1], dash='solid'),
                                    name=temp.replace('-temp','')))
                elif 'set' in temp:
                    fig.add_trace(go.Scatter(x=channels[temp]['times'], y=channels[temp]['values'], 
                                    mode='lines', opacity=0.7,
                                    line=dict(color=zone_colors[int(temp[4])-1], dash='dash'),
                                    showlegend=False))



        fig.update_layout(yaxis=dict(zeroline=False))
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

        fig.update_layout(
            xaxis=dict(
                range=[min_time_ms_dt, max_time_ms_dt],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='rgb(42,63,96)',
                ),
            yaxis=dict(
                range = [50, 80],
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='rgb(42,63,96)',
                ),
            legend=dict(
                x=0,
                y=1,
                xanchor='left',
                yanchor='top',
            )
        )

        fig.write_html(f'zones_{request.house_alias}.html')

        # --------------------------------------
        # PLOT 5
        # --------------------------------------




        # ax[2].set_title('Zones')
        # ax22 = ax[2].twinx()

        # colors = {}
        # for zone in zones:
        #     for temp in zones[zone]:
        #         if 'temp' in temp:
        #             color = ax[2].plot(channels[temp]['times'], channels[temp]['values'], line_style, label=temp, alpha=0.7)[0].get_color()
        #             colors[temp] = color
        #         elif 'set' in temp:
        #             base_temp = temp.replace('-set', '-temp')
        #             if base_temp in colors:
        #                 ax22.plot(channels[temp]['times'], channels[temp]['values'], '-'+line_style, label=temp, 
        #                             color=colors[base_temp], alpha=0.7)
                        
        # ax[2].set_ylabel('Temperature [F]')
        # ax22.set_yticks([])
        # lower_bound = min(ax[2].get_ylim()[0], ax22.get_ylim()[0]) - 5
        # upper_bound = max(ax[2].get_ylim()[1], ax22.get_ylim()[1]) + 15
        # ax[2].set_ylim([lower_bound, upper_bound])
        # ax22.set_ylim([lower_bound, upper_bound])
        # legend = ax[2].legend(loc='upper left', fontsize=9)
        # legend.get_frame().set_facecolor('none')
        # legend = ax22.legend(loc='upper right', fontsize=9)
        # legend.get_frame().set_facecolor('none')






    





    if MATPLOTLIB_PLOT:
        fig, ax = plt.subplots(5,1, figsize=(12,22), sharex=True)
        line_style = '-x' if 'show-points'in request.selected_channels else '-'

        # --------------------------------------
        # PLOT 1
        # --------------------------------------

        ax[0].set_title('Heat pump')

        # Temperature
        temp_plot = False
        if 'hp-lwt' in request.selected_channels:
            temp_plot = True
            channels['hp-lwt']['values'] = [to_fahrenheit(x/1000) for x in channels['hp-lwt']['values']]
            ax[0].plot(channels['hp-lwt']['times'], channels['hp-lwt']['values'], line_style, color='tab:red', alpha=0.7, label='HP LWT')
        if 'hp-ewt' in request.selected_channels:
            temp_plot = True
            channels['hp-ewt']['values'] = [to_fahrenheit(x/1000) for x in channels['hp-ewt']['values']]
            ax[0].plot(channels['hp-ewt']['times'], channels['hp-ewt']['values'], line_style, color='tab:blue', alpha=0.7, label='HP EWT')
        if temp_plot:
            if 'hp-odu-pwr' in request.selected_channels or 'hp-idu-pwr' in request.selected_channels or 'primary-pump-pwr' in request.selected_channels:
                ax[0].set_ylim([0,230])
            else:
                lower_bound = ax[0].get_ylim()[0] - 5
                upper_bound = ax[0].get_ylim()[1] + 25
                ax[0].set_ylim([lower_bound, upper_bound])
            ax[0].set_ylabel('Temperature [F]')
            legend = ax[0].legend(loc='upper left', fontsize=9)
            legend.get_frame().set_facecolor('none')
            ax20 = ax[0].twinx()
        else:
            ax20 = ax[0]

        # Power
        power_plot = False
        if 'hp-odu-pwr' in request.selected_channels:
            power_plot = True
            channels['hp-odu-pwr']['values'] = [x/1000 for x in channels['hp-odu-pwr']['values']]
            ax20.plot(channels['hp-odu-pwr']['times'], channels['hp-odu-pwr']['values'], line_style, color='tab:green', alpha=0.7, label='HP outdoor')
        if 'hp-idu-pwr' in request.selected_channels:
            power_plot = True
            channels['hp-idu-pwr']['values'] = [x/1000 for x in channels['hp-idu-pwr']['values']]
            ax20.plot(channels['hp-idu-pwr']['times'], channels['hp-idu-pwr']['values'], line_style, color='#ff7f0e', alpha=0.7, label='HP indoor')
        if 'primary-pump-pwr' in request.selected_channels:
            power_plot = True
            channels['primary-pump-pwr']['values'] = [x/10 for x in channels['primary-pump-pwr']['values']]
            ax20.plot(channels['primary-pump-pwr']['times'], channels['primary-pump-pwr']['values'], line_style, 
                    color='purple', alpha=0.7, label='Primary pump x100')
        if power_plot:
            if temp_plot:
                ax20.set_ylim([0,30])
            else:
                upper_bound = ax[0].get_ylim()[1] + 2.5
                ax[0].set_ylim([-1, upper_bound])
            ax20.set_ylabel('Power [kW]')
            legend = ax20.legend(loc='upper right', fontsize=9)
            legend.get_frame().set_facecolor('none')
        else:
            ax20.set_yticks([])

        # --------------------------------------
        # PLOT 2
        # --------------------------------------

        ax[1].set_title('Distribution')

        # Temperature
        temp_plot = False
        if 'dist-swt' in request.selected_channels:  
            temp_plot = True    
            channels['dist-swt']['values'] = [to_fahrenheit(x/1000) for x in channels['dist-swt']['values']]
            ax[1].plot(channels['dist-swt']['times'], channels['dist-swt']['values'], line_style, color='tab:red', alpha=0.7, label='Distribution SWT')
        if 'dist-rwt' in request.selected_channels:  
            temp_plot = True    
            channels['dist-rwt']['values'] = [to_fahrenheit(x/1000) for x in channels['dist-rwt']['values']]
            ax[1].plot(channels['dist-rwt']['times'], channels['dist-rwt']['values'], line_style, color='tab:blue', alpha=0.7, label='Distribution RWT')
        if temp_plot:
            ax[1].set_ylabel('Temperature [F]')
            if 'zone_heat_calls' in request.selected_channels:
                ax[1].set_ylim([0,260])
            else:
                lower_bound = ax[1].get_ylim()[0] - 5
                upper_bound = ax[1].get_ylim()[1] + 25
                ax[1].set_ylim([lower_bound, upper_bound])
            legend = ax[1].legend(loc='upper left', fontsize=9)
            legend.get_frame().set_facecolor('none')
            ax21 = ax[1].twinx()
        else:
            ax21 = ax[1]

        # Distribution pump power
        power_plot = False   
        if 'dist-pump-pwr'in request.selected_channels:
            power_plot = True
            ax21.plot(channels['dist-pump-pwr']['times'], [x/10 for x in channels['dist-pump-pwr']['values']], alpha=0.8, 
                    color='pink', label='Distribution pump power /10') 
        if 'dist-flow' in request.selected_channels and 'dist-flow'in channels:
            power_plot = True
            ax21.plot(channels['dist-flow']['times'], [x/100 for x in channels['dist-flow']['values']], alpha=0.4, 
                    color='tab:purple', label='Distribution flow') 

        # Zone heat calls
        num_zones = len(zones.keys())
        height_of_stack = 0
        stacked_values = None
        scale = 1
        if 'zone_heat_calls' in request.selected_channels:
            for zone in zones:
                for key in [x for x in zones[zone] if 'state' in x]:
                    if stacked_values is None:
                        stacked_values = np.zeros(len(channels[key]['times']))
                    if len(stacked_values) != len(channels[key]['values']):
                        height_of_stack += scale
                        stacked_values = np.ones(len(channels[key]['times'])) * height_of_stack
                    ax21.bar(channels[key]['times'], [x*scale for x in channels[key]['values']], alpha=0.7, bottom=stacked_values, 
                                label=key.replace('-state',''), width=0.003)
                    stacked_values += [x*scale for x in channels[key]['values']]   
                    # Print the value of the last 1 in the list
                    # ones_times = [
                    #     channels[key]['times'][i] 
                    #     for i in range(len(channels[key]['times']))
                    #     if channels[key]['values'][i]==1]
                    # if ones_times:
                    #     print(f"{key}: {ones_times[-1]}")

        if temp_plot and power_plot:
            if 'dist-flow' in request.selected_channels:
                upper_bound = max(channels['dist-flow']['values'])/100 * 2.5
            else:
                upper_bound = max(channels['dist-pump-pwr']['values'])/100 * 2.5
            ax21.set_ylim([0,upper_bound])
            ax21.set_ylabel('Flow rate [GPM] or Power [W]')
        elif temp_plot and not power_plot:
            upper_bound = num_zones * scale / 0.3
            ax21.set_ylim([0,upper_bound])
            ax21.set_ylabel('Heat calls')
        elif not temp_plot and power_plot:
            upper_bound = (max(channels['dist-pump-pwr']['values']) + 10)/10
            ax21.set_ylim([0,upper_bound])
            ax21.set_ylabel('Flow rate [GPM] or Power [W]')
        elif not temp_plot and not power_plot:
            upper_bound = num_zones * scale
            ax21.set_ylim([0,upper_bound])
            ax21.set_ylabel('Heat calls')
            ax21.set_yticks([])

        legend = ax21.legend(loc='upper right', fontsize=9)
        legend.get_frame().set_facecolor('none')


        # --------------------------------------
        # PLOT 3
        # --------------------------------------
            
        ax[2].set_title('Zones')
        ax22 = ax[2].twinx()

        colors = {}
        for zone in zones:
            for temp in zones[zone]:
                if 'temp' in temp:
                    color = ax[2].plot(channels[temp]['times'], channels[temp]['values'], line_style, label=temp, alpha=0.7)[0].get_color()
                    colors[temp] = color
                elif 'set' in temp:
                    base_temp = temp.replace('-set', '-temp')
                    if base_temp in colors:
                        ax22.plot(channels[temp]['times'], channels[temp]['values'], '-'+line_style, label=temp, 
                                    color=colors[base_temp], alpha=0.7)
                        
        ax[2].set_ylabel('Temperature [F]')
        ax22.set_yticks([])
        lower_bound = min(ax[2].get_ylim()[0], ax22.get_ylim()[0]) - 5
        upper_bound = max(ax[2].get_ylim()[1], ax22.get_ylim()[1]) + 15
        ax[2].set_ylim([lower_bound, upper_bound])
        ax22.set_ylim([lower_bound, upper_bound])
        legend = ax[2].legend(loc='upper left', fontsize=9)
        legend.get_frame().set_facecolor('none')
        legend = ax22.legend(loc='upper right', fontsize=9)
        legend.get_frame().set_facecolor('none')

        # --------------------------------------
        # PLOT 4
        # --------------------------------------

        ax[3].set_title('Buffer')

        buffer_channels = []
        if 'buffer-depths' in request.selected_channels:
            buffer_channels = sorted([key for key in channels.keys() if 'buffer-depth' in key and 'micro-v' not in key])
            for buffer_channel in buffer_channels:
                channels[buffer_channel]['values'] = [to_fahrenheit(x/1000) for x in channels[buffer_channel]['values']]
                ax[3].plot(channels[buffer_channel]['times'], channels[buffer_channel]['values'], line_style, 
                        color=buffer_colors[buffer_channel], alpha=0.7, label=buffer_channel)

        if not buffer_channels:
            if 'buffer-hot-pipe' in request.selected_channels:
                channels['buffer-hot-pipe']['values'] = [to_fahrenheit(x/1000) for x in channels['buffer-hot-pipe']['values']]
                ax[3].plot(channels['buffer-hot-pipe']['times'], channels['buffer-hot-pipe']['values'], line_style, 
                        color='tab:red', alpha=0.7, label='Buffer hot pipe')
            if 'buffer-cold-pipe' in request.selected_channels:
                channels['buffer-cold-pipe']['values'] = [to_fahrenheit(x/1000) for x in channels['buffer-cold-pipe']['values']]
                ax[3].plot(channels['buffer-cold-pipe']['times'], channels['buffer-cold-pipe']['values'], line_style, 
                        color='tab:blue', alpha=0.7, label='Buffer cold pipe')

        ax[3].set_ylabel('Temperature [F]')
        legend = ax[3].legend(loc='upper left', fontsize=9)
        legend.get_frame().set_facecolor('none')
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

        if 'storage-depths' in request.selected_channels:
            temp_plot = True
            tank_channels = sorted([key for key in channels.keys() if 'tank' in key and 'micro-v' not in key])
            for tank_channel in tank_channels:
                channels[tank_channel]['values'] = [to_fahrenheit(x/1000) for x in channels[tank_channel]['values']]
                ax[4].plot(channels[tank_channel]['times'], channels[tank_channel]['values'], line_style, 
                        color=storage_colors[tank_channel], alpha=0.7, label=tank_channel)

        if not tank_channels:
            if 'store-hot-pipe' in request.selected_channels:
                temp_plot = True
                channels['store-hot-pipe']['values'] = [to_fahrenheit(x/1000) for x in channels['store-hot-pipe']['values']]
                ax[4].plot(channels['store-hot-pipe']['times'], channels['store-hot-pipe']['values'], line_style, 
                        color='tab:red', alpha=0.7, label='Storage hot pipe')
            if 'store-cold-pipe' in request.selected_channels:
                temp_plot = True
                channels['store-cold-pipe']['values'] = [to_fahrenheit(x/1000) for x in channels['store-cold-pipe']['values']]
                ax[4].plot(channels['store-cold-pipe']['times'], channels['store-cold-pipe']['values'], line_style, 
                        color='tab:blue', alpha=0.7, label='Storage cold pipe')
                
        if temp_plot:
            ax24 = ax[4].twinx()
        else:
            ax24 = ax[4]

        # Power
        power_plot = False
        if 'store-pump-pwr' in request.selected_channels:
            power_plot = True
            channels['store-pump-pwr']['values'] = [x/10 for x in channels['store-pump-pwr']['values']]
            ax24.plot(channels['store-pump-pwr']['times'], channels['store-pump-pwr']['values'], line_style, 
                    color='tab:green', alpha=0.7, label='Storage pump x100')

        if power_plot:
            if temp_plot:
                ax24.set_ylim([-1,40])
            ax24.set_ylabel('Power [kW]')
            legend = ax24.legend(loc='upper right', fontsize=9)
            legend.get_frame().set_facecolor('none')
        else:
            ax24.set_yticks([])

        if temp_plot:
            if 'store-pump-pwr' in request.selected_channels:
                lower_bound = ax[4].get_ylim()[0] - 5 - max(channels['store-pump-pwr']['values'])
            else:
                lower_bound = ax[4].get_ylim()[0] - 5
            upper_bound = ax[4].get_ylim()[1] + 0.5*(ax[4].get_ylim()[1] - ax[4].get_ylim()[0])
            ax[4].set_ylim([lower_bound, upper_bound])
            ax[4].set_ylabel('Temperature [F]')
            legend = ax[4].legend(loc='upper left', fontsize=9)
            legend.get_frame().set_facecolor('none')

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
        plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=200, transparent=True)
        img_buf.seek(0)
        plt.close()

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        with open(f'heatpump_{request.house_alias}.html', 'rb') as html_file:
            zip_file.writestr('plot1.html', html_file.read())
        with open(f'distribution_{request.house_alias}.html', 'rb') as html_file:
            zip_file.writestr('plot2.html', html_file.read())
        with open(f'heatcalls_{request.house_alias}.html', 'rb') as html_file:
            zip_file.writestr('plot3.html', html_file.read())
        with open(f'zones_{request.house_alias}.html', 'rb') as html_file:
            zip_file.writestr('plot4.html', html_file.read())
        if MATPLOTLIB_PLOT:
            zip_file.writestr(f'plot.png', img_buf.getvalue())

    zip_buffer.seek(0)
    return StreamingResponse(zip_buffer, media_type='application/zip', headers={"Content-Disposition": "attachment; filename=plots.zip"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
