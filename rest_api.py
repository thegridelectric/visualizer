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

@app.post("/{house_alias}/thermostats")
async def get_latest_temperature(house_alias: str, request: DataRequest):

    if request.password != valid_password:
        raise HTTPException(status_code=403, detail="Unauthorized")

    session = Session()
    timezone = "America/New_York"
    start = pendulum.datetime(2024, 1, 1, 0, 0, tz=timezone)
    start_ms = int(start.timestamp() * 1000)

    last_message = session.query(MessageSql).filter(
        MessageSql.from_alias.like(f'%{house_alias}%'),
        MessageSql.message_persisted_ms >= start_ms
    ).order_by(desc(MessageSql.message_persisted_ms)).first()

    if not last_message:
        raise HTTPException(status_code=404, detail="No messages found.")

    temperature_data = []
    for channel in last_message.payload['ChannelReadingList']:
        if ('zone' in channel['ChannelName'] and 'gw' not in channel['ChannelName'] 
            and ('temp' in channel['ChannelName'] or 'set' in channel['ChannelName'])):
                temperature_data.append({
                    "zone": channel['ChannelName'],
                    "temperature": channel['ValueList'][-1] / 1000,
                    "time": last_message.message_persisted_ms
                })

    return temperature_data


@app.post("/{house_alias}")
async def get_latest_temperature(house_alias: str, request: DataRequest, start_ms: int, end_ms: int):

    if request.password != valid_password:
        raise HTTPException(status_code=403, detail="Unauthorized")

    session = Session()

    messages = session.query(MessageSql).filter(
        MessageSql.from_alias.like(f'%{house_alias}%'),
        MessageSql.message_persisted_ms >= start_ms,
        MessageSql.message_persisted_ms <= end_ms,
    ).order_by(desc(MessageSql.message_persisted_ms)).all()

    if not messages:
        raise HTTPException(status_code=404, detail="No messages found.")

    hp_odu_pwr = []
    hp_idu_pwr = []
    for message in messages:
        for channel in message.payload['ChannelReadingList']:
            if 'hp-odu-pwr' in channel['ChannelName']:
                hp_odu_pwr.extend(channel['ValueList'])            
            elif 'hp-idu-pwr' in channel['ChannelName']:
                hp_idu_pwr.extend(channel['ValueList'])

    hp_power_data = {
        'hp_odu_pwr': hp_odu_pwr,
        'hp_idu_pwr': hp_idu_pwr,
    }

    return hp_power_data

@app.post('/{house_alias}/plots')
async def get_plots(house_alias: str, request: DataRequest, start_ms: int, end_ms: int):

    if request.password != valid_password:
        raise HTTPException(status_code=403, detail="Unauthorized")

    session = Session()

    messages = session.query(MessageSql).filter(
        MessageSql.from_alias.like(f'%{house_alias}%'),
        MessageSql.message_persisted_ms >= start_ms,
        MessageSql.message_persisted_ms <= end_ms,
    ).order_by(desc(MessageSql.message_persisted_ms)).all()

    if not messages:
        raise HTTPException(status_code=404, detail="No messages found.")
    
    channels = {}
    for message in messages:
        for channel in message.payload['ChannelReadingList']:
            if 'zone' in channel['ChannelName']:
                continue
            if channel['ChannelName'] not in channels:
                channels[channel['ChannelName']] = {
                    'values': channel['ValueList'],
                    'times': channel['ScadaReadTimeUnixMsList']
                }
            else:
                channels[channel['ChannelName']]['values'].extend(channel['ValueList'])
                channels[channel['ChannelName']]['times'].extend(channel['ScadaReadTimeUnixMsList'])

    # Sort values according to time
    first_time = 1e9**2
    for key in channels.keys():
        sorted_times_values = sorted(zip(channels[key]['times'], channels[key]['values']))
        sorted_times, sorted_values = zip(*sorted_times_values)
        channels[key]['times'] = list(sorted_times)
        channels[key]['values'] = list(sorted_values)
        if channels[key]['times'][0] < first_time:
            first_time = channels[key]['times'][0]

    # Create a BytesIO object for the zip file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for plot_type in ['pwr', 'wt', 'pipe']:
            plt.figure(figsize=(12,4))
            for key in channels.keys():
                if plot_type not in key:
                    continue
                times = [(x - first_time) / 1000 / 60 for x in channels[key]['times']]
                if max(times) > 120:
                    times_hours = [x/60 for x in times]
                values = [x / 1000 for x in channels[key]['values']] if plot_type != 'pwr' else channels[key]['values']
                plt.plot(times, values, label=key) if max(times)<120 else plt.plot(times_hours, values, label=key)
                plt.title(f'Starting at {pendulum.from_timestamp(first_time / 1000, tz="America/New_York").format("YYYY-MM-DD HH:mm:ss")}')
            plt.xlabel('Time [min]') if max(times)<120 else plt.xlabel('Time [hours]')
            if plot_type=='pwr':
                plt.ylabel('Power [W]')
            elif plot_type=='wt':
                plt.ylabel('Temperature [C]')
            elif plot_type=='pipe':
                plt.ylabel('Temperature [C]')
            plt.legend()

            # Save the plot to a BytesIO object
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            img_buf.seek(0)  # Move to the beginning of the BytesIO object
            
            # Write the image to the zip file
            zip_file.writestr(f'{plot_type}_plot.png', img_buf.getvalue())
            plt.close()

    zip_buffer.seek(0)  # Move to the beginning of the BytesIO object
    return StreamingResponse(zip_buffer, media_type='application/zip', headers={"Content-Disposition": "attachment; filename=plots.zip"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
