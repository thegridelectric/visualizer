from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import dotenv
import pendulum
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker
from config import Settings
from models import MessageSql

settings = Settings(_env_file=dotenv.find_dotenv())
valid_password = settings.thermostat_api_key.get_secret_value()
engine = create_engine(settings.db_url.get_secret_value())
Session = sessionmaker(bind=engine)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ThermostatRequest(BaseModel):
    password: str

@app.post("/thermostats/{house_alias}")
async def get_latest_temperature(house_alias: str, request: ThermostatRequest):

    if request.password != valid_password:
        raise HTTPException(status_code=403, detail="Unauthorized")

    session = Session()
    timezone = "America/New_York"
    start = pendulum.datetime(2022, 1, 1, 0, 0, tz=timezone)
    start_ms = int(start.timestamp() * 1000)

    last_message = session.query(MessageSql).filter(
        MessageSql.from_alias.like(f'%{house_alias}%'),
        MessageSql.message_persisted_ms >= start_ms
    ).order_by(desc(MessageSql.message_persisted_ms)).first()

    if not last_message:
        raise HTTPException(status_code=404, detail="No messages found.")

    temperature_data = []
    for elem in last_message.payload['DataChannelList']:
        if 'zone' in elem['Name'] and 'gw' not in elem['Name'] and ('temp' in elem['Name'] or 'set' in elem['Name']):
            zone_name = elem['Name']
            for reading in last_message.payload['ChannelReadingList']:
                if reading['ChannelId'] == elem['Id']:
                    temperature_data.append({
                        "zone": zone_name,
                        "temperature": reading['ValueList'][0] / 1000
                    })

    return temperature_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
