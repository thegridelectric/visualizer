from flo import DGraph
from named_types import FloParamsHouse0
import dotenv
import pendulum
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker
from fake_config import Settings
from fake_models import MessageSql
import os

def download_excel(house_alias, start_ms):
    settings = Settings(_env_file=dotenv.find_dotenv())
    engine = create_engine(settings.db_url.get_secret_value())
    Session = sessionmaker(bind=engine)
    session = Session()

    flo_params_msg = session.query(MessageSql).filter(
        MessageSql.message_type_name == "flo.params.house0",
        MessageSql.from_alias.like(f'%{house_alias}%'),
        MessageSql.message_persisted_ms >= start_ms - 48*3600*1000,
        MessageSql.message_persisted_ms <= start_ms,
    ).order_by(desc(MessageSql.message_persisted_ms)).first()

    if not flo_params_msg:
        print("No FLO params")
        if os.path.exists('result.xlsx'):
            os.remove('result.xlsx')
        return

    for key, value in flo_params_msg.payload.items():
        print(f'{key}: {value}')
        if key=='AlphaTimes10': 
            flo_params_msg.payload[key] = 103
        elif key=='BetaTimes100': 
            flo_params_msg.payload[key] = -19
        elif key=='GammaEx6':
            flo_params_msg.payload[key] = 1500
        elif key=='DdPowerKw':
            flo_params_msg.payload[key] = 10.3

    flo_params = FloParamsHouse0(**flo_params_msg.payload)

    print("Running Dijkstra and saving analysis to excel...")
    g = DGraph(flo_params)
    g.solve_dijkstra()
    g.export_to_excel()
    print("Done.")

start = pendulum.datetime(2025, 1, 4, 12, 5, tz="America/New_York").timestamp()*1000
download_excel("oak", start)

# def generate_excel(params:FloParamsHouse0):
#     print("Running Dijkstra and saving analysis to excel...")
#     g = DGraph(params)
#     g.solve_dijkstra()
#     g.export_to_excel()
#     print("Done.")

# flo_parameters = FloParamsHouse0(

# )