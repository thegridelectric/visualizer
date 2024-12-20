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
        return

    flo_params = FloParamsHouse0(**flo_params_msg.payload)
    # for key, value in flo_params_msg.payload.items():
    #     print(f'{key}: {value}')

    os.remove('result.xlsx')

    print("Running Dijkstra and saving analysis to excel...")
    g = DGraph(flo_params)
    g.solve_dijkstra()
    g.export_to_excel()
    print("Done.")