from flo import DGraph
from hinge import FloHinge
from named_types import FloParamsHouse0
import dotenv
import pendulum
from sqlalchemy import create_engine, desc, asc, or_
from sqlalchemy.orm import sessionmaker
from fake_config import Settings
from fake_models import MessageSql
import matplotlib.pyplot as plt
from typing import List
import numpy as np
import pandas as pd
from datetime import timedelta

PRINT = False


class QuickFloEval():

    def __init__(self, house_alias, start_ms, timezone):
        settings = Settings(_env_file=dotenv.find_dotenv())
        engine = create_engine(settings.db_url.get_secret_value())
        Session = sessionmaker(bind=engine)
        self.session = Session()
        self.house_alias = house_alias
        self.flo_start_ms = start_ms
        self.timezone_str = timezone
        self.selected_messages = None
        self.get_flo_params()

    def unix_ms_to_date(self, time_ms):
        return pendulum.from_timestamp(time_ms/1000, tz=self.timezone_str)

    def get_flo_params(self):
        flo_params_messages: List[MessageSql] = self.session.query(MessageSql).filter(
            MessageSql.message_type_name == "flo.params.house0",
            MessageSql.from_alias.like(f'%{self.house_alias}%'),
            MessageSql.message_persisted_ms >= self.flo_start_ms - 0.5*3600*1000,
            MessageSql.message_persisted_ms <= self.flo_start_ms + 12*3600*1000,
        ).order_by(asc(MessageSql.message_persisted_ms)).all()

        self.flo_params_messages: List[MessageSql] = []
        self.flo_params_list: List[FloParamsHouse0] = []
        for m in flo_params_messages:
            if not [x for x in self.flo_params_messages if x.payload['StartUnixS'] == m.payload['StartUnixS']]:
                print(f"Found flo params at hour {self.unix_ms_to_date(m.payload['StartUnixS']*1000)}")
                self.flo_params_messages.append(m)
                self.flo_params_list.append(FloParamsHouse0(**m.payload))

        self.plots_times, self.plots_hps, self.plots_lmps, self.plots_energy = [], [], [], []
        for flo_params in self.flo_params_list:
            f = FloHinge(flo_params, hinge_hours=5, num_nodes=[10,3,3,3,3])
            self.plots_times.append(f.plot_time)
            self.plots_hps.append(f.plot_hp)
            self.plots_lmps.append(f.plot_lmp)
            self.plots_energy.append(f.plot_energy)