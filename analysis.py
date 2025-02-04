from flo import DGraph
from named_types import FloParamsHouse0
import dotenv
import pendulum
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker
from fake_config import Settings
from fake_models import MessageSql
import os
import io
import matplotlib.pyplot as plt
import base64
from fastapi.responses import StreamingResponse
from datetime import datetime 
import pytz

def download_excel(house_alias, start_ms):
    print("Finding latest Dijstra...")
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

    print(f"Found up to time {pendulum.from_timestamp(flo_params_msg.message_persisted_ms/1000, tz='America/New_York')}")

    if not flo_params_msg:
        print("No FLO params")
        if os.path.exists('result.xlsx'):
            os.remove('result.xlsx')
        return

    # for key, value in flo_params_msg.payload.items():
    #     # print(f'{key}: {value}')
    #     if key=='AlphaTimes10': 
    #         flo_params_msg.payload[key] = 103
    #     elif key=='BetaTimes100': 
    #         flo_params_msg.payload[key] = -19
    #     elif key=='GammaEx6':
    #         flo_params_msg.payload[key] = 1500
    #     elif key=='DdPowerKw':
    #         flo_params_msg.payload[key] = 10.3

    flo_params = FloParamsHouse0(**flo_params_msg.payload)

    print("Running Dijkstra and saving analysis to excel...")
    g = DGraph(flo_params)
    g.solve_dijkstra()
    g.export_to_excel()
    print("Done.")

# just_before = pendulum.datetime(2025, 1, 18, 9, 5, tz="America/New_York").timestamp()*1000
# download_excel("oak", just_before)

# def generate_excel(params:FloParamsHouse0):
#     print("Running Dijkstra and saving analysis to excel...")
#     g = DGraph(params)
#     g.solve_dijkstra()
#     g.export_to_excel()
#     print("Done.")

# flo_parameters = FloParamsHouse0(

# )

import zipfile
import base64

def get_bids(house_alias, start_ms, end_ms):
    print("Getting bids...")

    settings = Settings(_env_file=dotenv.find_dotenv())
    engine = create_engine(settings.db_url.get_secret_value())
    Session = sessionmaker(bind=engine)
    session = Session()

    flo_params_msg = session.query(MessageSql).filter(
        MessageSql.message_type_name == "flo.params.house0",
        MessageSql.from_alias.like(f'%{house_alias}%'),
        MessageSql.message_persisted_ms >= start_ms,
        MessageSql.message_persisted_ms <= end_ms,
    ).order_by(desc(MessageSql.payload['StartUnixS'])).all()

    print(f"Found {len(flo_params_msg)} FLOs for {house_alias}")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:

        for i, flo_params_m in enumerate(flo_params_msg):
            try:
                flo_params = FloParamsHouse0(**flo_params_m.payload)
                print("Getting pq pairs plot...")
                g = DGraph(flo_params)
                g.solve_dijkstra()
                
                pq_pairs = g.generate_bid()
                prices = [x.PriceTimes1000 for x in pq_pairs]
                quantities = [x.QuantityTimes1000/1000 for x in pq_pairs]
                # To plot quantities on x-axis and prices on y-axis
                ps, qs = [], []
                index_p = 0
                expected_price_usd_mwh = g.params.elec_price_forecast[0] * 10
                for p in sorted(list(range(min(prices), max(prices)+1)) + [expected_price_usd_mwh*1000]):
                    ps.append(p/1000)
                    if index_p+1 < len(prices) and p >= prices[index_p+1]:
                        index_p += 1
                    if p == expected_price_usd_mwh*1000:
                        interesection = (quantities[index_p], expected_price_usd_mwh)
                    qs.append(quantities[index_p])
                plt.plot(qs, ps, label='demand (bid)')
                prices = [x.PriceTimes1000/1000 for x in pq_pairs]
                plt.scatter(quantities, prices)
                plt.plot([min(quantities)-1, max(quantities)+1],[expected_price_usd_mwh]*2, label="supply (expected market price)")
                plt.scatter(interesection[0], interesection[1])
                plt.text(interesection[0]+0.25, interesection[1]+15, f'({round(interesection[0],3)}, {round(interesection[1],1)})', fontsize=10, color='tab:orange')
                plt.xticks(quantities)
                if min([x-expected_price_usd_mwh for x in prices]) < 5:
                    plt.yticks(prices)
                else:
                    plt.yticks(prices + [expected_price_usd_mwh])
                plt.yticks(prices+[expected_price_usd_mwh])
                plt.ylabel("Price [USD/MWh]")
                plt.xlabel("Quantity [kWh]")
                plt.title(datetime.fromtimestamp(g.params.start_time, tz=pytz.timezone("America/New_York")).strftime('%Y-%m-%d %H:%M'))
                plt.grid(alpha=0.3)
                plt.legend()
                plt.tight_layout()

                img_buf = io.BytesIO()
                plt.savefig(img_buf, format='png', dpi=300)
                img_buf.seek(0)  # Move to the beginning of the BytesIO object
                
                # Write the image to the zip file
                zip_file.writestr(f'pq_plot_{i}.png', img_buf.getvalue())
                plt.close()

            except Exception as e:
                print(f"Error generating plot for FLO: {e}")

    zip_buffer.seek(0)
    return StreamingResponse(zip_buffer, media_type='application/zip', headers={"Content-Disposition": "attachment; filename=plots.zip"})
