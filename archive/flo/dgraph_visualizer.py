import os
import pytz
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from datetime import datetime, timedelta
from openpyxl.drawing.image import Image
from openpyxl.worksheet.worksheet import Worksheet
from gridflo import DGraph
from gridflo.dijkstra_types import DNode


class DGraphVisualizer():
    def __init__(self, d_graph: DGraph):
        self.g: DGraph = d_graph

    def plot(self, show=True, print_details=False):
        self.g.find_initial_node()
        # Walk along the shortest path (sp)
        sp_top_temp = []
        sp_middle_temp = []
        sp_bottom_temp = []
        sp_thermocline = []
        sp_thermocline2 = []
        sp_hp_heat_out = []
        sp_stored_energy = []
        node_i = self.g.initial_node
        the_end = False
        while not the_end:
            if node_i.next_node is None:
                the_end = True
                sp_hp_heat_out.append(edge_i.hp_heat_out)
            else:
                edge_i = [e for e in self.g.edges[node_i] if e.head==node_i.next_node][0]
                if print_details:
                    losses = self.g.params.storage_losses_percent/100 * (node_i.energy-self.g.min_node_energy)
                    energy_to_store = edge_i.hp_heat_out-self.g.params.load_forecast[node_i.time_slice]-losses
                    model_energy_to_store = edge_i.head.energy-edge_i.tail.energy
                    if model_energy_to_store>energy_to_store:
                        print(f"\n{edge_i}, model thinks {abs(round(model_energy_to_store-energy_to_store,1))} kWh more in store than reality")
                    elif model_energy_to_store<energy_to_store:
                        print(f"\n{edge_i}, model thinks {abs(round(model_energy_to_store-energy_to_store,1))} kWh less in store than reality")
                    else:
                        print(f"\n{edge_i}, model could not be more accurate!")
                sp_hp_heat_out.append(edge_i.hp_heat_out)
            sp_top_temp.append(node_i.top_temp)
            sp_bottom_temp.append(node_i.bottom_temp)
            sp_thermocline.append(node_i.thermocline1)
            sp_middle_temp.append(node_i.middle_temp)
            sp_thermocline2.append(node_i.thermocline2)
            sp_stored_energy.append(node_i.energy)
            node_i = node_i.next_node
        sp_soc = [(x-self.g.min_node_energy) / (self.g.max_node_energy-self.g.min_node_energy) * 100 for x in sp_stored_energy]
        sp_time = list(range(self.g.params.horizon+1))
        
        # Plot the shortest path
        fig, ax = plt.subplots(2,1, sharex=False, figsize=(12,5.5))
        start = datetime.fromtimestamp(self.g.params.start_time, tz=pytz.timezone("America/New_York")).strftime('%Y-%m-%d %H:%M')
        end = (datetime.fromtimestamp(self.g.params.start_time, tz=pytz.timezone("America/New_York")) 
               + timedelta(hours=self.g.params.horizon)).strftime('%Y-%m-%d %H:%M')
        fig.suptitle(f'From {start} to {end}\nCost: {round(self.g.initial_node.pathcost,2)} $', fontsize=10)

        # Top plot
        ax[0].step(sp_time, sp_hp_heat_out, where='post', color='tab:blue', alpha=0.6, label='HP')
        ax[0].step(sp_time[:-1], self.g.params.load_forecast, where='post', color='tab:red', alpha=0.6, label='Load')
        ax[0].legend(loc='upper left')
        ax[0].set_ylabel('Heating power [kW]')
        if max(sp_hp_heat_out)>0:
            ax[0].set_ylim([-0.5, 1.5*max(sp_hp_heat_out)])
        ax2 = ax[0].twinx()
        ax2.step(sp_time[:-1], self.g.params.elec_price_forecast, where='post', color='gray', alpha=0.6, label='Electricity price')
        ax2.legend(loc='upper right')
        ax2.set_ylabel('Electricity price [cts/kWh]')
        m = 0 if min(self.g.params.elec_price_forecast)>0 else min(self.g.params.elec_price_forecast)-5
        ax2.set_ylim([m,max(self.g.params.elec_price_forecast)*1.3])
        
        # Bottom plot
        norm = Normalize(vmin=60, vmax=180)
        cmap = matplotlib.colormaps['Reds']
        tank_top_colors = [cmap(norm(x)) for x in sp_top_temp]
        tank_middle_colors = [cmap(norm(x)) for x in sp_middle_temp]
        tank_bottom_colors = [cmap(norm(x)) for x in sp_bottom_temp]

        # Reversing thermocline positions
        sp_thermocline_reversed1 = [self.g.params.num_layers - x for x in sp_thermocline]
        sp_thermocline_reversed2 = [self.g.params.num_layers - x for x in sp_thermocline2]

        # Stacking the temperatures and thermoclines
        bars_top = ax[1].bar(sp_time, 
                             sp_thermocline, 
                             bottom=sp_thermocline_reversed1, 
                             color=tank_top_colors, alpha=0.7, width=0.9) #, align='edge')
        bars_middle = ax[1].bar(sp_time, 
                                [y-x for x,y in zip(sp_thermocline, sp_thermocline2)], 
                                bottom=sp_thermocline_reversed2, 
                                color=tank_middle_colors, alpha=0.7, width=0.9) #, align='edge')
        bars_bottom = ax[1].bar(sp_time, 
                                sp_thermocline_reversed2, 
                                bottom=0, 
                                color=tank_bottom_colors, alpha=0.7, width=0.9) #, align='edge')
        ax[1].set_xlabel('Time [hours]')
        ax[1].set_ylabel('Storage state')
        ax[1].set_ylim([0, self.g.params.num_layers])
        ax[1].set_yticks([])
        if len(sp_time)>10 and len(sp_time)<50:
            start_time = datetime.fromtimestamp(self.g.params.start_time, tz=pytz.timezone("America/New_York"))
            sp_time_hours = [f"{(start_time+timedelta(hours=x)).hour}:00" for x in range(0,len(sp_time)+1,2)]
            ax[0].set_xticks(list(range(0,len(sp_time)+1,2)))
            ax[0].set_xticklabels(sp_time_hours, fontsize=8)
            ax[1].set_xticks(list(range(0,len(sp_time)+1,2)))
            ax[1].set_xticklabels(sp_time_hours, fontsize=8)
        for i, bar in enumerate(bars_top):
            height = bar.get_height()
            bar_color = 'white'
            if i < len(self.g.params.rswt_forecast) and self.g.params.rswt_forecast[i] <= sp_top_temp[i]:
                bar_color = 'green'
            elif sp_top_temp[i]<100:
                bar_color = 'gray'
            ax[1].text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2, 
                    f'{int(sp_top_temp[i])}', ha='center', va='center', color=bar_color, fontsize=6)
        for i, bar in enumerate(bars_middle):
            height = bar.get_height()
            bar_color = 'white'
            if i < len(self.g.params.rswt_forecast) and self.g.params.rswt_forecast[i] <= sp_middle_temp[i]:
                bar_color = 'green'
            elif sp_middle_temp[i]<100:
                bar_color = 'gray'
            if height>1:
                ax[1].text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2, 
                        f'{int(sp_middle_temp[i])}', ha='center', va='center', color=bar_color, fontsize=6)
        for i, bar in enumerate(bars_bottom):
            height = bar.get_height()
            bar_color = 'white'
            if i < len(self.g.params.rswt_forecast) and self.g.params.rswt_forecast[i] <= sp_bottom_temp[i]:
                bar_color = 'green'
            elif sp_bottom_temp[i]<100:
                bar_color = 'gray'
            if sp_thermocline2[i]==self.g.params.num_layers:
                continue
            ax[1].text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2, 
                    f'{int(sp_bottom_temp[i])}', ha='center', va='center', color=bar_color, fontsize=6)
        ax3 = ax[1].twinx()
        ax3.plot(sp_time, sp_soc, color='black', alpha=0.4, label='SoC')
        ax3.set_ylabel('State of charge [%]')
        ax3.set_ylim([-1,101])

        # Color bar
        # boundaries = sorted(list(range(60,175,5)), reverse=False)
        # colors = [plt.cm.Reds(i/(len(boundaries)-1)) for i in range(len(boundaries))]
        # cmap = ListedColormap(colors)
        # norm = BoundaryNorm(boundaries, cmap.N, clip=True)
        # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.06, pad=0.15, alpha=0.7)
        # cbar.set_ticks(sorted(list(range(60,175,5)), reverse=False))
        # cbar.set_label('Temperature [F]')
        
        plt.tight_layout()
        plt.savefig('plot.png', dpi=130)
        if show:
            plt.show()
        plt.close()

    def export_to_excel(self):        
        # Add the parameters to a seperate sheet
        parameters = self.g.params.flo_params.to_dict()
        parameters_df = pd.DataFrame(list(parameters.items()), columns=['Variable', 'Value'])

        # Add the PQ pairs to a seperate sheet and plot the curve
        self.g.generate_bid()
        prices = [x.PriceTimes1000 for x in self.g.pq_pairs]
        quantities = [x.QuantityTimes1000/1000 for x in self.g.pq_pairs]
        pqpairs_df = pd.DataFrame({'price':[x/1000 for x in prices], 'quantity':quantities})
        # To plot quantities on x-axis and prices on y-axis
        ps, qs = [], []
        index_p = 0
        expected_price_usd_mwh = self.g.params.elec_price_forecast[0] * 10
        for p in sorted(list(range(min(prices), max(prices)+1)) + [expected_price_usd_mwh*1000]):
            ps.append(p/1000)
            if index_p+1 < len(prices) and p >= prices[index_p+1]:
                index_p += 1
            if p == expected_price_usd_mwh*1000:
                interesection = (quantities[index_p], expected_price_usd_mwh)
            qs.append(quantities[index_p])
        plt.plot(qs, ps, label='demand (bid)')
        prices = [x.PriceTimes1000/1000 for x in self.g.pq_pairs]
        plt.scatter(quantities, prices)
        plt.plot([min(quantities)-1, max(quantities)+1],[expected_price_usd_mwh]*2, label="supply (expected market price)")
        plt.scatter(interesection[0], interesection[1])
        plt.text(interesection[0]+0.25, interesection[1]+15, f'({round(interesection[0],3)}, {round(interesection[1],1)})', 
                 fontsize=10, color='tab:orange')
        plt.xticks(quantities)
        if min([abs(x-expected_price_usd_mwh) for x in prices]) < 5:
            plt.yticks(prices)
        else:
            plt.yticks(prices + [expected_price_usd_mwh])
        plt.ylabel("Price [USD/MWh]")
        plt.xlabel("Quantity [kWh]")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('plot_pq.png', dpi=130)
        plt.close()

        # Write to Excel
        # start = datetime.fromtimestamp(self.g.params.start_time, tz=pytz.timezone("America/New_York")).strftime('%Y-%m-%d %H:%M')
        # os.makedirs('results', exist_ok=True)
        # file_path = os.path.join('results', f'result_{start}.xlsx')
        file_path = 'result.xlsx'
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:

            self.plot(show=False)
            plot_sheet: Worksheet = writer.book.create_sheet(title='Plot')
            plot_sheet.add_image(Image('plot.png'), 'A1')

            parameters_df.to_excel(writer, index=False, sheet_name='Parameters')

            plot2_sheet: Worksheet = writer.book.create_sheet(title='PQ pairs')
            pqpairs_df.to_excel(writer, index=False, sheet_name='PQ pairs')
            plot2_sheet.add_image(Image('plot_pq.png'), 'C1')

        os.remove('plot.png')        
        os.remove('plot_pq.png')


class DNodeVisualizer():
    def __init__(self, node: DNode):
        self.n = node

    def plot(self):
        norm = Normalize(vmin=80, vmax=190)
        cmap = matplotlib.colormaps['Reds'] 
        tank_top_colors = [cmap(norm(x)) for x in [self.n.top_temp]]
        tank_middle_colors = [cmap(norm(x)) for x in [self.n.middle_temp]]
        tank_bottom_colors = [cmap(norm(x)) for x in [self.n.bottom_temp]]
        thermocline1_reversed = self.n.params.num_layers - self.n.thermocline1
        thermocline2_reversed = self.n.params.num_layers - self.n.thermocline2
        bars_top = plt.bar(
            [0], [self.n.thermocline1], bottom=thermocline1_reversed, 
            color=tank_top_colors, alpha=0.9, width=0.5
        )
        bars_middle = plt.bar(
            [0], [self.n.thermocline2 - self.n.thermocline1], bottom=[thermocline2_reversed], 
            color=tank_middle_colors, alpha=0.9, width=0.5
        )
        bars_bottom = plt.bar(
            [0], [thermocline2_reversed], bottom=[0], 
            color=tank_bottom_colors, alpha=0.9, width=0.5
        )
        for bar in bars_top:
            plt.text(
                bar.get_x() + bar.get_width()/2, bar.get_y() + bar.get_height()/2, 
                f'{int(self.n.top_temp)}', ha='center', va='center', color='white'
            )
        for bar in bars_middle:
            plt.text(
                bar.get_x() + bar.get_width()/2, bar.get_y() + bar.get_height()/2, 
                f'{int(self.n.middle_temp)}', ha='center', va='center', color='white'
            )
        for bar in bars_bottom:
            plt.text(
                bar.get_x() + bar.get_width()/2, bar.get_y() + bar.get_height()/2, 
                f'{int(self.n.bottom_temp)}', ha='center', va='center', color='white'
            )
        plt.xlim([-1,1])
        plt.xticks([])
        plt.title(repr(self))
        plt.show()