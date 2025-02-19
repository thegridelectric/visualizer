from flo import DGraph, DParams, to_kelvin
from named_types import FloParamsHouse0
from fake_models import MessageSql
import json
from typing import List
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


class HingeNode():
    def __init__(self, time_slice:int, params: DParams,
                 top_temp:float, middle_temp:float, bottom_temp:float, thermocline1:float, thermocline2: float):
        self.time_slice = time_slice
        self.top_temp = top_temp
        self.middle_temp = middle_temp
        self.bottom_temp = bottom_temp
        self.thermocline1 = thermocline1
        self.thermocline2 = thermocline2
        self.params = params
        self.energy = self.get_energy()

    def __repr__(self):
        if self.thermocline2 is not None:
            return f"{self.top_temp}({self.thermocline1}){self.middle_temp}({self.thermocline2}){self.bottom_temp}"
        else:
            return f"{self.top_temp}({self.thermocline1}){self.bottom_temp}"

    def get_energy(self):
        m_layer_kg = self.params.storage_volume*3.785 / self.params.num_layers
        if self.middle_temp is not None:
            kWh_top = (self.thermocline1-0.5)*m_layer_kg * 4.187/3600 * to_kelvin(self.top_temp)
            kWh_midlle = (self.thermocline2-self.thermocline1)*m_layer_kg * 4.187/3600 * to_kelvin(self.middle_temp)
            kWh_bottom = (self.params.num_layers-self.thermocline2+0.5)*m_layer_kg * 4.187/3600 * to_kelvin(self.bottom_temp)
        else:        
            kWh_top = (self.thermocline1-0.5)*m_layer_kg * 4.187/3600 * to_kelvin(self.top_temp)
            kWh_midlle = 0
            kWh_bottom = (self.params.num_layers-self.thermocline1+0.5)*m_layer_kg * 4.187/3600 * to_kelvin(self.bottom_temp)
        return kWh_top + kWh_midlle + kWh_bottom
    
    def plot(self, title=''):  
        norm = Normalize(vmin=self.params.available_top_temps[0]-20, vmax=self.params.available_top_temps[-1]+20)
        cmap = matplotlib.colormaps['Reds'] 
        mt = self.middle_temp if self.middle_temp is not None else self.bottom_temp 
        tank_top_colors = [cmap(norm(x)) for x in [self.top_temp]]
        tank_middle_colors = [cmap(norm(x)) for x in [mt]]
        tank_bottom_colors = [cmap(norm(x)) for x in [self.bottom_temp]]
        th2 = self.thermocline2 if self.thermocline2 is not None else self.thermocline1
        thermocline_reversed = self.params.num_layers - self.thermocline1 + 1
        thermocline2_reversed = self.params.num_layers - th2 + 1
        bars_top = plt.bar([0], [self.thermocline1], bottom=thermocline_reversed, color=tank_top_colors, alpha=0.9, width=0.5)
        bars_middle = plt.bar([0], [th2 - self.thermocline1], bottom=[thermocline2_reversed], color=tank_middle_colors, alpha=0.9, width=0.5)
        bars_bottom = plt.bar([0], [thermocline2_reversed], bottom=[0], color=tank_bottom_colors, alpha=0.9, width=0.5)
        plt.xlim([-1,1])
        plt.xticks([])
        for bar in bars_top:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, 
                     f'{int(self.top_temp)}', ha='center', va='center', color='white')
        if self.middle_temp is not None:
            for bar in bars_middle:
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, 
                        f'{int(mt)}', ha='center', va='center', color='white')
        for bar in bars_bottom:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, 
                     f'{int(self.bottom_temp)}', ha='center', va='center', color='white')
        plt.title(title)
        plt.show()

class FloHinge():

    def __init__(self, flo_params: FloParamsHouse0):
        # Graph in which charging is accurate
        self.g = DGraph(flo_params)
        self.g.solve_dijkstra()
        # Graph in which discharging is accurate
        flo_params.DdDeltaTF = 45
        self.dg = DGraph(flo_params)
        self.dg.solve_dijkstra()
        self.hinge_steps: List[HingeNode] = []
        self.start()

    def start(self):
        self.get_hinge_start_state()
        self.evaluate_branches()
        self.plot_hinge()

    def plot_hinge(self, combo=''):
        # Want to show initial, discharging, and the result of the 3 branches
        sp_time = list(range(self.turn_on_hour+3+1))
        sp_top_temp = []
        sp_middle_temp = []
        sp_bottom_temp = []
        sp_thermocline = []
        sp_thermocline2 = []
        for step in self.hinge_steps:
            sp_top_temp.append(step.top_temp)
            sp_bottom_temp.append(step.bottom_temp)
            sp_thermocline.append(step.thermocline1)
            if step.middle_temp is not None:
                sp_middle_temp.append(step.middle_temp)
                sp_thermocline2.append(step.thermocline2)
            else:
                sp_middle_temp.append(step.bottom_temp)
                sp_thermocline2.append(step.thermocline1)
        # Bottom plot
        norm = Normalize(vmin=self.g.params.available_top_temps[0]-20, vmax=self.g.params.available_top_temps[-1]+20)
        cmap = matplotlib.colormaps['Reds']
        tank_top_colors = [cmap(norm(x)) for x in sp_top_temp]
        tank_middle_colors = [cmap(norm(x)) for x in sp_middle_temp]
        tank_bottom_colors = [cmap(norm(x)) for x in sp_bottom_temp]

        # Reversing thermocline positions
        sp_thermocline_reversed1 = [self.g.params.num_layers - x + 1 for x in sp_thermocline]
        sp_thermocline_reversed2 = [self.g.params.num_layers - x + 1 for x in sp_thermocline2]

        # Stacking the temperatures and thermoclines
        bars_top = plt.bar(sp_time, sp_thermocline, bottom=sp_thermocline_reversed1, color=tank_top_colors, alpha=0.7, width=0.9)
        bars_middle = plt.bar(sp_time, [y-x for x,y in zip(sp_thermocline, sp_thermocline2)], bottom=sp_thermocline_reversed2, color=tank_middle_colors, alpha=0.7, width=0.9)
        bars_bottom = plt.bar(sp_time, sp_thermocline_reversed2, bottom=0, color=tank_bottom_colors, alpha=0.7, width=0.9)
        plt.xlabel('Time [hours]')
        plt.ylabel('Storage state')
        plt.ylim([0, self.g.params.num_layers])
        plt.yticks([])
        for i, bar in enumerate(bars_top):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, 
                    f'{int(sp_top_temp[i])}', ha='center', va='center', color='white')
        for i, bar in enumerate(bars_middle):
            if sp_middle_temp[i] != sp_bottom_temp[i]:
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, 
                        f'{int(sp_middle_temp[i])}', ha='center', va='center', color='white')
        for i, bar in enumerate(bars_bottom):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, 
                    f'{int(sp_bottom_temp[i])}', ha='center', va='center', color='white')
        plt.title(f"{'d-'*self.turn_on_hour}{self.best_combination if combo=='' else combo}-knit")
        plt.show()

    def get_hinge_start_state(self):
        # Find hour at which the HP is turned on (we trust the decisions to discharge)
        node_i = self.g.initial_node
        self.initial_node = HingeNode(
            time_slice = node_i.time_slice,
            top_temp = node_i.top_temp,
            middle_temp = node_i.middle_temp,
            bottom_temp = node_i.bottom_temp,
            thermocline1 = node_i.thermocline1,
            thermocline2 = node_i.thermocline2,
            params = self.g.params
            )
        self.hinge_steps.append(self.initial_node)
        for i in range(48):
            heat_out = [e.hp_heat_out for e in self.g.edges[node_i] if e.head==node_i.next_node][0]
            if heat_out > 2:
                self.turn_on_hour = i
                break
            node_i = node_i.next_node
        print(f"FLO turns on HP at hour {self.turn_on_hour}")
        # Find what the storage would look like if we discharged until then
        node_i = self.dg.initial_node
        for i in range(self.turn_on_hour):
            node_i = self.dg.edges[node_i][0].head
        self.turn_on_node = HingeNode(
            time_slice = node_i.time_slice,
            top_temp = node_i.top_temp,
            middle_temp = node_i.middle_temp,
            bottom_temp = node_i.bottom_temp,
            thermocline1 = node_i.thermocline1,
            thermocline2 = node_i.thermocline2,
            params = self.g.params
            )
        print(f"Estimated storage at the start of hour {self.turn_on_hour}: {self.turn_on_node}")
        self.hinge_steps.append(self.turn_on_node)

    def evaluate_branches(self):
        self.feasible_branches = {}
        for branch1_charge in [True, False]:
            for branch2_charge in [True, False]:
                for branch3_charge in [True, False]:
                    combination_name = f"{'C' if branch1_charge else 'D'}-"
                    combination_name += f"{'C' if branch2_charge else 'D'}-"
                    combination_name += f"{'C' if branch3_charge else 'D'}"
                    self.follow_branch(branch1_charge, branch2_charge, branch3_charge, combination_name)
        self.knit_branches()

        for branch in self.feasible_branches:
            print(f"\nCombination: {branch}")
            print(f"- Ends at {self.feasible_branches[branch]['final_state']}")
            print(f"- Knitted to {self.feasible_branches[branch]['knitted_to']}")
            print(f"- Total pathcost: {self.feasible_branches[branch]['total_pathcost']}")

        self.best_combination = min(self.feasible_branches, key=lambda k: self.feasible_branches[k]['total_pathcost'])
        print(f"\nThe best path forward is {self.best_combination}")

        for combo in self.feasible_branches:
            b1, b2, b3 = [True if x=='C' else False for x in combo.split('-')]
            self.hinge_steps = [self.initial_node, self.turn_on_node]
            self.follow_branch(b1, b2, b3, combo, final=True)
            self.hinge_steps.append(self.feasible_branches[combo]['knitted_to'])
            self.plot_hinge(combo=combo)


    def follow_branch(self, branch1_charge, branch2_charge, branch3_charge, combination_name, final=False):
        node0 = self.turn_on_node
        total_hinge_cost_usd = 0
        # First hour
        node1 = self.charge_from(node0) if branch1_charge else self.discharge_from(node0)
        if final:
            self.hinge_steps.append(node1)
        h = self.turn_on_node.time_slice
        if branch1_charge:
            total_hinge_cost_usd += self.g.params.elec_price_forecast[h] * self.g.params.max_hp_elec_in / 100
        else:
            RSWT = self.g.params.rswt_forecast[h]
            if node0.top_temp < RSWT or node1.top_temp < RSWT - self.g.params.delta_T(RSWT):
                return
        # Second hour
        node2 = self.charge_from(node1) if branch2_charge else self.discharge_from(node1)
        if final:
            self.hinge_steps.append(node2)
        h += 1
        if branch2_charge:
            total_hinge_cost_usd += self.g.params.elec_price_forecast[h] * self.g.params.max_hp_elec_in / 100
        else:
            RSWT = self.g.params.rswt_forecast[h]
            if node1.top_temp < RSWT or node2.top_temp < RSWT - self.g.params.delta_T(RSWT):
                return
        # Third hour
        node3 = self.charge_from(node2) if branch3_charge else self.discharge_from(node2)
        if final:
            self.hinge_steps.append(node3)
        h += 1
        if branch3_charge:
            total_hinge_cost_usd += self.g.params.elec_price_forecast[h] * self.g.params.max_hp_elec_in / 100
        else:
            RSWT = self.g.params.rswt_forecast[h]
            if node2.top_temp < RSWT or node3.top_temp < RSWT - self.g.params.delta_T(RSWT):
                return
        # Add to feasible branches
        if not final:
            self.feasible_branches[combination_name] = {'hinge_cost': total_hinge_cost_usd, 'final_state': node3}


    def discharge_from(self, n: HingeNode):
        next_node_top_temp = n.top_temp
        next_node_energy = n.energy - self.g.params.load_forecast[n.time_slice]
        
        if n.top_temp - self.dg.params.delta_T(n.top_temp) < n.bottom_temp or n.middle_temp is not None:
            flo_params_temporary: FloParamsHouse0 = self.dg.params.config.model_copy()
            flo_params_temporary.HorizonHours = 2
            flo_params_temporary.InitialTopTempF = n.top_temp if n.top_temp<=175 else 175
            flo_params_temporary.InitialBottomTempF = n.bottom_temp if n.middle_temp is None else n.middle_temp
            flo_params_temporary.InitialThermocline = n.thermocline1 if n.thermocline2 is None else (self.dg.params.num_layers-n.thermocline2+n.thermocline1)
            temporary_g = DGraph(flo_params_temporary)
            node_after = min(temporary_g.nodes[0], key=lambda x: abs(x.energy-next_node_energy))
            next_node_top_temp = node_after.top_temp
            next_node_middle_temp = node_after.middle_temp
            next_node_bottom_temp = node_after.bottom_temp
            next_node_thermocline = node_after.thermocline1
            next_node_thermocline2 = node_after.thermocline2
        else:
            temporary_g = None
            next_node_middle_temp = None
            next_node_bottom_temp = n.bottom_temp
            next_node_thermocline2 = None
            # Find thermocline position such that kWh_top + kWh_bottom = next_node_energy
            m_layer_kg = self.g.params.storage_volume*3.785 / self.g.params.num_layers       
            top, bottom = to_kelvin(next_node_top_temp), to_kelvin(next_node_bottom_temp)
            A = m_layer_kg * 4.187/3600
            next_node_thermocline = int(1/(top-bottom) * (next_node_energy/A - (-0.5*top + (self.g.params.num_layers+0.5)*bottom)))
            while next_node_thermocline < 1:
                next_node_top_temp = next_node_bottom_temp
                next_node_bottom_temp = round(next_node_bottom_temp - self.g.params.delta_T(next_node_bottom_temp))
                top, bottom = to_kelvin(next_node_top_temp), to_kelvin(next_node_bottom_temp)
                next_node_thermocline = int(1/(top-bottom) * (next_node_energy/A - (-0.5*top + (self.g.params.num_layers+0.5)*bottom)))

        next_node = HingeNode(
            time_slice = n.time_slice+1,
            top_temp = next_node_top_temp,
            middle_temp = next_node_middle_temp,
            bottom_temp = next_node_bottom_temp,
            thermocline1 = next_node_thermocline,
            thermocline2 = next_node_thermocline2,
            params = self.g.params
        )
        return next_node

    def charge_from(self, n: HingeNode):
        next_node_bottom_temp = n.bottom_temp
        load = self.g.params.load_forecast[n.time_slice]
        hp = self.g.params.max_hp_elec_in * self.g.params.COP(self.g.params.oat_forecast[n.time_slice], 0)
        heat_to_store = hp - load
        next_node_energy = n.energy + heat_to_store

        if n.bottom_temp + self.g.params.delta_T(n.bottom_temp) < n.top_temp:
            if n.middle_temp is not None:
                top_mixed = (n.top_temp*n.thermocline1 + n.middle_temp*(n.thermocline2-n.thermocline1))/n.thermocline2
                next_node_top_temp = round(
                    (top_mixed*n.thermocline2 + (n.bottom_temp+self.g.params.delta_T(n.bottom_temp))*(self.g.params.num_layers-n.thermocline2))/self.g.params.num_layers
                    )
            else:
                next_node_top_temp = round(
                    n.thermocline1/self.g.params.num_layers * n.top_temp 
                    + (self.g.params.num_layers-n.thermocline1)/self.g.params.num_layers * (n.bottom_temp + self.g.params.delta_T(n.bottom_temp))
                    )
        else:
            next_node_top_temp = n.top_temp

        # Find thermocline position such that kWh_top + kWh_bottom = next_node_energy
        m_layer_kg = self.g.params.storage_volume*3.785 / self.g.params.num_layers       
        top, bottom = to_kelvin(next_node_top_temp), to_kelvin(next_node_bottom_temp)
        A = m_layer_kg * 4.187/3600
        next_node_thermocline = int(1/(top-bottom) * (next_node_energy/A - (-0.5*top + (self.g.params.num_layers+0.5)*bottom)))

        while next_node_thermocline > self.g.params.num_layers:
            next_node_bottom_temp = next_node_top_temp
            next_node_top_temp = round(next_node_top_temp + self.g.params.delta_T(next_node_top_temp))
            top, bottom = to_kelvin(next_node_top_temp), to_kelvin(next_node_bottom_temp)
            next_node_thermocline = int(1/(top-bottom) * (next_node_energy/A - (-0.5*top + (self.g.params.num_layers+0.5)*bottom)))

        next_node = HingeNode(
            time_slice = n.time_slice+1,
            top_temp = next_node_top_temp,
            middle_temp = None,
            bottom_temp = next_node_bottom_temp,
            thermocline1 = next_node_thermocline,
            thermocline2 = None,
            params = self.g.params
        )
        return next_node

    def knit_branches(self):
        for branch in self.feasible_branches:
            n: HingeNode = self.feasible_branches[branch]['final_state']

            top_temp = min(self.g.params.available_top_temps, key= lambda x: abs(x-n.top_temp))
            nodes_with_same_top = []
            # nodes_with_same_top = [x for x in self.g.nodes[self.turn_on_hour+3] if x.top_temp==top_temp]
            # nodes_with_same_top = [x for x in nodes_with_same_top if x.thermocline1==n.thermocline1]
            if not nodes_with_same_top:
                nodes_not_same_top = [n for n in self.g.nodes[self.turn_on_hour+3] if n.top_temp!=top_temp]
                closest_one = min(nodes_not_same_top, key=lambda x: abs(x.top_temp-top_temp))
                nodes_with_same_top = [n for n in nodes_with_same_top if n.top_temp==closest_one.top_temp and n.thermocline1==n.thermocline1]
            if not nodes_with_same_top:
                nodes_with_same_top = [min(self.g.nodes[n.time_slice], key= lambda x: abs(x.energy-n.energy))]
            knitted_node = nodes_with_same_top[0]
            
            self.feasible_branches[branch]['knitted_to'] = knitted_node
            self.feasible_branches[branch]['total_pathcost'] = round(knitted_node.pathcost + self.feasible_branches[branch]['hinge_cost'],2)


if __name__ == '__main__':

    # ----------------------------------
    # Load a FLO params
    def from_dict_msg(data):
        message = MessageSql(
                message_id=data["MessageId"],
                from_alias=data["FromAlias"],
                message_type_name=data["MessageTypeName"],
                message_persisted_ms=data["MessagePersistedMs"],
                payload=data["Payload"],
                message_created_ms=data.get("MessageCreatedMs")  # This is optional
            )
        return message
    with open('messages.json', 'r') as file:
        messages_dict = json.load(file)
    message_loaded = from_dict_msg(messages_dict)
    flo_params = FloParamsHouse0(**message_loaded.payload)
    # ----------------------------------
    
    f = FloHinge(flo_params)
            