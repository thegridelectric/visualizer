from typing import Dict, List, Tuple
import json
import time
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from named_types import FloParamsHouse0
from dijkstra_types import DParams, DNode, to_kelvin


class SuperGraphGenerator():
    def __init__(self, flo_params: FloParamsHouse0):
        self.params = DParams(flo_params)

    def generate(self):
        start_time = time.time()
        self.create_nodes()
        self.create_edges()
        self.save_to_json()
        print(f"\nGenerating SuperGraph took {int(time.time()-start_time)} seconds.")

    def create_nodes(self):
        print("Creating nodes...")
        self.top_temps = sorted(list(range(110,170+10,10)), reverse=True)
        self.middle_temps = [x-10 for x in self.top_temps[:-1]] 
        self.bottom_temps = [100]

        thermocline_combinations = []
        for t1 in range(1,self.params.num_layers+1):
            for t2 in range(1,self.params.num_layers+1):
                if t2>=t1:
                    thermocline_combinations.append((t1,t2))
        print(f"=> {len(thermocline_combinations)} thermocline combinations")

        temperature_combinations = []
        for t in self.top_temps:
            for m in self.middle_temps:
                for b in self.bottom_temps:
                    if b<=m-10 and m<=t-10:
                        temperature_combinations.append((t,m,b))
        print(f"=> {len(temperature_combinations)} temperature combinations")

        temperature_combinations += [(110,100,100)]
        temperature_combinations += [(100,80,80), (80,60,60)]

        self.nodes: List[DNode] = []
        self.nodes_by: Dict[Tuple, Dict[Tuple, DNode]] = {
            tmb: {th: None for th in thermocline_combinations} for tmb in temperature_combinations
            }
                
        total_nodes=0
        for tmb in temperature_combinations:
            for th in thermocline_combinations:
                t, m, b = tmb
                th1, th2 = th
                if m==b and th1!=th2:
                    continue
                node = DNode(
                    top_temp=t,
                    middle_temp=m,
                    bottom_temp=b,
                    thermocline1=th1,
                    thermocline2=th2,
                    parameters=self.params
                    )
                self.nodes.append(node)
                self.nodes_by[tmb][th] = node
                total_nodes += 1     
        print(f"=> Created a total of {total_nodes} nodes")

    def create_edges(self):
        print("\nCreating edges...")
        max_hp_out = self.params.max_hp_elec_in * self.params.COP(oat=50)
        max_load = max_hp_out
        store_heat_in_range = [x/10 for x in range(-int(max_load*10), int(max_hp_out*10)+1)]
        print(
            f"The store can receive heat in the "
            f"[{store_heat_in_range[0]}, {store_heat_in_range[-1]}] kWh range, with a 0.1 kWh step."
            )
        
        allowed_temperatures = {
            'top': self.top_temps,
            'middle': self.middle_temps,
            'bottom': self.bottom_temps
        }
        storage_model = DataBasedStorageModel(self.params.flo_params, self.nodes, self.nodes_by, allowed_temperatures)
        self.super_graph: Dict[str, Dict[str, str]] = {}

        for store_heat_in in store_heat_in_range:
            print(f"Generating edges for store_heat_in = {store_heat_in} kWh")
            self.super_graph[str(store_heat_in)] = {}
            for node in self.nodes:
                predicted_next_node = storage_model.next_node(node, store_heat_in)
                closest_next_node = self.find_closest_node(predicted_next_node, print_detail=False)
                # print(f"True: {predicted_next_node}, associated to {closest_next_node}")
                self.super_graph[str(store_heat_in)][node.to_string()] = closest_next_node.to_string()

    def find_closest_node(self, true_n: DNode, print_detail: bool=False) -> DNode:
        if print_detail: print(f"Looking for closest of {true_n}")

        # Cold nodes
        if true_n.top_temp <= 100:
            closest_top_temp = min([100,80], key= lambda x: abs(x-true_n.top_temp))
            nodes_with_similar_temps = [
                n for n in self.nodes if 
                n.top_temp==closest_top_temp and 
                n.middle_temp==closest_top_temp-20 and 
                n.bottom_temp==closest_top_temp-20 and
                n.thermocline1==true_n.thermocline1 and
                n.thermocline2==true_n.thermocline2
            ]
            if not nodes_with_similar_temps:
                nodes_with_similar_temps = [
                    n for n in self.nodes if 
                    n.top_temp==closest_top_temp and 
                    n.middle_temp==closest_top_temp-20 and 
                    n.bottom_temp==closest_top_temp-20 and
                    n.thermocline1==true_n.thermocline1
                    ]
            closest_node = min(nodes_with_similar_temps, key = lambda x: abs(x.energy-true_n.energy))
            return closest_node

        # Find closest available top, middle and bottom temps
        closest_top_temp = min(self.top_temps, key=lambda x: abs(float(x)-true_n.top_temp))
        closest_middle_temp = min(self.middle_temps, key=lambda x: abs(float(x)-true_n.middle_temp))
        closest_bottom_temp = min(self.bottom_temps, key=lambda x: abs(float(x)-true_n.bottom_temp))

        # Need at least 10F between top and middle
        if closest_top_temp - closest_middle_temp < 10:
            closest_middle_temp = closest_top_temp-10 if closest_top_temp>115 else 100

        # Correct for the 120,100,100 case
        if closest_top_temp == 120 and closest_middle_temp==100:
            closest_middle_temp = 110

        if print_detail: print(f"{closest_top_temp},{closest_middle_temp},{closest_bottom_temp}")

        # Top temperature is impossible to reach
        if true_n.top_temp > max(self.top_temps):
            nodes_with_similar_temps = [
                n for n in self.nodes if 
                n.top_temp == max(self.top_temps) and
                n.middle_temp==closest_middle_temp and
                n.bottom_temp==closest_bottom_temp and
                n.thermocline2==true_n.thermocline2
            ]
            closest_node = min(nodes_with_similar_temps, key = lambda x: abs(x.energy-true_n.energy))
            return closest_node

        # Both top and middle were rounded above
        if closest_top_temp > true_n.top_temp and closest_middle_temp > true_n.middle_temp:
            nodes_with_similar_temps = [
                n for n in self.nodes if 
                n.top_temp<=closest_top_temp and
                n.top_temp>=closest_top_temp-10 and
                n.middle_temp<=closest_middle_temp and
                n.middle_temp>=closest_middle_temp-10 and
                n.bottom_temp==closest_bottom_temp and
                n.thermocline2==true_n.thermocline2
            ]
            closest_node = min(nodes_with_similar_temps, key = lambda x: abs(x.energy-true_n.energy))
            return closest_node
        
        # Both top and middle were rounded below
        if closest_top_temp < true_n.top_temp and closest_bottom_temp < true_n.middle_temp:
            nodes_with_similar_temps = [
                n for n in self.nodes if 
                n.top_temp<=closest_top_temp+10 and
                n.top_temp>=closest_top_temp and
                n.middle_temp<=closest_middle_temp+10 and
                n.middle_temp>=closest_middle_temp and
                n.bottom_temp==closest_bottom_temp and
                n.thermocline2==true_n.thermocline2
            ]
            closest_node = min(nodes_with_similar_temps, key = lambda x: abs(x.energy-true_n.energy))
            return closest_node

        # Top was rounded above but not middle: flexible th1
        if closest_top_temp > true_n.top_temp:
            nodes_with_similar_temps = [
                n for n in self.nodes if 
                n.top_temp==closest_top_temp and
                n.middle_temp==closest_middle_temp and
                n.bottom_temp==closest_bottom_temp and
                n.thermocline2==true_n.thermocline2
            ]
            closest_node = min(nodes_with_similar_temps, key = lambda x: abs(x.energy-true_n.energy))
            return closest_node

        # Middle was rounded above but not top: flexible th2
        if closest_top_temp > true_n.top_temp:
            nodes_with_similar_temps = [
                n for n in self.nodes if 
                n.top_temp==closest_top_temp and
                n.middle_temp==closest_middle_temp and
                n.bottom_temp==closest_bottom_temp and
                n.thermocline1==true_n.thermocline1
            ]
            closest_node = min(nodes_with_similar_temps, key = lambda x: abs(x.energy-true_n.energy))
            return closest_node

        nodes_with_similar_temps = [
            n for n in self.nodes if 
            n.top_temp==closest_top_temp and 
            n.middle_temp==closest_middle_temp and 
            n.bottom_temp==closest_bottom_temp and
            n.thermocline1==true_n.thermocline1 and
            n.thermocline2==true_n.thermocline2
        ]

        if not nodes_with_similar_temps:
            nodes_with_similar_temps = [
                n for n in self.nodes if 
                n.top_temp==closest_top_temp and 
                n.middle_temp==closest_middle_temp and 
                n.bottom_temp==closest_bottom_temp and
                n.thermocline1==true_n.thermocline1
                ]
            
        closest_node = min(nodes_with_similar_temps, key = lambda x: abs(x.energy-true_n.energy))
        return closest_node
        
    def save_to_json(self):
        print("\nSaving SuperGraph to JSON...")
        json_file_path = "super_graph.json"
        with open(json_file_path, 'w') as f:
            json.dump(self.super_graph, f)
        print("Done.")


class DataBasedStorageModel():
    def __init__(self, flo_params: FloParamsHouse0, nodes: List, nodes_by: Dict, allowed_temperatures: Dict):
        self.params = DParams(flo_params)
        self.nodes: List[DNode] = nodes
        self.nodes_by: Dict[Tuple, Dict[Tuple, DNode]] = nodes_by
        self.top_temps = allowed_temperatures['top']
        self.middle_temps = allowed_temperatures['middle']
        self.bottom_temps = allowed_temperatures['bottom']
        data_file = 'storage_data_maple.csv'
        if os.path.exists(data_file):
            self.data_file = data_file
        else:
            raise Exception(f"Could not find {data_file}")
        self.learn_from_data()

    def next_node(self, n:DNode, store_heat_in:float, print_detail:bool=True) -> DNode:
        data_inputs = pd.DataFrame({
            'store_heat_in': store_heat_in,
            'init_t': n.top_temp,
            'init_m': n.middle_temp,
            'init_b': n.bottom_temp,
            'init_th1': n.thermocline1,
            'init_th2': n.thermocline2,
        }, index=[0])
        predicted_final_state = [int(x) for x in list(list(self.data_based_model.predict(data_inputs))[0])]
        t, m, b, th1, th2 = predicted_final_state
        th1 = int(max(1, min(24, th1)))
        th2 = int(max(1, min(24, th2)))

        predicted_next_node = DNode(
            top_temp = t,
            middle_temp = m,
            bottom_temp = b,
            thermocline1 = th1,
            thermocline2 = th2,
            parameters=n.params
        )
        if print_detail: print(f"{n} --{store_heat_in}--> {predicted_next_node}")
        return predicted_next_node
    
    def learn_from_data(self):
        df = pd.read_csv(self.data_file)

        all_init_t, all_init_m, all_init_b = [],[],[]
        all_init_th1, all_init_th2 = [],[]
        all_final_t, all_final_m, all_final_b = [],[],[]
        all_final_th1, all_final_th2 = [],[]

        for row in range(len(df)):
            initial_state = df.iloc[row].to_list()[2:12+2]
            t, m, b, th1, th2 = self.three_layer_model(initial_state)
            all_init_t.append(t)
            all_init_m.append(m)
            all_init_b.append(b)
            all_init_th1.append(th1*2)
            all_init_th2.append(th2*2)

            final_state = df.iloc[row].to_list()[12+2:]
            t, m, b, th1, th2 = self.three_layer_model(final_state)
            all_final_t.append(t)
            all_final_m.append(m)
            all_final_b.append(b)
            all_final_th1.append(th1*2)
            all_final_th2.append(th2*2)

        df['init_t'] = all_init_t
        df['init_m'] = all_init_m
        df['init_b'] = all_init_b
        df['init_th1'] = all_init_th1
        df['init_th2'] = all_init_th2
        df['final_t'] = all_final_t
        df['final_m'] = all_final_m
        df['final_b'] = all_final_b
        df['final_th1'] = all_final_th1
        df['final_th2'] = all_final_th2
        df = df.drop(columns=[x for x in df.columns if 'tank' in x])

        X = df[[c for c in df.columns if 'init' in c or 'store_heat_in' in c]]
        y = df[[c for c in df.columns if 'final' in c]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        self.data_based_model = Ridge(alpha=1.0)
        self.data_based_model.fit(X_train, y_train)
        y_pred_ridge = self.data_based_model.predict(X_test)
        mse_ridge = mean_squared_error(y_test, y_pred_ridge)
        r2_ridge = r2_score(y_test, y_pred_ridge)
        ridge_cv_score = cross_val_score(self.data_based_model, X, y, cv=5, scoring='neg_mean_squared_error')
        print(f"Regressor performance on given data ({self.data_file}):")
        print(f"- RMSE: {round(np.sqrt(abs(mse_ridge)),1)}")
        print(f"- R-squared: {round(r2_ridge,1)}")
        print(f"- Cross-Validation RMSE: {round(np.sqrt(abs(ridge_cv_score.mean())),1)}\n")

    def three_layer_model(self, tank_temps, prints=False):
        if isinstance(tank_temps, list):
            tank_temps = {
                f'tank{tank}-depth{depth}': tank_temps[tank*4 + depth] for tank in range(3) for depth in range(4)
            }

        # Process layer temperatures
        layer_temps = [tank_temps[key] for key in tank_temps]
        iter_count = 0
        if prints: print(f"Before processing: {layer_temps}")
        while (sorted(layer_temps, reverse=True) != layer_temps and iter_count<20):
            iter_count += 1
            layer_temps = []
            for layer in tank_temps:
                if layer_temps:
                    if tank_temps[layer] > layer_temps[-1]:
                        mean = round((layer_temps[-1] + tank_temps[layer]) / 2)
                        layer_temps[-1] = mean
                        layer_temps.append(mean)
                    else:
                        layer_temps.append(tank_temps[layer])
                else:
                    layer_temps.append(tank_temps[layer])
            for i, layer in enumerate(tank_temps):
                tank_temps[layer] = layer_temps[i]
            if iter_count == 20:
                layer_temps = sorted(layer_temps, reverse=True)
        if prints: print(f"After processing: {layer_temps}")

        # Cluster 10 times and select the result with the highest top temperature
        data = layer_temps.copy()
        clustering_runs = []
        for i in range(10):
            labels = self.kmeans(data, k=3)
            cluster_0 = sorted([data[i] for i in range(len(data)) if labels[i] == 0], reverse=True)
            cluster_1 = sorted([data[i] for i in range(len(data)) if labels[i] == 1], reverse=True)
            cluster_2 = sorted([data[i] for i in range(len(data)) if labels[i] == 2], reverse=True)
            cluster_top = max(cluster_0, cluster_1, cluster_2, key=lambda x: np.mean(x) if len(x)>0 else 0)
            top_temp = sum(cluster_top)/len(cluster_top)
            clustering_runs.append({
                'cluster_0': cluster_0,
                'cluster_1': cluster_1,
                'cluster_2': cluster_2,
                'top_temp': top_temp
            })
        best_run = max(clustering_runs, key=lambda x: x['top_temp'])
        cluster_0 = best_run['cluster_0']
        cluster_1 = best_run['cluster_1']
        cluster_2 = best_run['cluster_2']

        # Dealing with 3 clusters
        if cluster_0 and cluster_1 and cluster_2:
            cluster_top = max(cluster_0, cluster_1, cluster_2, key=lambda x: sum(x)/len(x))
            cluster_bottom = min(cluster_0, cluster_1, cluster_2, key=lambda x: sum(x)/len(x))
            cluster_middle = [
                cluster_x for cluster_x in [cluster_0, cluster_1, cluster_2]
                if cluster_x != cluster_top
                and cluster_x != cluster_bottom
                ][0]
            if prints: print(f"{cluster_top}, {cluster_middle}, {cluster_bottom}")

            thermocline1 = max(1, len(cluster_top))
            thermocline2 = thermocline1 + len(cluster_middle)
            if prints: print(f"Thermocline 1: {thermocline1}/12, thermocline 2: {thermocline2}/12")

            top_temp = round(sum(cluster_top)/len(cluster_top))
            middle_temp = round(sum(cluster_middle)/len(cluster_middle))
            bottom_temp = round(sum(cluster_bottom)/len(cluster_bottom))
            if prints: print(f"{top_temp}({thermocline1}){middle_temp}({thermocline2}){bottom_temp}")
            return top_temp, middle_temp, bottom_temp, thermocline1, thermocline2

        # Dealing with less than 3 clusters
        else:
            if cluster_0 and cluster_2:
                cluster_1 = cluster_2
            elif cluster_1 and cluster_2:
                cluster_0 = cluster_2
            # Two clusters
            if cluster_0 and cluster_1:
                cluster_top = max(cluster_0, cluster_1, key=lambda x: sum(x)/len(x))
                cluster_bottom = min(cluster_0, cluster_1, key=lambda x: sum(x)/len(x))
                thermocline1 = len(cluster_top)
                top_temp = round(sum(cluster_top)/len(cluster_top))
                bottom_temp = round(sum(cluster_bottom)/len(cluster_bottom))
                if prints: print(f"{top_temp}({thermocline1}){bottom_temp}")
                return top_temp, top_temp, bottom_temp, thermocline1, thermocline1
            # Single cluster
            else:
                cluster_top = max(cluster_0, cluster_1, cluster_2, key=lambda x: len(x))
                top_temp = round(sum(cluster_top)/len(cluster_top))
                thermocline1 = 12
                if prints: print(f"{top_temp}({thermocline1})")
                return top_temp, top_temp, top_temp, thermocline1, thermocline1
            
    def kmeans(self, data, k=3, max_iters=100, tol=1e-4):
        data = np.array(data).reshape(-1, 1)
        centroids = data[np.random.choice(len(data), k, replace=False)]
        for _ in range(max_iters):
            labels = np.argmin(np.abs(data - centroids.T), axis=1)
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                cluster_points = data[labels == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = cluster_points.mean()
                else:
                    new_centroids[i] = data[np.random.choice(len(data))]
            if np.all(np.abs(new_centroids - centroids) < tol):
                break
            centroids = new_centroids
        return labels


class RuleBasedStorageModel():
    def __init__(self, flo_params: FloParamsHouse0, nodes: List, nodes_by: Dict, allowed_temperatures: Dict):
        self.params = DParams(flo_params)
        self.nodes: List[DNode] = nodes
        self.nodes_by: Dict[Tuple, Dict[Tuple, DNode]] = nodes_by
        self.top_temps = allowed_temperatures['top']
        self.middle_temps = allowed_temperatures['middle']
        self.bottom_temps = allowed_temperatures['bottom']

    def next_node(self, node_now:DNode, store_heat_in:float, print_detail:bool=False) -> DNode:
        if store_heat_in > 0:
            if print_detail: print(f"Charge {node_now} by {store_heat_in}")
            next_node = self.charge(node_now, store_heat_in, print_detail)
        elif store_heat_in < -1:
            if print_detail: print(f"Discharge {node_now} by {-store_heat_in}")
            next_node = self.discharge(node_now, store_heat_in, print_detail)
        else:
            if print_detail: print("IDLE")
            tmb = (node_now.top_temp, node_now.middle_temp, node_now.bottom_temp)
            th = (node_now.thermocline1, node_now.thermocline2)
            next_node = self.nodes_by[tmb][th]
        return next_node
    
    def discharge(self, n: DNode, store_heat_in: float, print_detail: bool) -> DNode:
        next_node_energy = n.energy + store_heat_in
        candidate_nodes: List[DNode] = []
        if print_detail: print(f"Current energy {round(n.energy,2)}, looking for {round(next_node_energy,2)}")
        # Starting from current node
        th1 = n.thermocline1-1
        th2 = n.thermocline2-1

        # Node to discharge is a cold node
        if n.top_temp <= 100:
            if n.top_temp==80 and th1==0:
                return min(self.nodes, key=lambda x: x.energy)
            # Go through the top being at 100 or at 80
            while th1>0:
                if print_detail: print(f"Looking for {n.top_temp}({th1}){n.middle_temp}({th2}){n.bottom_temp}")
                tmb = (n.top_temp, n.middle_temp, n.bottom_temp)
                th = (th1, th2)
                node = self.nodes_by[tmb][th]
                if print_detail: print(f"Energy: {round(node.energy,2)}")
                th1 += -1
                th2 += -1
                candidate_nodes.append(node)
                if next_node_energy >= node.energy:
                    break
            # If the top was at 100 try now 80
            if n.top_temp == 100:
                th1 = self.params.num_layers
                th2 = self.params.num_layers
                while th1>0:
                    if print_detail: print(f"Looking for {n.top_temp-20}({th1}){n.middle_temp-20}({th2}){n.bottom_temp-20}")
                    tmb = (80, 60, 60)
                    th = (th1, th2)
                    node = self.nodes_by[tmb][th]
                    if print_detail: print(f"Energy: {round(node.energy,2)}")
                    th1 += -1
                    th2 += -1
                    candidate_nodes.append(node)
                    if next_node_energy >= node.energy:
                        break
            closest_node = min([x for x in candidate_nodes], key=lambda x: abs(x.energy-next_node_energy))
            return closest_node

        need_to_break = False
        while True:
            # Moving up step by step until the end of the top layer
            while th1>0:
                if print_detail: print(f"Looking for {n.top_temp}({th1}){n.middle_temp}({th2}){n.bottom_temp}")
                tmb = (n.top_temp, n.middle_temp, n.bottom_temp)
                th = (th1, th2)
                node = self.nodes_by[tmb][th]
                if print_detail: print(f"Energy: {round(node.energy,2)}")
                th1 += -1
                th2 += -1
                candidate_nodes.append(node)
                if next_node_energy >= node.energy:
                    need_to_break = True
                    break
            if need_to_break: break

            # There is no middle layer (cold nodes reached)
            if n.middle_temp == n.bottom_temp or (n.bottom_temp==100 and th1==th2):
                # Go through the top being at 100
                top_temp = 100
                middle_temp = 80
                bottom_temp = 80
                th1 = self.params.num_layers
                th2 = self.params.num_layers
                while th1>0:
                    if print_detail: print(f"Looking for {top_temp}({th1}){middle_temp}({th2}){bottom_temp}")
                    tmb = (top_temp, middle_temp, bottom_temp)
                    th = (th1, th2)
                    node = self.nodes_by[tmb][th]
                    if print_detail: print(f"Energy: {round(node.energy,2)}")
                    th1 += -1
                    th2 += -1
                    candidate_nodes.append(node)
                    if next_node_energy >= node.energy:
                        break
                # Go through the top being at 80
                top_temp = 80
                middle_temp = 60
                bottom_temp = 60
                th1 = self.params.num_layers
                th2 = self.params.num_layers
                while th1>0:
                    if print_detail: print(f"Looking for {top_temp}({th1}){middle_temp}({th2}){bottom_temp}")
                    tmb = (top_temp, middle_temp, bottom_temp)
                    th = (th1, th2)
                    node = self.nodes_by[tmb][th]
                    if print_detail: print(f"Energy: {round(node.energy,2)}")
                    th1 += -1
                    th2 += -1
                    candidate_nodes.append(node)
                    if next_node_energy >= node.energy:
                        break
                closest_node = min([x for x in candidate_nodes], key=lambda x: abs(x.energy-next_node_energy))
                return closest_node

            # Moving up step by step until the end of the middle layer
            top_temp = n.middle_temp
            th1 = th2
            while th1>0:
                if print_detail: print(f"Looking for {top_temp}({th1})-({th2}){n.bottom_temp}")
                node = [
                    x for x in self.nodes
                    if x.top_temp==top_temp
                    and x.bottom_temp==n.bottom_temp
                    and x.thermocline1==th1
                    and x.thermocline2==th2
                ][0]
                if print_detail: print(f"Energy: {round(node.energy,2)}")
                th1 += -1
                th2 += -1
                candidate_nodes.append(node)
                if next_node_energy >= node.energy:
                    break
            break

        # Find the candidate node which has closest energy to the target
        closest_node = min([x for x in candidate_nodes], key=lambda x: abs(x.energy-next_node_energy))
        return closest_node
        
    def charge(self, n: DNode, store_heat_in: float, print_detail: bool) -> DNode:
        next_node_energy = n.energy + store_heat_in

        # Charging from cold states
        if n.top_temp<=100:
            next_node_top_temp = n.top_temp
            next_node_bottom_temp = n.bottom_temp
            next_node_thermocline = self.find_thermocline(next_node_top_temp, next_node_bottom_temp, next_node_energy)
            if next_node_thermocline > self.params.num_layers and n.top_temp == 80:
                next_node_top_temp = 100
                next_node_bottom_temp = 80
                next_node_thermocline = self.find_thermocline(next_node_top_temp, next_node_bottom_temp, next_node_energy)
            # Need to rise above cold states
            if next_node_thermocline > self.params.num_layers:
                # At this point we know we have
                next_node_top_temp = 110
                next_node_bottom_temp = 100

        else:
            # If there is a bottom layer
            if n.thermocline2 < self.params.num_layers:
                heated_bottom = n.bottom_temp + self.params.delta_T(n.bottom_temp)
                if print_detail: print(f"heated_bottom = {heated_bottom}")

                # Find the new top temperature after mixing (or not)
                if heated_bottom < n.top_temp:
                    top_and_middle_mixed = (n.top_temp*n.thermocline1 + n.middle_temp*(n.thermocline2-n.thermocline1))/n.thermocline2
                    if print_detail: print(f"top_and_middle_mixed = {top_and_middle_mixed}")
                    top_and_middle_and_heated_bottom_mixed = (
                        (top_and_middle_mixed*n.thermocline2 + heated_bottom*(self.params.num_layers-n.thermocline2))/self.params.num_layers
                        )
                    if print_detail: print(f"top_and_middle_and_heated_bottom_mixed = {round(top_and_middle_and_heated_bottom_mixed,1)}")
                    next_node_top_temp = round(top_and_middle_and_heated_bottom_mixed)
                else:
                    next_node_top_temp = heated_bottom   
                
                # Bottom layer stays the same
                next_node_bottom_temp = n.bottom_temp

            # If there is no bottom layer but there is a middle layer
            elif n.thermocline1 < self.params.num_layers:     
                heated_middle = n.middle_temp + self.params.delta_T(n.middle_temp)
                if print_detail: print(f"heated_middle = {heated_middle}")

                # Find the new top temperature after mixing (or not)
                if heated_middle < n.top_temp:
                    top_and_heated_middle_mixed = (
                        (n.top_temp*n.thermocline1 + heated_middle*(self.params.num_layers-n.thermocline1))/self.params.num_layers
                        )
                    if print_detail: print(f"top_and_heated_middle_mixed = {round(top_and_heated_middle_mixed,1)}")
                    next_node_top_temp = round(top_and_heated_middle_mixed)
                else:
                    next_node_top_temp = heated_middle   

                # Bottom layer is the middle
                next_node_bottom_temp = n.middle_temp

            # If there is only a top layer
            else:
                heated_top = n.top_temp + self.params.delta_T(n.top_temp)
                if print_detail: print(f"heated_top = {heated_top}")
                next_node_top_temp = heated_top   
                # Bottom layer is the top
                next_node_bottom_temp = n.top_temp

        # Starting with that top and current bottom, find the thermocline position that matches the next node energy
        next_node_thermocline = self.find_thermocline(next_node_top_temp, next_node_bottom_temp, next_node_energy)
        if print_detail: print(f"Next node ({next_node_top_temp}, {next_node_bottom_temp}) thermocline: {next_node_thermocline}")
        while next_node_thermocline > self.params.num_layers:
            next_node_bottom_temp = next_node_top_temp
            next_node_top_temp = round(next_node_top_temp + self.params.delta_T(next_node_top_temp))
            next_node_thermocline = self.find_thermocline(next_node_top_temp, next_node_bottom_temp, next_node_energy)
            if print_detail: print(f"Next node ({next_node_top_temp}, {next_node_bottom_temp}) thermocline: {next_node_thermocline}")

        if next_node_top_temp <= 100:
            node_next_true = DNode(
                parameters = self.params,
                top_temp = next_node_top_temp,
                middle_temp = next_node_bottom_temp,
                bottom_temp = next_node_bottom_temp,
                thermocline1 = next_node_thermocline,
                thermocline2 = next_node_thermocline,
            )
        
        else:
            if next_node_bottom_temp != 100:
                if next_node_thermocline > 0:
                    node_next_true = DNode(
                        parameters = self.params,
                        top_temp = next_node_top_temp,
                        middle_temp = next_node_bottom_temp,
                        bottom_temp = n.bottom_temp,
                        thermocline1 = next_node_thermocline,
                        thermocline2 = self.params.num_layers,
                    )
                else:
                    node_next_true = DNode(
                        parameters = self.params,
                        top_temp = next_node_bottom_temp,
                        middle_temp = next_node_bottom_temp-10 if next_node_bottom_temp>=120 else 100,
                        bottom_temp = n.bottom_temp,
                        thermocline1 = self.params.num_layers,
                        thermocline2 = self.params.num_layers,
                    )
            else:
                node_next_true = DNode(
                    parameters = self.params,
                    top_temp = next_node_top_temp,
                    middle_temp = next_node_top_temp,
                    bottom_temp = next_node_bottom_temp,
                    thermocline1 = next_node_thermocline,
                    thermocline2 = next_node_thermocline
                )

        return node_next_true
        
    def find_thermocline(self, top_temp, bottom_temp, energy):
        top, bottom = to_kelvin(top_temp), to_kelvin(bottom_temp)
        m_layer_kg = self.params.storage_volume*3.785 / self.params.num_layers    
        if top==bottom: top+=1  
        return int(1/(top-bottom)*(energy/(m_layer_kg*4.187/3600)-(-0.5*top+(self.params.num_layers+0.5)*bottom)))


if __name__ == '__main__':

    from models import MessageSql
    with open('messages.json', 'r') as file:
        data = json.load(file)
        message = MessageSql(
            message_id=data["MessageId"],
            from_alias=data["FromAlias"],
            message_type_name=data["MessageTypeName"],
            message_persisted_ms=data["MessagePersistedMs"],
            payload=data["Payload"],
            message_created_ms=data.get("MessageCreatedMs")
        )
    flo_params = FloParamsHouse0(**message.payload)

    generator = SuperGraphGenerator(flo_params)
    generator.generate()