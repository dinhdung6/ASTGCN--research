import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.widgets import Slider

def load_npz_data(npz_path):
    data = np.load(npz_path)
    traffic_data = data['data']
    return traffic_data

def load_distance_data(distance_csv_path):
    distance_df = pd.read_csv(distance_csv_path)
    return distance_df

def extract_node_ids(distance_df):
    node_ids_csv = set(distance_df['from']).union(set(distance_df['to']))
    return node_ids_csv

def check_node_count(node_ids_csv, traffic_data):
    num_nodes_npz = traffic_data.shape[1]
    if len(node_ids_csv) != num_nodes_npz:
        print(f"The number of nodes in the CSV ({len(node_ids_csv)}) and NPZ ({num_nodes_npz}) files do not match.")
    return num_nodes_npz

def create_node_mappings(node_ids_csv):
    node_id_to_index = {node_id: index for index, node_id in enumerate(sorted(node_ids_csv))}
    index_to_node_id = {index: node_id for node_id, index in node_id_to_index.items()}
    return node_id_to_index, index_to_node_id

def create_graph(distance_df):
    G = nx.Graph()
    for index, row in distance_df.iterrows():
        G.add_node(int(row['from']))
        G.add_node(int(row['to']))
        G.add_edge(int(row['from']), int(row['to']), weight=row['cost'])
    return G

def read_traffic_flow_data(use_predicted):
    with open('npz_content_full.txt', 'r') as file:
        lines = file.readlines()[1:308]  # Read lines 2 to 308
        traffic_flow = np.array([list(map(float, line.split())) for line in lines])
    return traffic_flow

def update_hover(event, ax, G, pos, node_id_to_index, traffic_data, time_slice, use_predicted):
    if event.inaxes == ax:
        trans = ax.transData.inverted()
        x, y = trans.transform((event.x, event.y))
        for node in G.nodes:
            node_x, node_y = pos[node]
            node_disp = ax.transData.transform((node_x, node_y))
            if abs(node_disp[0] - event.x) < 10 and abs(node_disp[1] - event.y) < 10:
                try:
                    node_index = node_id_to_index[int(node)]
                    if use_predicted:
                        traffic_flow = read_traffic_flow_data(use_predicted)
                        traffic_flow_value = traffic_flow[node_index, time_slice]
                    else:
                        traffic_flow_value = traffic_data[time_slice, node_index, 0]
                    avg_speed = traffic_data[time_slice, node_index, 1]
                    avg_occupancy = traffic_data[time_slice, node_index, 2]
                    text.set_text(f"Node: {node}\nTraffic Flow: {traffic_flow_value}\nAvg Speed: {avg_speed}\nAvg Occupancy: {avg_occupancy}")
                    plt.draw()
                    return
                except IndexError:
                    text.set_text(f"Node: {node}\nData not available")
                    plt.draw()
                    return
        text.set_text("")
        plt.draw()

def draw_graph(G, pos, traffic_data, node_id_to_index, use_predicted):
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.2)
    ax.set_facecolor('black')
    nx.draw(G, pos, with_labels=False, node_size=50, node_color='red', edge_color='green')
    node_labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_color='black', font_size=8, verticalalignment='bottom', horizontalalignment='right')
    
    global time_slice
    time_slice = 0
    
    fig.canvas.mpl_connect("motion_notify_event", lambda event: update_hover(event, ax, G, pos, node_id_to_index, traffic_data, time_slice, use_predicted))
    
    ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    if use_predicted:
        slider = Slider(ax_slider, 'Time Slice', 0, 11, valinit=0, valstep=1)  # 12 time slices
    else:
        slider = Slider(ax_slider, 'Time Slice', 0, traffic_data.shape[0] - 1, valinit=0, valstep=1)
    
    slider.on_changed(lambda val: update(val))
    
    global text
    text = ax.text(0.05, 0.95, "", transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=1.0))
    
    fig.canvas.manager.set_window_title('Traffic Flow Data Visualization')
    
    plt.title('Traffic Network Visualization', color='white')
    plt.show()

def update(val):
    global time_slice
    time_slice = int(val)
    plt.draw()

def main():
    data_npz_path = '../data/PEMS04/pems04.npz'
    distance_csv_path = '../data/PEMS04/distance.csv'
    
    traffic_data = load_npz_data(data_npz_path)
    distance_df = load_distance_data(distance_csv_path)
    
    node_ids_csv = extract_node_ids(distance_df)
    # print(f"Number of nodes in the csv file: {len(node_ids_csv)}")
    
    num_nodes_npz = check_node_count(node_ids_csv, traffic_data)
    # print(f"Number of nodes in the npz file: {num_nodes_npz}")
    
    use_predicted = input("Use predicted traffic flow data from npz_content_full.txt? (yes/no): ").strip().lower() in ['yes', 'y']
    
    node_id_to_index, index_to_node_id = create_node_mappings(node_ids_csv)
    
    G = create_graph(distance_df)
    pos = nx.spring_layout(G)
    
    draw_graph(G, pos, traffic_data, node_id_to_index, use_predicted)

if __name__ == '__main__':
    main()