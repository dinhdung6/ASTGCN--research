import csv
import plotly.graph_objects as go

# Define the file path
csv_file = r'C:\Users\bong\Downloads\COS30018-project--main\COS30018-project--main\experiments\PEMS04\astgcn_r_h1d0w0_channel1_1.000000e-02\epoch_metrics.csv'

# Initialize lists to store the data
epochs = []
mae_values = []
rmse_values = []
mape_values = []

# Read the CSV file
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Skip the header row
    for row in reader:
        epochs.append(int(row[0]))  # Epoch
        mae_values.append(float(row[1]))  # MAE
        rmse_values.append(float(row[2]))  # RMSE
        mape_values.append(float(row[3]))  # MAPE

# Extract the "all" values from the last row
all_mae = mae_values[-1]
all_rmse = rmse_values[-1]
all_mape = mape_values[-1]

# Create the plot
fig = go.Figure()

# Add MAE trace
fig.add_trace(go.Scatter(
    x=epochs, 
    y=mae_values, 
    mode='lines+markers', 
    name='MAE',
    marker=dict(size=8),
    hovertemplate='Epoch: %{x}<br>MAE: %{y}<extra></extra>'
))

# Add RMSE trace
fig.add_trace(go.Scatter(
    x=epochs, 
    y=rmse_values, 
    mode='lines+markers', 
    name='RMSE',
    marker=dict(size=8),
    hovertemplate='Epoch: %{x}<br>RMSE: %{y}<extra></extra>'
))

# Add MAPE trace
fig.add_trace(go.Scatter(
    x=epochs, 
    y=mape_values, 
    mode='lines+markers', 
    name='MAPE',
    marker=dict(size=8),
    hovertemplate='Epoch: %{x}<br>MAPE: %{y}<extra></extra>'
))

# Add annotations for "all" metrics
fig.add_annotation(
    x=0.95, y=0.95,  # Place in the top-right corner
    xref='paper', yref='paper',  # Relative coordinates
    text=f"<b>All Metrics:</b><br>MAE: {all_mae:.2f}<br>RMSE: {all_rmse:.2f}<br>MAPE: {all_mape:.2f}",
    showarrow=False,
    align='left',
    font=dict(size=12),
    bgcolor="rgba(255, 255, 255, 0.8)",
    bordercolor="black"
)

# Update the layout
fig.update_layout(
    title='RMSE, MAE, and MAPE over Epochs',
    xaxis_title='Epochs',
    yaxis_title='Values',
    legend_title='Metrics',
    hovermode='x unified',
    template='plotly_white',
    width=900,
    height=600
)

# Show the plot
fig.show()
