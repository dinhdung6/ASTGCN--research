import matplotlib.pyplot as plt

def read_data(file_path):
    with open(file_path, 'r') as f:
        data = f.read().strip()
    # Remove the square brackets and split by commas
    data = data[1:-1].split(',')
    # Convert to floats
    data = [float(x) for x in data]
    return data

# Read data from the ASTGCN model
data_astgcn = read_data('mae_and_rmse_astgcn.txt')
print(f"ASTGCN data length: {len(data_astgcn)}")

# Extract MAE and RMSE values for ASTGCN
mae_values_astgcn = data_astgcn[0::3]
rmse_values_astgcn = data_astgcn[1::3]

# Read data from the DSTAGNN model
data_dstagnn = read_data('mae_and_rmse_dstagnn.txt')
print(f"DSTAGNN data length: {len(data_dstagnn)}")

# Extract MAE and RMSE values for DSTAGNN
mae_values_dstagnn = data_dstagnn[0::3]
rmse_values_dstagnn = data_dstagnn[1::3]

# Read data from the ASTGNN model
data_astgnn = read_data('mae_and_rmse_astgnn.txt')
print(f"ASTGNN data length: {len(data_astgnn)}")

# Extract MAE and RMSE values for ASTGNN
mae_values_astgnn = data_astgnn[0::3]
rmse_values_astgnn = data_astgnn[1::3]

# Generate the range of prediction points
predict_points_astgcn = list(range(len(mae_values_astgcn)))
predict_points_dstagnn = list(range(len(mae_values_dstagnn)))
predict_points_astgnn = list(range(len(mae_values_astgnn)))

# Plot RMSE values
plt.figure(1)
plt.plot(predict_points_astgcn, rmse_values_astgcn, marker='o', color='b', label='ASTGCN')
plt.plot(predict_points_dstagnn, rmse_values_dstagnn, marker='o', color='r', label='DSTAGNN')
plt.plot(predict_points_astgnn, rmse_values_astgnn, marker='o', color='g', label='ASTGNN')
plt.xlabel('Prediction Points')
plt.ylabel('RMSE')
plt.title('RMSE over Prediction Points')
plt.legend()
plt.grid(True)

# Plot MAE values
plt.figure(2)
plt.plot(predict_points_astgcn, mae_values_astgcn, marker='o', color='b', label='ASTGCN')
plt.plot(predict_points_dstagnn, mae_values_dstagnn, marker='o', color='r', label='DSTAGNN')
plt.plot(predict_points_astgnn, mae_values_astgnn, marker='o', color='g', label='ASTGNN')
plt.xlabel('Prediction Points')
plt.ylabel('MAE')
plt.title('MAE over Prediction Points')
plt.legend()
plt.grid(True)

# Show both figures
plt.show(block=False)

# Keep the plots open
input("Press Enter to close the plots...")