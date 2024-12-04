import tkinter as tk
from subprocess import Popen, PIPE, STDOUT, CREATE_NEW_PROCESS_GROUP
import os
import threading
import signal

subprocesses = []  # List to keep track of subprocesses

# Function to asynchronously update the console in the GUI
def update_console(process, text_widget):
    for line in iter(process.stdout.readline, ''):
        text_widget.insert(tk.END, line)
        text_widget.see(tk.END)
    process.stdout.close()

def run_script(script_name, config_file, text_widget):
    command = f"python -u {script_name} --config {config_file}"
    proc = Popen(command, shell=True, stdout=PIPE, stderr=STDOUT, text=True, bufsize=1, creationflags=CREATE_NEW_PROCESS_GROUP)
    subprocesses.append(proc)  # Add the process to the list
    threading.Thread(target=update_console, args=(proc, text_widget), daemon=True).start()
    return proc

# Function to handle Data Preparation
def prepare_data():
    dataset = dataset_var.get()
    config_file = f"configurations/{dataset}_astgcn.conf"
    console_output.insert(tk.END, f"Preparing data for {dataset} dataset...\n")
    prepare_button.config(state=tk.DISABLED)  # Disable the Prepare Data button
    
    # Run the prepareData script and wait for it to complete before enabling the training button
    process = run_script("prepareData.py", config_file, console_output)
    process.wait()  # Wait for the data preparation to finish
    train_button.config(state=tk.NORMAL)  # Enable the training button after data is prepared
    prepare_button.config(state=tk.NORMAL)  # Re-enable the Prepare Data button

# Function to handle Start Button for training
def start_training():
    dataset = dataset_var.get()
    compute = compute_var.get()
    config_file = f"configurations/{dataset}_astgcn.conf"
    os.environ['DEVICE_PREFERENCE'] = compute
    global train_process
    train_process = run_script("train_ASTGCN_r.py", config_file, console_output)
    console_output.insert(tk.END, f"Started training on {compute} using {dataset} dataset\n")
    prepare_button.config(state=tk.DISABLED)  # Disable the Prepare Data button during training
    train_button.config(state=tk.DISABLED)  # Disable the Start Training button during training

def stop_training():
    # Terminate all subprocesses
    while subprocesses:
        process = subprocesses.pop()
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)  # Send CTRL_BREAK_EVENT signal to the process group
            process.wait(timeout=5)  # Wait for the process to terminate
            console_output.insert(tk.END, "Training interrupted by user.\n")
        except Exception as e:
            console_output.insert(tk.END, f"Error terminating process: {e}\n")
    prepare_button.config(state=tk.NORMAL)  # Re-enable the Prepare Data button after training is stopped
    train_button.config(state=tk.NORMAL)  # Re-enable the Start Training button after training is stopped

# Create main window
root = tk.Tk()
root.title("ASTGCN Training Interface")
root.geometry('800x600')  # Set the window size

# Variables for user selection
dataset_var = tk.StringVar(value="PEMS04")
compute_var = tk.StringVar(value="CPU")

# Dataset selection
tk.Label(root, text="Select Dataset:").pack()
tk.OptionMenu(root, dataset_var, "PEMS04", "PEMS08").pack()

# Compute selection
tk.Label(root, text="Select Compute Option:").pack()
tk.Radiobutton(root, text="GPU", variable=compute_var, value="GPU").pack()
tk.Radiobutton(root, text="CPU", variable=compute_var, value="CPU").pack()

# Prepare Data and Start Training buttons
prepare_button = tk.Button(root, text="Prepare Data", command=prepare_data)
prepare_button.pack()

train_button = tk.Button(root, text="Start Training", command=start_training, state=tk.DISABLED)
train_button.pack()

# Stop button
tk.Button(root, text="Stop Training", command=stop_training).pack()

# Console Output
console_output = tk.Text(root, height=20, width=100, font=('Courier', 10))
console_output.pack()

# Start the GUI event loop
root.mainloop()
