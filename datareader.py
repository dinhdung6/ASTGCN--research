def read_params_file(file_path):
    """Reads a .params file and returns a dictionary of key-value pairs."""
    params = {}
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Strip any leading/trailing whitespace and ignore empty lines
                line = line.strip()
                if line and '=' in line:
                    # Split key and value by the '=' sign
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Try to convert value to appropriate type (int, float, or leave as string)
                    if value.isdigit():
                        value = int(value)
                    else:
                        try:
                            value = float(value)
                        except ValueError:
                            pass  # Keep as string if not a number
                    
                    # Add key-value pair to dictionary
                    params[key] = value
        return params
    
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return None
    except Exception as e:
        print(f"Error reading the file {file_path}: {e}")
        return None

def write_params_to_txt(params, output_file):
    """Writes the parameters dictionary to a text file."""
    try:
        with open(output_file, 'w') as file:
            for key, value in params.items():
                file.write(f"{key} = {value}\n")
        print(f"Parameters successfully written to {output_file}")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")

if __name__ == "__main__":
    # Path to your .params file
    input_file = "epoch_0.params"
    
    # Path to output text file
    output_txt_file = "params_output.txt"
    
    # Read the parameters from the .params file
    params = read_params_file(input_file)
    
    if params:
        # Output to console or write to another file
        print("Parameters from the file:")
        for key, value in params.items():
            print(f"{key}: {value}")
        
        # Write parameters to a text file
        write_params_to_txt(params, output_txt_file)
