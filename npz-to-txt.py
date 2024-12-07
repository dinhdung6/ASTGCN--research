import numpy as np

def npz_to_txt(npz_filepath, txt_filepath):
    # Load the .npz file
    data = np.load(npz_filepath)
    
    # Open the .txt file for writing
    with open(txt_filepath, 'w') as txt_file:
        for array_name in data.files:
            # Write array name as a header
            txt_file.write(f"Array: {array_name}\n")
            txt_file.write(f"{data[array_name]}\n\n")

    print(f"Data saved to {txt_filepath}")

# Example usage
npz_filepath = 'C:\Users\bong\Downloads\COS30018-project--main\COS30018-project--main\data\PEMS04\pems04.npz'  # Replace with your actual .npz file path
txt_filepath = 'C:\Users\bong\Downloads\COS30018-project--main\COS30018-project--main'  # Replace with your desired .txt file path
npz_to_txt(npz_filepath, txt_filepath)
