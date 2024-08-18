# Read the content of the .loc file and filter for specific channels
input_file_path = 'cap19.loc'
output_file_path = 'filtered_channels.loc'

# List of channels to keep
channels_to_keep = ['F8', 'F4', 'F3', 'F7']

# Read the original .loc file and filter the lines
with open(input_file_path, 'r') as file:
    lines = file.readlines()

# Write the filtered content to a new .loc file
with open(output_file_path, 'w') as file:
    for line in lines:
        if any(channel in line for channel in channels_to_keep):
            file.write(line)

output_file_path
