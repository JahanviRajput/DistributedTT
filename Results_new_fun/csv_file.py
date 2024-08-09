import pandas as pd
import re

# Read the file content
with open('log_ss_protes_fl.txt', 'r') as file:
    data = file.read()

# Regular expression pattern to match each function block
pattern = r'F-\d{2} \| d \d+\s+(.*?)(?=\n\s*F-\d{2} \| d \d+|\Z)'

# Find all function blocks
matches = re.findall(pattern, data, re.DOTALL)

# Initialize an empty DataFrame
df = pd.DataFrame(columns=['F-XX', 'm', 't', 'y'])

# Process each match
for idx, match in enumerate(matches):
    F_XX = f"F-{idx+1:02d}"
    # Find all 'm', 't', and 'y' values in the block
    values = re.findall(r'm (\d+) \| t ([+-]?[0-9]*\.?[0-9]+) \| y ([+-]?[0-9]*\.?[0-9]+)', match)
    if values:
        # Get the last set of values
        m_value, t_value, y_value = values[-1]
        df = df.append({'F-XX': F_XX, 'm': int(m_value), 't': float(t_value), 'y': float(y_value)}, ignore_index=True)

# Display the DataFrame
df.to_csv("log_ss_protes.csv")




















































# import pandas as pd
# import re

# # Define the path to the text file and the output CSV file
# input_file_path = 'log_protes_new_fun_21_34.txt'
# output_csv_path = 'y_values.csv'

# # Initialize a dictionary to store the data
# data = {}

# # Define a regex pattern to match the lines
# pattern = re.compile(r'(P-\d+):\s+(Noisy_protes|BS-\d+)\s+>\s+m\s+([\d\.e\+\-]+)\s+\|\s+t\s+([\d\.e\+\-]+)\s+\|\s+y\s+([\d\.e\+\-]+)\s+<<<\s+DONE')

# # Read the file and process each line
# with open(input_file_path, 'r') as file:
#     for line in file:
#         match = pattern.match(line.strip())
#         if match:
#             p, bs, m, t, y = match.groups()
#             if p not in data:
#                 data[p] = {}
#             data[p][bs] = {'y': float(y)}
#             # data[p][bs] = {'t': float(t)}

# # # Create the DataFrame from the dictionary
# columns = ['FED PROTES'] + [f'BS-{i}' for i in range(8)] +['Noisy_protes'] + ['Noisy_def_protes']
# df = pd.DataFrame(index=data.keys(), columns=columns)

# for p in data:
#     for bs in data[p]:
#         df.at[p, bs] = data[p][bs]['y']  # Use 'y' value as specified

# print(f'Data successfully written to {output_csv_path}')

# import pandas as pd
# import re

# # Function to extract y and time values from the text
# def extract_values_from_text(text):
#     pattern = r'(P-\d+): y opt =\s+([\d\.e\+\-]+) \| time =\s+([\d\.e\+\-]+)'
#     matches = re.findall(pattern, text)
#     return matches

# # Function to read the text file and return the extracted values as a DataFrame
# def process_text_file(text_file_path):
#     with open(text_file_path, 'r') as file:
#         text = file.read()

#     values = extract_values_from_text(text)
#     columns = ['fun', 'y_opt', 'time']

#     # Create a DataFrame from the extracted values
#     df = pd.DataFrame(values, columns=columns)

#     # Convert the columns to appropriate data types
#     df['y_opt'] = df['y_opt'].astype(float)
#     df['time'] = df['time'].astype(float)

#     return df

# new_df = process_text_file('log_fed_new_fun_21_34.txt')

# df['FED PROTES'] = list(new_df['y_opt'])
# # df['FED PROTES'] = list(new_df['time'])
# # Save the DataFrame to a CSV file



# import pandas as pd
# import re

# def extract_data_from_text(file_path, start_index=21, end_index=34):
#     """
#     Extracts y opt and time values from a text file for indices from P-start_index to P-end_index.
    
#     Args:
#         file_path (str): Path to the text file.
#         start_index (int): Starting index (default is 21).
#         end_index (int): Ending index (default is 34).
    
#     Returns:
#         pd.DataFrame: DataFrame containing the indices, y opt, and time values.
#     """
#     # Initialize lists to store the data
#     indices = []
#     y_opts = []
#     times = []
    
#     # Define the pattern for extraction
#     pattern = re.compile(r'P-(\d+):\s*y opt\s*=\s*([\d\.\-e+]+)\s*\|\s*time\s*=\s*([\d\.]+)')
    
#     # Read the content of the text file
#     with open(file_path, 'r') as file:
#         for line in file:
#             match = pattern.search(line)
#             if match:
#                 index = int(match.group(1))
#                 if start_index <= index <= end_index:
#                     y_opt = float(match.group(2))
#                     time = float(match.group(3))
                    
#                     # Append data
#                     indices.append(f'P-{index}')
#                     y_opts.append(y_opt)
#                     times.append(time)
    
#     # Create DataFrame
#     df = pd.DataFrame({
#         'Index': indices,
#         'y opt': y_opts,
#         'time': times
#     })
    
#     return df

# # Example usage
# file_path = 'log_noisy_functions.txt'
# df1 = extract_data_from_text(file_path)
# # df['Noisy_def_protes'] = list(df1['time'])
# df['Noisy_def_protes'] = list(df1['y opt'])

# # df.to_csv(output_csv_path)