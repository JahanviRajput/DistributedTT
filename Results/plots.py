import pandas as pd
import matplotlib.pyplot as plt
import re

def ss_plot(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Initialize dictionary to store data for each function
    data = {}

    # Current function being processed
    current_fun = None

    # Extract values from the text lines
    for line in lines:
        line = line.strip()  # Strip leading/trailing whitespace
        # Check if the line contains the function identifier
        fun_match = re.match(r'fun (F-\d+)', line)
        if fun_match:
            current_fun = fun_match.group(1)
            if current_fun not in data:
                data[current_fun] = {'m': [], 't': [], 'y' : []}
        elif current_fun and line.startswith('m '):  # Remove the leading space
            m_match = re.search(r'm (\d+)', line)
            t_match = re.search(r't ([\d\.E+-]+)', line)  # Handle scientific notation
            y_match = re.search(r'y ([\d\.E+-]+)', line)  # Handle scientific notation
            if m_match and t_match:
                data[current_fun]['m'].append(int(m_match.group(1)))
                data[current_fun]['t'].append(float(t_match.group(1)))
                data[current_fun]['y'].append(float(y_match.group(1)))


    print(data)
    # Plot the m and t values for each function
    plt.figure(figsize=(10, 6))

    for fun, values in data.items():
        df = pd.DataFrame(values)
        print(df)
    #     plt.plot(df['t'], df['y'], marker='o', label=fun)

    # plt.xlabel('m')
    # plt.ylabel('t')
    # plt.title('Plot of m vs. t for Multiple Functions')
    # plt.legend(title="PROTES")
    # plt.grid(True)
    # # plt.savefig("Fig/SS_protes.pdf")
    # plt.show()

# Example usage
ss_plot('../Results_new_fun/log_ss_protes_fl.txt')

### nbb plots


def nbb_plot(file_name):

    import pandas as pd
    import re

    # Read the text file
    with open(file_name, 'r') as file:
        data = file.read()
    # Define the pattern to match the required sections
    header_pattern = r'fun F-(\d+) \| nbb (\d+) \|'
    detail_pattern = r'm (\d+) \| t ([\d\.]+) \| y ([\d\.]+) \|'

    # Extract the headers first
    header_matches = re.findall(header_pattern, data)
    rows = {}

    # Loop over the headers and extract details for each
    for header_match in header_matches:
        fun_value = f"F-{header_match[0]}"
        nbb_value = int(header_match[1])
        
        # Find the corresponding details section for this header
        header_section_pattern = f'fun F-{header_match[0]} \| nbb {header_match[1]} \|((?:\s*m (\d+) \| t ([\d\.]+) \| y ([\d\.]+) \|)*)'
        header_section_match = re.search(header_section_pattern, data)
        
        if header_section_match:
            details_section = header_section_match.group(1)
            
            # Find all m, t, y values within the details section
            detail_matches = re.findall(detail_pattern, details_section)
            
            for detail_match in detail_matches:
                m_value = int(detail_match[0])
                t_value = float(detail_match[1])
                y_value = float(detail_match[2])
                # Use a tuple of fun and nbb as the key to store the last value
                rows[(fun_value, nbb_value)] = [m_value, t_value, y_value]

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(rows, orient='index', columns=['m', 't', 'y'])
    df.index = pd.MultiIndex.from_tuples(df.index, names=['fun', 'nbb'])
    file = '../Results_new_fun/log_fed_nbb_new_fun_21_34.csv'
    df.to_csv(file)

    # Print the DataFrame to verify
    print(df)

# 
# nbb_plot('../Results_new_fun/log_fed_nbb_new_fun_21_34.txt')

def plots_nbb(file_name):
    import pandas as pd
    import matplotlib.pyplot as plt

    # Read the CSV data into a DataFrame
    data = pd.read_csv(file_name)

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Plot 'nbb' vs 'm' for each 'fun' value
    for fun in data['fun'].unique():
        subset = data[data['fun'] == fun]
        # removed P01 and P04
        plt.plot(subset['nbb'], subset['t'], marker='o', linestyle='-', label=fun, markersize = 6, linewidth = 2)

    # Add labels and title
    plt.xlabel('Number of Black Boxes', fontsize = 14)
    plt.ylabel('Time taken (in sec)', fontsize = 14)
    # plt.title('Plot of')
    plt.legend(title='Functions', fontsize = 14, title_fontsize=14)
    plt.grid(True)
    plt.savefig("Fig/fed_nbb_t.pdf")
    # Show the plot

    plt.show()
#
# plots_nbb('log_fed_nbb.csv')

## baselines df

# import pandas as pd
# import re

def extract_data(input_file, y=True):
    # Read the file content
    with open(input_file, 'r') as file:
        content = file.read()

    # Initialize a dictionary to store data for each P-XX index
    data = {f'P-{str(i).zfill(2)}': {} for i in range(1, 15)}
    
    # Regular expression to extract relevant data
    pattern = r'(P-\d{2}):\s+([^\>]+)\s+>\s+m\s+([\d.e+-]+)\s+\|\s+t\s+([\d.e+-]+)\s+\|\s+y\s+([\d.e+-]+)\s+<<<\s+DONE'
    matches = re.findall(pattern, content)
    
    # Populate the dictionary with extracted data
    for match in matches:
        index, bs_line, m_value, t_value, y_value = match
        if index not in data:
            continue
        if y:
            if bs_line.strip() not in data[index]:
                data[index][bs_line.strip()] = float(y_value)
        else:
            if bs_line.strip() not in data[index]:
                data[index][bs_line.strip()] = float(t_value)
    
    # Convert the dictionary to a DataFrame
    df_dict = {}
    for index, bs_lines in data.items():
        df_dict[index] = pd.Series(bs_lines)
    
    df = pd.DataFrame(df_dict).T
    if y:
        df.to_csv("protes_baselines_y_data.csv")
    else:
        df.to_csv("protes_baselines_t_data.csv")
    
    return df.T

# extract_data('log_protes_baselines.txt', False)




## making format of text correct 
def nbb_format(input_file, output_file):
    import re
    with open(input_file, 'r') as file:
        content = file.read()
    
    # Find all blocks of interest in the text
    blocks = re.findall(r'(Number of black boxes: \d+)\s+(.*?)(?=\nNumber of black boxes:|\Z)', content, re.DOTALL)
    
    with open(output_file, 'w') as file:
        for header, values in blocks:
            file.write(f"{header}\n")
            file.write(f"{values.strip()}\n\n")

# nbb_format('log_fed_nbb.txt', 'log_fed_nbb.txt')



### plot from text file read and make a data frame
def text_df(file_name, saved_file_name):
    import pandas as pd

    # Read the file
    with open(file_name, 'r') as file:
        lines = file.readlines()

    # Initialize a dictionary to store the data for each P-ID
    data_dict = {}
    current_p = None
    m_values, t_values, y_values = [], [], []

    # Parse the file content
    for line in lines:
        line = line.strip()
        if line.startswith('P-'):
            if current_p:
                data_dict[current_p] = {
                    'm': m_values,
                    't': t_values,
                    'y': y_values
                }
            current_p = line.split(':')[0]
            m_values, t_values, y_values = [], [], []
        elif line.startswith('m '):
            parts = line.split('|')
            m_values.append(int(parts[0].split()[1]))
            t_values.append(float(parts[1].split()[1]))
            y_values.append(float(parts[2].split()[1]))

    # Add the last P-ID data to the dictionary
    if current_p:
        data_dict[current_p] = {
            'm': m_values,
            't': t_values,
            'y': y_values
        }

    # Create the final DataFrame
    df = pd.DataFrame()

    for p_id, values in data_dict.items():
        p_df = pd.DataFrame({
            'Fun': [p_id] * len(values['m']),
            'm': values['m'],
            't': values['t'],
            'y': values['y']
        })
        if df.empty:
            df = p_df
        else:
            df = pd.concat([df, p_df], axis=0)

    # Reset the index to have a clean DataFrame
    df.set_index(df.columns[0], inplace = True)
    df.to_csv(saved_file_name)

# text_df('../Results_new_fun/log_fed_new_fun_21_34.txt', "../Results_new_fun/log_fed_new_fun_21_34.csv")

def plot(file_path):
    import pandas as pd
    import matplotlib.pyplot as plt
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Initialize a plot
    plt.figure()

    # Loop through P-01 to P-10 and plot each one
    for i in range(1, 11):
        group = f'P-{i:02d}'
        df_group = df[df['Fun'] == group]
        plt.plot(df_group['m'], df_group['y'], label=group)

    # Add labels and title
    plt.xlabel('m')
    plt.ylabel('y')
    plt.title('Plot of m vs y for P-01 to P-10')
    plt.legend()

    # Save the plot
    plt.savefig("Fig/fed_m_y.pdf")

    # Show the plot
    plt.show()

# plot('log_fed_m_y_t.csv')

