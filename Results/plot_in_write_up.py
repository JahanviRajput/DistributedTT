### time v/s nbb section fed protes

def nbb_plot(file_name):

    import pandas as pd
    import re

    # Read the text file
    with open(file_name, 'r') as file:
        data = file.read()
    # Define the pattern to match the required sections
    header_pattern = r'fun P-(\d+) \| nbb (\d+) \|'
    detail_pattern = r'm (\d+) \| t ([\d\.]+) \| y ([\d\.]+) \|'

    # Extract the headers first
    header_matches = re.findall(header_pattern, data)
    rows = {}

    # Loop over the headers and extract details for each
    for header_match in header_matches:
        fun_value = f"P-{header_match[0]}"
        nbb_value = int(header_match[1])
        
        # Find the corresponding details section for this header
        header_section_pattern = f'fun P-{header_match[0]} \| nbb {header_match[1]} \|((?:\s*m (\d+) \| t ([\d\.]+) \| y ([\d\.]+) \|)*)'
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
    df.to_csv("log_fed_nbb.csv")

    # Print the DataFrame to verify
    print(df)

# nbb_plot('log_fed_nbb.txt')

def plots_nbb(file_name):
    import pandas as pd
    import matplotlib.pyplot as plt

    # Read the CSV data into a DataFrame
    data = pd.read_csv(file_name)

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Plot 'nbb' vs 'm' for each 'fun' value
    for fun in data['fun'].unique():
        if fun in ['F-04', 'F-03', 'F-13', 'F-06','F-12']:
            subset = data[data['fun'] == fun]
            # removed P01 and P04
            plt.plot(subset['nbb'], subset['t'], marker='o', linestyle='-', label=fun, markersize = 8, linewidth = 4)

    # Add labels and title
    plt.xlabel('Number of Black Boxes', fontsize = 14)
    plt.ylabel('Time taken (in sec)', fontsize = 14)
    # plt.title('Plot of')
    plt.legend(title='Functions', fontsize = 14, title_fontsize=14)
    plt.grid(True)
    plt.savefig("../Results_new_fun/Fig/fed_nbb.pdf")
    # Show the plot

    plt.show()

plots_nbb('../Results_new_fun/log_fed_nbb_new_fun_21_34.csv')