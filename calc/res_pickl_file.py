def create_pickel():
    import pandas as pd
    import pickle

    # Read the CSV file
    # csv_file = '../Results/log_fed_m_y_t.csv'
    csv_file = '../Results_new_fun/log_fed_new_fun_21_34.csv'
    df = pd.read_csv(csv_file)

    # Initialize the dictionary to store the data
    data_dict = {}

    # Extract data for each unique 'Fun' value
    for fun in df['Fun'].unique():
        subset = df[df['Fun'] == fun]
        m_values = subset['m'].tolist()
        y_values = subset['y'].tolist()
        y = [y_values[-1]]
        data_dict[fun] = {'fed': [m_values, y_values, y]}

    # Save the dictionary to a pickle file
    pkl_file = '../Results_new_fun/fed_res.pkl'
    with open(pkl_file, 'wb') as f:
        pickle.dump(data_dict, f)

    print(f'Data saved to {pkl_file}')
    # Load the pickle file
    with open(pkl_file, 'rb') as file:
        data = pickle.load(file)
    # print(data)

create_pickel()


def plot_m_y():    
    import pickle
    import matplotlib.pyplot as plt
    # Path to your pickle file
    file_path = '../Results_new_fun/combine_res_m_y.pkl'

    # Load the pickle file
    with open(file_path, 'rb') as file:
        data = pickle.load(file)


    # Extract data for P-01
    p01_data = data['P-01']

    # Initialize the plot
    plt.figure()

    # Plot data for each key in P-01
    for key in p01_data:
        entry = p01_data[key]
        m = entry[0]
        y = entry[1]

        plt.plot(m, y, marker='o', label=key)


    # Set plot labels and title
    plt.xlabel('m')
    plt.ylabel('y')
    plt.title('P-01 Data Plot')
    plt.legend()
    plt.grid(True)
    plt.savefig("../Results_new_fun/M_Y_plot.pdf")
    # Show the plot
    plt.show()

# plot_m_y()

import pickle
import matplotlib.pyplot as plt

def plot_combined_m_y():
    # Paths to your pickle files
    file_paths = ['res.pickle', '../Results_new_fun/fed_res.pkl']

    # Initialize a dictionary to hold combined data
    combined_data = {}

    # Load the pickle files and combine the data
    for file_path in file_paths:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            for key in data:
                if key not in combined_data:
                    combined_data[key] = data[key]
                else:
                    if key != 'Noisy_protes':
                        combined_data[key].update(data[key])

    # Extract data for P-01
    l = list(data.keys())
    for i in l:
        p01_data = combined_data.get(i, {})

        # Initialize the plot
        plt.figure()

        # Plot data for each key in P-01
        for key in p01_data:
            entry = p01_data[key]
            # print(key)
            m = entry[0]
            y = entry[1]
            if key != 'fed' and  key != 'Noisy_protes':
                plt.plot(m, y, marker='o',label=key, markersize = 6, linewidth = 2)
            elif key == 'fed':
                plt.plot(m, y, marker='o',label='FED PROTES', markersize = 6, linewidth = 2)
            else:
                continue
        # Set plot labels and title
        plt.xlabel('Number of Request M', fontsize = 14)
        plt.ylabel('Function value', fontsize = 14)
        # plt.title('Combined P-01 Data Plot')
        plt.legend(title='Functions', fontsize = 8, title_fontsize=8)
        plt.grid(True)
        plt.savefig(f"../Results_new_fun/Fig/{i}_plot.pdf")
        # Show the plot
        plt.show()

# Call the function to plot the combined data
plot_combined_m_y()


