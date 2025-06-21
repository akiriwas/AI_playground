import pandas as pd
import matplotlib.pyplot as plt
import re
import os
from mpl_toolkits.mplot3d import Axes3D # Import for 3D plotting

def analyze_and_plot_results(filepath="data.txt"):
    """
    Reads a data file, parses performance metrics, calculates statistics,
    and generates 2D and 3D plots, saving them to image files.

    Assumptions:
    1. The data file is located at the specified 'filepath' (default: data.txt).
    2. Each relevant line in the file follows the format:
       "RESULTS -- O: X, C: Y, R: Z, IT: A, QT: B"
       where X, Y, Z, A, B are numerical values (integers or floats).
    3. 'O' and 'C' are independent parameters that are varied.
    4. 'IT' (Iteration Time) and 'QT' (Query Time) are dependent timing results.
    5. 'R' (Recall/Effectiveness) is another dependent result.
    6. For each unique combination of (O, C), the script will calculate
       the mean and standard deviation for IT and QT, and the mean for R.
    7. Plots will show the mean IT, QT, and R against 'O' for different 'C' values
       in 2D line plots, with error bars for IT and QT.
    8. An additional 3D plot will show Average Recall (R) as a surface/scatter
       against O and C.
    9. The plots will be saved as 'average_it_vs_o.png', 'average_qt_vs_o.png',
       'average_recall_vs_o.png', and '3d_recall_vs_oc.png'
       in the same directory as the script.

    Args:
        filepath (str): The path to the data file.
    """

    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'. Please ensure the data file exists.")
        print("Creating a dummy 'data.txt' for demonstration purposes.")
        dummy_content = """
RESULTS -- O: 3, C: 49, R: 0.0, IT: 96.595244, QT: 0.260323
RESULTS -- O: 3, C: 49, R: 0.0, IT: 96.167212, QT: 0.259569
RESULTS -- O: 3, C: 49, R: 0.0, IT: 96.663801, QT: 0.258461
RESULTS -- O: 3, C: 49, R: 0.0, IT: 95.851123, QT: 0.257880
RESULTS -- O: 3, C: 49, R: 0.0, IT: 95.847144, QT: 0.259208
RESULTS -- O: 3, C: 49, R: 0.0, IT: 95.813371, QT: 0.257743
RESULTS -- O: 3, C: 49, R: 0.0, IT: 96.018635, QT: 0.257874
RESULTS -- O: 3, C: 49, R: 0.0, IT: 96.019632, QT: 0.257955
RESULTS -- O: 3, C: 49, R: 0.0, IT: 96.600158, QT: 0.258618
RESULTS -- O: 3, C: 49, R: 0.0, IT: 97.073794, QT: 0.258072
RESULTS -- O: 5, C: 49, R: 0.1, IT: 100.12345, QT: 0.300000
RESULTS -- O: 5, C: 49, R: 0.1, IT: 101.56789, QT: 0.310000
RESULTS -- O: 5, C: 49, R: 0.1, IT: 99.87654, QT: 0.295000
RESULTS -- O: 3, C: 50, R: 0.05, IT: 97.00000, QT: 0.270000
RESULTS -- O: 3, C: 50, R: 0.05, IT: 97.50000, QT: 0.275000
RESULTS -- O: 5, C: 50, R: 0.15, IT: 102.00000, QT: 0.320000
RESULTS -- O: 7, C: 49, R: 0.2, IT: 105.00000, QT: 0.350000
RESULTS -- O: 7, C: 49, R: 0.2, IT: 106.00000, QT: 0.355000
RESULTS -- O: 7, C: 50, R: 0.25, IT: 108.00000, QT: 0.370000
RESULTS -- O: 3, C: 51, R: 0.08, IT: 98.00000, QT: 0.280000
RESULTS -- O: 5, C: 51, R: 0.18, IT: 103.00000, QT: 0.330000
RESULTS -- O: 7, C: 51, R: 0.3, IT: 110.00000, QT: 0.380000
        """
        with open(filepath, "w") as f:
            f.write(dummy_content.strip())
        print(f"Dummy '{filepath}' created. Please run the script again.")
        return

    data = []
    pattern = re.compile(r"O:\s*(\d+),\s*C:\s*(\d+),\s*R:\s*([\d.]+),\s*IT:\s*([\d.]+),\s*QT:\s*([\d.]+)")

    print(f"Reading data from '{filepath}'...")
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                match = pattern.search(line)
                if match:
                    o_val = int(match.group(1))
                    c_val = int(match.group(2))
                    r_val = float(match.group(3))
                    it_val = float(match.group(4))
                    qt_val = float(match.group(5))
                    data.append({
                        'O': o_val,
                        'C': c_val,
                        'R': r_val,
                        'IT': it_val,
                        'QT': qt_val
                    })
                else:
                    if line.strip():
                        print(f"Warning: Line {line_num} did not match expected format: '{line.strip()}'")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    if not data:
        print("No data found or parsed from the file. Exiting.")
        return

    df = pd.DataFrame(data)

    print("\nRaw Data Head:")
    print(df.head())

    grouped_data = df.groupby(['O', 'C']).agg(
        IT_mean=('IT', 'mean'),
        IT_std=('IT', 'std'),
        QT_mean=('QT', 'mean'),
        QT_std=('QT', 'std'),
        R_mean=('R', 'mean')
    ).reset_index()

    grouped_data['IT_std'] = grouped_data['IT_std'].fillna(0)
    grouped_data['QT_std'] = grouped_data['QT_std'].fillna(0)

    print("\nAggregated Data Head (Mean and Std Dev):")
    print(grouped_data.head())

    unique_c_values = sorted(grouped_data['C'].unique())

    # --- Plotting IT vs O for different C values ---
    plt.figure(figsize=(12, 6))
    for c_val in unique_c_values:
        subset = grouped_data[grouped_data['C'] == c_val]
        plt.errorbar(
            subset['O'],
            subset['IT_mean'],
            yerr=subset['IT_std'],
            label=f'C: {c_val}',
            marker='o',
            capsize=4
        )
    plt.title('Average Iteration Time (IT) vs. O Parameter', fontsize=16)
    plt.xlabel('O Parameter', fontsize=12)
    plt.ylabel('Average IT (ms)', fontsize=12)
    plt.legend(title='C Value', loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(sorted(grouped_data['O'].unique()))
    plt.tight_layout()
    plt.savefig('average_it_vs_o.png')
    plt.close()
    print("average_it_vs_o.png")

    # --- Plotting QT vs O for different C values ---
    plt.figure(figsize=(12, 6))
    for c_val in unique_c_values:
        subset = grouped_data[grouped_data['C'] == c_val]
        plt.errorbar(
            subset['O'],
            subset['QT_mean'],
            yerr=subset['QT_std'],
            label=f'C: {c_val}',
            marker='o',
            capsize=4
        )
    plt.title('Average Query Time (QT) vs. O Parameter', fontsize=16)
    plt.xlabel('O Parameter', fontsize=12)
    plt.ylabel('Average QT (ms)', fontsize=12)
    plt.legend(title='C Value', loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(sorted(grouped_data['O'].unique()))
    plt.tight_layout()
    plt.savefig('average_qt_vs_o.png')
    plt.close()
    print("average_qt_vs_o.png")

    # --- Plotting R (Recall) vs O for different C values (2D) ---
    plt.figure(figsize=(12, 6))
    for c_val in unique_c_values:
        subset = grouped_data[grouped_data['C'] == c_val]
        plt.plot(
            subset['O'],
            subset['R_mean'],
            label=f'C: {c_val}',
            marker='o'
        )
    plt.title('Average Recall (R) vs. O Parameter', fontsize=16)
    plt.xlabel('O Parameter', fontsize=12)
    plt.ylabel('Average R (Recall)', fontsize=12)
    plt.legend(title='C Value', loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(sorted(grouped_data['O'].unique()))
    plt.tight_layout()
    plt.savefig('average_recall_vs_o.png')
    plt.close()
    print("average_recall_vs_o.png")

    # --- 3D Plotting Average Recall (R) vs O and C ---
    fig = plt.figure(figsize=(12, 10)) # Create a new figure for the 3D plot
    ax = fig.add_subplot(111, projection='3d') # Add a 3D subplot

    # Extract data for 3D plot
    o_values = grouped_data['O']
    c_values = grouped_data['C']
    r_mean_values = grouped_data['R_mean']

    # Use plot_trisurf for a surface plot if data density allows, otherwise scatter
    # Since the example data is sparse, a scatter plot might be clearer
    #ax.scatter(o_values, c_values, r_mean_values, c=r_mean_values, cmap='viridis', s=100)
    ax.plot_trisurf(o_values, c_values, r_mean_values, cmap='viridis', edgecolor='none')

    ax.set_title('3D Plot of Average Recall (R) vs. O and C Parameters', fontsize=16)
    ax.set_xlabel('O Parameter', fontsize=12)
    ax.set_ylabel('C Parameter', fontsize=12)
    ax.set_zlabel('Average R (Recall)', fontsize=12)

    # You might need to adjust view angle for better visualization
    #ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig('3d_recall_vs_oc.png') # Save the 3D plot
    plt.close()
    print("3d_recall_vs_oc.png")

# To run the script, call the function:
analyze_and_plot_results("benchmark1")

