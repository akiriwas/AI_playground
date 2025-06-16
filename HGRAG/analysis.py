import pandas as pd
import matplotlib.pyplot as plt
import re
import os # Import os module to handle file existence

def analyze_and_plot_results(filepath="data.txt"):
    """
    Reads a data file, parses performance metrics, calculates statistics,
    and generates plots.

    Assumptions:
    1. The data file is located at the specified 'filepath' (default: data.txt).
    2. Each relevant line in the file follows the format:
       "RESULTS -- O: X, C: Y, R: Z, IT: A, QT: B"
       where X, Y, Z, A, B are numerical values (integers or floats).
    3. 'O' and 'C' are independent parameters that are varied.
    4. 'IT' (Iteration Time) and 'QT' (Query Time) are dependent timing results.
    5. 'R' (Result) is a parameter that might be constant or vary, but
       its primary use here is for grouping, and its average will be calculated.
    6. For each unique combination of (O, C), the script will calculate
       the mean and standard deviation for IT and QT. Only the mean is calculated for R.
    7. Plots will show the mean IT and QT against 'O' for different 'C' values,
       using different colored lines. Error bars will indicate the standard deviation.

    Args:
        filepath (str): The path to the data file.
    """

    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'. Please ensure the data file exists.")
        print("Creating a dummy 'data.txt' for demonstration purposes.")
        # Create a dummy file for demonstration if it doesn't exist
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
RESULTS -- O: 5, C: 49, R: 0.0, IT: 100.12345, QT: 0.300000
RESULTS -- O: 5, C: 49, R: 0.0, IT: 101.56789, QT: 0.310000
RESULTS -- O: 5, C: 49, R: 0.0, IT: 99.87654, QT: 0.295000
RESULTS -- O: 3, C: 50, R: 0.0, IT: 97.00000, QT: 0.270000
RESULTS -- O: 3, C: 50, R: 0.0, IT: 97.50000, QT: 0.275000
RESULTS -- O: 5, C: 50, R: 0.0, IT: 102.00000, QT: 0.320000
RESULTS -- O: 7, C: 49, R: 0.0, IT: 105.00000, QT: 0.350000
RESULTS -- O: 7, C: 49, R: 0.0, IT: 106.00000, QT: 0.355000
RESULTS -- O: 7, C: 50, R: 0.0, IT: 108.00000, QT: 0.370000
        """
        with open(filepath, "w") as f:
            f.write(dummy_content.strip())
        print(f"Dummy '{filepath}' created. Please run the script again.")
        return

    data = []
    # Regular expression to parse the line
    # It captures the numerical values after 'O:', 'C:', 'R:', 'IT:', 'QT:'
    pattern = re.compile(r"O:\s*(\d+),\s*C:\s*(\d+),\s*R:\s*([\d.]+),\s*IT:\s*([\d.]+),\s*QT:\s*([\d.]+)")

    print(f"Reading data from '{filepath}'...")
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                match = pattern.search(line)
                if match:
                    # Extract values and convert to appropriate types
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
                    # Warn about lines that don't match the expected format
                    if line.strip(): # Only warn if the line is not empty
                        print(f"Warning: Line {line_num} did not match expected format: '{line.strip()}'")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    if not data:
        print("No data found or parsed from the file. Exiting.")
        return

    # Create a Pandas DataFrame
    df = pd.DataFrame(data)

    print("\nRaw Data Head:")
    print(df.head())

    # Group by 'O' and 'C' and calculate mean and standard deviation
    # We calculate mean for R, IT, QT and std for IT, QT
    grouped_data = df.groupby(['O', 'C']).agg(
        IT_mean=('IT', 'mean'),
        IT_std=('IT', 'std'),
        QT_mean=('QT', 'mean'),
        QT_std=('QT', 'std'),
        R_mean=('R', 'mean')
    ).reset_index()

    # Fill NaN std values with 0 if there was only one data point for a group
    grouped_data['IT_std'] = grouped_data['IT_std'].fillna(0)
    grouped_data['QT_std'] = grouped_data['QT_std'].fillna(0)

    print("\nAggregated Data Head (Mean and Std Dev):")
    print(grouped_data.head())

    # Get unique 'C' values to plot separate lines
    unique_c_values = sorted(grouped_data['C'].unique())

    # --- Plotting IT vs O for different C values ---
    plt.figure(figsize=(12, 6)) # Create a figure for the plot
    for c_val in unique_c_values:
        subset = grouped_data[grouped_data['C'] == c_val]
        # Plot with error bars (standard deviation)
        plt.errorbar(
            subset['O'],
            subset['IT_mean'],
            yerr=subset['IT_std'],
            label=f'C: {c_val}',
            marker='o', # Marker for data points
            capsize=4 # Size of the caps on the error bars
        )

    plt.title('Average Iteration Time (IT) vs. O Parameter', fontsize=16)
    plt.xlabel('O Parameter', fontsize=12)
    plt.ylabel('Average IT (ms)', fontsize=12)
    plt.legend(title='C Value', loc='best') # Add a legend
    plt.grid(True, linestyle='--', alpha=0.7) # Add a grid
    plt.xticks(sorted(grouped_data['O'].unique())) # Ensure x-ticks are only at O values
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show() # Display the plot

    # --- Plotting QT vs O for different C values ---
    plt.figure(figsize=(12, 6)) # Create another figure for the second plot
    for c_val in unique_c_values:
        subset = grouped_data[grouped_data['C'] == c_val]
        # Plot with error bars (standard deviation)
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
    plt.show() # Display the plot

# To run the script, call the function:
analyze_and_plot_results("data.txt")

