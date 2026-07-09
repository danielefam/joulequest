import pandas as pd
import os
import data_processing as dp
import argparse

# Init your data holders
detailed_rows = []

layer_type = "linear_nograd"
data_folder = "/home/daniele/Desktop/tirocinio/energyBANERA-main/Data/new_measures/jetson_nano/Linear_nograd"
plot_folder = "/home/daniele/Desktop/tirocinio/energyBANERA-main/Plot/new_measures/jetson_nano/Linear_nograd"


def build_layer_name(filename):
    """Build a readable layer name from the CSV filename."""
    stem = os.path.splitext(filename)[0]
    parts = stem.split("_")
    if not parts:
        return stem

    layer = parts[0]
    params = parts[1:]
    if not params:
        return layer
    return f"{layer}({', '.join(params)})"


def main():
    parser = argparse.ArgumentParser(
         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_folder", type=str, default=data_folder, help="Path to the folder containing CSV data files.")
    parser.add_argument("--plot_folder", type=str, default=plot_folder, help="Path to the folder where plots will be saved.")
    parser.add_argument("--output", type=str, default="", help="Output Excel path. If empty, it is saved in ../statistics/measures_statistics.xlsx.")
    args = parser.parse_args()

    os.makedirs(args.plot_folder, exist_ok=True)

    data_files = sorted(f for f in os.listdir(args.data_folder) if f.lower().endswith(".csv"))
    if not data_files:
        raise FileNotFoundError(f"No CSV files found in: {args.data_folder}")

    for filename in data_files:
        filepath = os.path.join(args.data_folder, filename)

        parts = filename.replace(".csv", "").split("_")

        # Create plot filename
        plot_filename = f"{os.path.splitext(filename)[0]}.svg"
        plot_path = os.path.join(args.plot_folder, plot_filename)

        df=pd.read_csv(filepath)

        results = dp.get_average_power(
                                                                df,
                                                                kernel_size=11,  # Median filter kernel size
                                                                fs=5,            # Sampling frequency
                                                                cutoff=0.1,      # Low-pass filter cutoff frequency
                                                                window_size=30   # Rolling average window size
                                                            )

        dp.show_data(df, plot_path,save=True)
        #average_power = regions["avg_power_active_W"] -  regions["avg_power_idle_W"]
        # Add detailed row


        #Manage naming in the final excel file
        print(filename)
        layer_name = build_layer_name(filename)

        detailed_rows.append({
            "Layer (params)": layer_name,
            "Power mean (W)": results["power_avg_W"],
            "Power variance (W^2)": results["power_var_W2"],
            "Energy mean (J)": results["energy_avg_J"],
            "Energy variance (J^2)": results["energy_var_J2"],
            "Plot link": plot_path
        })

    # Build final dataframe with the exact requested columns
    detailed_df = pd.DataFrame(detailed_rows, columns=[
        "Layer (params)",
        "Power mean (W)",
        "Power variance (W^2)",
        "Energy mean (J)",
        "Energy variance (J^2)",
        "Plot link",
    ])

    statistics_folder = os.path.normpath(os.path.join(args.data_folder, "..", "statistics"))
    os.makedirs(statistics_folder, exist_ok=True)
    output_path = args.output.strip() or os.path.join(statistics_folder, f"measures_statistics_{layer_type}.xlsx")

    # Write Excel report
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        # Write all columns except link path, then add a clickable link column.
        detailed_df.drop(columns=["Plot link"]).to_excel(writer, sheet_name="Measures Stats", index=False)

        workbook = writer.book
        worksheet = writer.sheets["Measures Stats"]
        link_col = detailed_df.drop(columns=["Plot link"]).shape[1]

        # Write link column header.
        worksheet.write(0, link_col, "Plot link")

        # Add hyperlinks to plots.
        for row_idx, file_path in enumerate(detailed_df["Plot link"], start=1):
            abs_path = os.path.abspath(file_path)
            plot_url = f"external:{abs_path}"
            worksheet.write_url(row_idx, link_col, plot_url, string="View Plot")

        # Basic column sizing for readability.
        worksheet.set_column(0, 0, 26)
        worksheet.set_column(1, 4, 20)
        worksheet.set_column(5, 5, 14)

    print(f"Excel report saved to: {output_path}")

if __name__ == "__main__":
    main()