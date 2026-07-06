import pandas as pd
import numpy as np
import os
import data_processing as dp
import argparse

# Init your data holders
detailed_rows = []
net_power_dict = {}  # Nested dict for matrix


data_folder = "/home/frederic/Documents/BANERA/Code/Data/Jetson_nano_power_record/Conv_3_0"
plot_folder = "/home/frederic/Documents/BANERA/Code/Plot/Jetson_nano_plots/Conv_3_0"
def main():
    parser = argparse.ArgumentParser(
         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_folder", type=str, default=data_folder, help="Path to the folder containing CSV data files.")
    parser.add_argument("--plot_folder", type=str, default=plot_folder, help="Path to the folder where plots will be saved.")
    args = parser.parse_args()

    data_files = sorted(os.listdir(args.data_folder))  # Assume: one file per layer config

    for filename in data_files:
        filepath = os.path.join(args.data_folder, filename)

        # Read the CSV file
        parts = filename.replace(".csv", "").split("_")
        in_size = int(parts[1])
        out_size = int(parts[2])

        # Create plot filename
        plot_filename = f"{parts[0]}_{in_size}x{out_size}.svg"
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
        if parts[0]=="Linear":
             layer_name = f"Linear({in_size}, {out_size})"
        elif parts[0]=="Conv":
             layer_name = f"Conv({in_size}, {out_size},{parts[3]},{parts[4]})"

        detailed_rows.append({
            "Layer": layer_name,
            "Power mean (W)": results["power_avg_W"],
            "Power variance (W²)": results["power_var_W2"],
            "Energy mean (J)": results["energy_avg_J"],
            "Energy variance (J²)": results["energy_var_J2"],
            "Plot": plot_path
        })

        # Fill matrix
        net_power_dict.setdefault(in_size, {})[out_size] = results["energy_avg_J"]

    # === Build final dataframes ===

    # Detailed sheet
    detailed_df = pd.DataFrame(detailed_rows)
    ## columns number to put the hyperlink plots
    num_cols=detailed_df.drop(columns=["Plot"]).shape[1]

    # Matrix sheet
    matrix_df = pd.DataFrame(net_power_dict).T.sort_index().sort_index(axis=1)
    matrix_df.index.name = "In\Out"

    # === Write to Excel ===
    # with pd.ExcelWriter("linear_power_report.xlsx", engine="openpyxl") as writer:
    #     matrix_df.to_excel(writer, sheet_name="Net Power Matrix")
    #     detailed_df.to_excel(writer, sheet_name="Detailed Stats", index=False)

    with pd.ExcelWriter("linear_power_report.xlsx", engine="xlsxwriter") as writer:
        # Write sheets
        matrix_df.to_excel(writer, sheet_name="Average Power Matrix")
        detailed_df.drop(columns=["Plot"]).to_excel(writer, sheet_name="Detailed Stats", index=False)

        workbook = writer.book
        worksheet = writer.sheets["Detailed Stats"]

        # Add hyperlinks to plots in the Detailed Stats sheet
        for row_idx, file_path in enumerate(detailed_df["Plot"], start=1):  # start=1 to skip header
            abs_path = os.path.abspath(file_path)
            #hyperlink = f'file:///{abs_path.replace(os.sep, "/")}'  # Make sure path is URI-friendly
            #hyperlink= os.path.join("Plot", plot_filename)
            #hyperlink = f"file://{detailed_df['Plot']}"
            worksheet.write_url(row_idx, num_cols, file_path, string="View Plot") 
        

        # Add hyperlinks from matrix to detailed stats
        matrix_sheet = writer.sheets["Average Power Matrix"]
        detailed_sheet_name = "Detailed Stats"

        row_idx = 1
        for row_idx, in_size in enumerate(matrix_df.index, start=1):  # start=1 for Excel row 2 (below header)
                for col_idx, out_size in enumerate(matrix_df.columns, start=1):  # start=1 for Excel col B
                    value = matrix_df.loc[in_size, out_size]

                    # Lookup corresponding row in detailed_df
                    layer_label = f"Linear({in_size}, {out_size})"
                    match = detailed_df[detailed_df["Layer"] == layer_label]
                    if not match.empty:
                        target_row_idx = match.index[0] + 2  # +2 to skip header and be 1-based
                        #formula = f'=HYPERLINK("#\'{detailed_sheet_name}\'!A{target_row_idx}", "{value:.6f}")'

                        # Write the hyperlink formula to the cell
                        #matrix_sheet.write_formula(row_idx, col_idx, formula)  # +1 to skip matrix header row
                        matrix_sheet.write(row_idx, col_idx, value)
if __name__ == "__main__":
    main()