import pandas as pd
import numpy as np

def csv_to_latex_smart(
    file_path, 
    highlight_col="",          
    columns_to_include=[],     
    decimal_places=2, 
    output_file=None,
    separator=';',
    add_in1k_validation=False,
    in1k_validation_file=None,
    in1k_source_col="imagenet_top1",
    match_key="model_key"
):
    print(f"--- Starting Processing ---")
    
    # 1. Read Main CSV
    try:
        df = pd.read_csv(file_path, sep=separator)
        print(f"Main CSV loaded. Columns found: {list(df.columns)}")
    except FileNotFoundError:
        print(f"Error: Main file '{file_path}' not found.")
        return

    # 2. Merge IN1K Validation Data (Do this BEFORE filtering)
    if add_in1k_validation:
        if not in1k_validation_file:
            print("Error: 'add_in1k_validation' is True, but no file provided.")
            return
        try:
            df_val = pd.read_csv(in1k_validation_file, sep=separator)
            print(f"Validation CSV loaded. Columns: {list(df_val.columns)}")
        except FileNotFoundError:
            print(f"Error: Validation file '{in1k_validation_file}' not found.")
            return
        
        if match_key not in df.columns:
            print(f"Error: Match key '{match_key}' NOT in Main CSV. Available: {list(df.columns)}")
            return
        if match_key not in df_val.columns:
            print(f"Error: Match key '{match_key}' NOT in Validation CSV. Available: {list(df_val.columns)}")
            return
            
        if in1k_source_col not in df_val.columns:
            print(f"Error: Source column '{in1k_source_col}' NOT in Validation CSV.")
            return

        df_merge = df_val[[match_key, in1k_source_col]].copy()
        new_col_name = "in1k_validation"
        df_merge.rename(columns={in1k_source_col: new_col_name}, inplace=True)
        
        before_merge_count = len(df.columns)
        df = pd.merge(df, df_merge, on=match_key, how='left')
        after_merge_count = len(df.columns)
        
        print(f"Merge successful. Added '{new_col_name}'. Total columns now: {after_merge_count}")
        print(f"Current columns after merge: {list(df.columns)}")

    # 3. Filter Columns
    if columns_to_include:
        print(f"\nRequested columns to include: {columns_to_include}")
        
        # Check exactly what matches
        found_cols = []
        missing_cols = []
        
        for req_col in columns_to_include:
            if req_col in df.columns:
                found_cols.append(req_col)
            else:
                missing_cols.append(req_col)
        
        if missing_cols:
            print(f"WARNING: The following columns were NOT found and will be skipped: {missing_cols}")
            print(f"Tip: Check for typos or extra spaces. Available columns are: {list(df.columns)}")
        
        if not found_cols:
            print("ERROR: No requested columns were found. Aborting.")
            return
            
        df = df[found_cols]
        print(f"Filtering applied. Keeping columns: {list(df.columns)}")
    
    else:
        print("No column filter applied. Keeping all columns.")

    # 4. Clean Headers & Insert Line Breaks
    new_columns = {}
    for col in df.columns:
        clean_name = col.replace('_', ' ').title()
        if col == "in1k_validation":
            clean_name = "In1k Validation"
        
        # Insert line breaks
        if "Top 1" in clean_name:
            clean_name = clean_name.replace("Top 1", "Top \\\\ 1")
        elif "Top 5" in clean_name:
            clean_name = clean_name.replace("Top 5", "Top \\\\ 5")
        elif "Delta" in clean_name and "Vs" in clean_name:
            clean_name = clean_name.replace(" Vs ", " \\\\ Vs ")
        elif "Base To Average" in clean_name:
            clean_name = clean_name.replace(" To ", " \\\\ To ")
        elif len(clean_name) > 14: 
            parts = clean_name.split()
            if len(parts) > 2:
                mid = len(parts) // 2
                clean_name = " ".join(parts[:mid]) + " \\\\ " + " ".join(parts[mid:])
        
        new_columns[col] = clean_name
    
    df.rename(columns=new_columns, inplace=True)
    
    # (Highlight logic remains same as before...)
    clean_highlight_col = ""
    if highlight_col:
        temp_clean = highlight_col.replace('_', ' ').title()
        # Simple mapping attempt
        found_hl = False
        for key in df.columns:
            # Remove the \\ for comparison
            clean_key = key.replace(' \\\\ ', ' ')
            if clean_key == temp_clean or key == temp_clean:
                clean_highlight_col = key
                found_hl = True
                break
        if not found_hl:
            print(f"Warning: Highlight column '{highlight_col}' not found in processed columns. Skipping highlight.")
            clean_highlight_col = "" # Disable highlight if not found

    # 5. Format Numbers
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(decimal_places)
    for col in numeric_cols:
        df[col] = df[col].apply(lambda x: f"{x:.{decimal_places}f}")

    # 6. Identify Max Values
    cols_to_process = []
    if clean_highlight_col:
        cols_to_process = [clean_highlight_col]
    else:
        cols_to_process = list(numeric_cols)
        
    max_values = {}
    for col in cols_to_process:
        if col in df.columns:
            max_values[col] = df[col].astype(float).max()

    # 7. Apply Formatting
    def format_cell(val, col_name):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return ""
        if isinstance(val, str):
            val = val.replace('_', '\\_')
        if col_name in max_values:
            try:
                if float(val) == max_values[col_name]:
                    return f"\\textbf{{{val}}}"
            except ValueError:
                pass 
        return val

    df_formatted = pd.DataFrame(index=df.index)
    for col in df.columns:
        df_formatted[col] = df[col].apply(lambda x: format_cell(x, col))

    # 8. Generate LaTeX
    num_cols = len(df.columns)
    col_format = '|' + '|'.join(['c'] * num_cols) + '|'
    
    latex_code = "\\begin{table}[h]\n"
    latex_code += "  \\centering\n"
    latex_code += "  \\small\n" #or \small ≈ 10pt footnotesize ≈ 9pt or \scriptsize ≈ 7pt
    latex_code += f"  \\begin{{tabular}}{{{col_format}}}\n"
    latex_code += "    \\toprule\n"
    
    headers = " & ".join(df_formatted.columns) + " \\\\\n"
    latex_code += "    " + headers
    latex_code += "    \\midrule\n"
    
    for index, row in df_formatted.iterrows():
        row_str = " & ".join(row.values) + " \\\\\n"
        latex_code += "    " + row_str
        
    latex_code += "    \\bottomrule\n"
    latex_code += "  \\end{tabular}\n"
    latex_code += "  \\caption{Model Rankings}\n"
    latex_code += "  \\label{tab:rankings}\n"
    latex_code += "\\end{table}"

    if output_file:
        with open(output_file, 'w') as f:
            f.write(latex_code)
        print(f"\nLaTeX table saved to {output_file}")
    else:
        print("\n--- COPY THIS INTO YOUR LATEX DOCUMENT ---\n")
        print(latex_code)
        print("\n--------------------------------------------\n")


# ================= CONFIGURATION =================
CSV_FILE = "results/plastic_dataset/comparative/model_rankings.csv"
ADD_IN1K_VALIDATION = False #True  # Set to True to add the column
IN1K_VALIDATION_FILE = "results/imagenet-1k/comparative/model_rankings.csv" # Path to the 2nd CSV
IN1K_SOURCE_COL = "imagenet_top1" # Column name in the 2nd CSV to pull
MATCH_KEY = "model_key"           # Column name present in BOTH files to match rows

# --- HIGHLIGHT & FILTER OPTIONS ---
COLUMN_TO_HIGHLIGHT = "" # Leave empty "" to highlight max in ALL numeric columns

# Select specific columns to display (Use ORIGINAL CSV names)
# Note: If you added IN1K validation, you can include "in1k_validation" here
COLUMNS_TO_INCLUDE = [
    "model_name", 
    #"in1k_validation",
    "imagenet_top1", 
    "imagenet_top5",
    #"shapenet_base_top1",
    "shapenet_top1"
]

DECIMAL_PLACES = 3                     
OUTPUT_FILE = None                     
SEPARATOR = ';' 

# ================= RUN =================
if __name__ == "__main__":
    csv_to_latex_smart(
        file_path=CSV_FILE,
        highlight_col=COLUMN_TO_HIGHLIGHT,
        columns_to_include=COLUMNS_TO_INCLUDE,
        decimal_places=DECIMAL_PLACES,
        output_file=OUTPUT_FILE,
        separator=SEPARATOR,
        add_in1k_validation=ADD_IN1K_VALIDATION,
        in1k_validation_file=IN1K_VALIDATION_FILE,
        in1k_source_col=IN1K_SOURCE_COL,
        match_key=MATCH_KEY
    )