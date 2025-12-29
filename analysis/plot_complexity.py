import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = 'analysis/assets/complexity.xlsx'
sheets = pd.read_excel(file_path, sheet_name=None, header=0)


def format_model_name(name: str) -> str:
    """
    Convert raw model name with optional ^ (superscript) and _ (subscript)
    into a LaTeX-formatted string for the legend.
    """
    if not isinstance(name, str):
        return str(name)

    base = name
    superscript_part = ''
    subscript_part = ''
    
    if base.find('_') != -1:
        if base.find('^') != -1:
            if base.find('_') > base.find('^'):
                base, subscript_part = base.split('_', 1)
                base, superscript_part = base.split('^', 1)
            else:
                base, superscript_part = base.split('^', 1)
                base, subscript_part = base.split('_', 1)
        else:
            base, superscript_part = base.split('_', 1)
    else:
        if base.find('^') != -1:
            base, superscript_part = base.split('^', 1)

    base = base.replace("BERT-base", "BERT_{{{base}}}")
    base = base.replace("KOMBO", "KOMBO_{{{base}}}")

    result = f"${base}"
    if subscript_part:
        result += f"_{{{subscript_part}}}"
    if superscript_part:
        result += f"^{{{superscript_part}}}"
    result += "$"
    return result


y_labels = {
    0: "GPU Memory (GB)",
    1: "Training Time (sec/epoch)"
}

for idx, (sheet_name, df) in enumerate(sheets.items()):
    # Assume first column holds model names if it's non‑numeric
    if not np.issubdtype(df.iloc[:, 0].dtype, np.number):
        df = df.set_index(df.columns[0])

    # Identify sequence length columns (must be numeric)
    seq_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    # Convert column names to numeric seq lengths
    seq_lengths = sorted([int(c) for c in seq_cols])

    # Ensure columns are ordered
    df = df[[str(n) if str(n) in df.columns else n for n in seq_lengths]]

    plt.figure()
    for model_name, row in df.iterrows():
        y = row.values.astype(float)
        new_model_name = format_model_name(model_name)
        plt.plot(seq_lengths, y, marker='o', label=new_model_name, linewidth=2)

    plt.xlabel("Input Sequence Length", fontsize=14, labelpad=10)
    plt.ylabel(y_labels.get(idx, "Metric Value"), fontsize=14, labelpad=10)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.legend(fontsize=13)
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.2, alpha=0.3)
    
    # Hide the right and top spines
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"analysis/assets/complexity_{sheet_name}.pdf")
    plt.show()

