import pandas as pd
import matplotlib.pyplot as plt


excel_f = pd.ExcelFile("analysis/assets/nlu_results_per_eps.xlsx")
sheet_names = excel_f.sheet_names
print(f"NLU Tasks: {sheet_names}")

sheets_dict = {sheet: excel_f.parse(sheet, index_col=False) for sheet in sheet_names}

all_scores = {}  # Store all model data for plotting

for sheet_name, df in sheets_dict.items():
    model_names = df.iloc[:, 0]  # Extract model names
    scores = df.iloc[:, 1:]  # Extract task scores (columns)
    all_scores[sheet_name] = {"models": model_names, "scores": scores}


# Plotting results for each NLU task individually

script_top_epoch = -1

fig, axes = plt.subplots(3, 3, figsize=(18, 13))
for idx, (sheet_name, data) in enumerate(all_scores.items()):
    ax = axes[idx // 3, idx % 3]

    colors = {"BERT_base": "#6497D3", "BERT_base+script": "#FF6961",
              "KOMBO_base": "#77DD77"}  # Define more vivid color scheme

    for model_idx, model_name in enumerate(data["models"]):
        scores = data["scores"].iloc[model_idx]
        scores = scores.rename({f'ep {i}': str(i) for i in range(1, len(scores) + 1)})
        top_epoch = scores.idxmax()  # Find the epoch with the Top Epoch
        if model_name == "BERT_base+script":
            script_top_epoch = top_epoch

        ax.plot(
            scores.index,
            scores,
            linewidth=2.5,
            label="$" + model_name.replace("script", f"SUB^{{{2}}}_{{{'jamo'}}}").replace(f"BERT_base", f"BERT_{{{'base'}}}").replace("KOMBO_base", f"KOMBO_{{{'base'}}}^{{{'jamo'}}}") + "$",
            marker="o",
            color=colors.get(model_name, "grey"),  # Assign color based on model name
            alpha=0.8  # Increase opacity for vivid colors
        )
        if model_name == "BERT_base+script":
            line_color = "#FF5C5C"  # Brighter red for BERT_base+script
            linewidth = 3
            label_name = f"Top Epoch (SUB\u00b2)"
        else:
            line_color = "grey"  # Light grey for other models
            linewidth = 2.
            label_name = f"Top Epoch (Others)"

        if (model_name != "BERT_base+script") and (top_epoch == script_top_epoch):
            continue
        else:
            ax.axvline(
                x=top_epoch,
                color=line_color,
                linestyle="--",
                linewidth=linewidth,
                alpha=1.,  # Reduce transparency for clearer visibility
                label=label_name,
            )  # Add vertical line at Top Epoch
        for spine in ax.spines.values():
            spine.set_edgecolor("grey")
        ax.set_title(f"{sheet_name.replace('_', '-')}", fontsize=20)

    # Add legend and format axes for each subplot
    handles, labels = ax.get_legend_handles_labels()
    order = [i for i, lbl in enumerate(labels) if lbl not in ['Top Epoch (SUB\u00b2)', 'Top Epoch (Others)']]
    order.extend([labels.index('Top Epoch (SUB\u00b2)'), labels.index('Top Epoch (Others)')])
    ax.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=24, facecolor="white", framealpha=1)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(False)

plt.tight_layout()
plt.savefig("analysis/assets/nlu_results_per_eps.pdf")
plt.savefig("analysis/assets/nlu_results_per_eps.png", dpi=300)
plt.show()
