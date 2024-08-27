import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned dataset
data = pd.read_csv(r'Portfolio 1\dataSet\converted_data.csv')

# Define the output directory for plots
output_dir = r'Portfolio 1\Outs\plots'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Univariate Analysis
cols = ['AT', 'V', 'AP', 'RH', 'PE']
length = len(cols)
cs = ["b", "r", "g", "c", "m", "k", "lime", "c"]
fig, axes = plt.subplots(4, 2, figsize=(13, 25))

for i, (ax, color) in enumerate(zip(axes.flatten(), cs)):
    if i < length:
        sns.histplot(data[cols[i]], ax=ax, color=color, kde=True)
        sns.rugplot(data[cols[i]], ax=ax, color=color)
        ax.set_facecolor("w")
        ax.axvline(data[cols[i]].mean(), linestyle="dashed", label="mean", color="k")
        ax.legend(loc="best")
        ax.set_title(f'{cols[i]} distribution', color="navy")
        ax.set_xlabel("")
    else:
        fig.delaxes(ax)  # Remove empty subplots

plt.tight_layout()

# Save univariate analysis plot
univariate_plot_path = os.path.join(output_dir, 'univariate_analysis.png')
plt.savefig(univariate_plot_path)
plt.show()
print(f"Univariate analysis plot saved to '{univariate_plot_path}'.")

# Multivariate Analysis

# Pairplot
pairplot = sns.pairplot(data[cols], diag_kind='kde', corner=True)
pairplot_path = os.path.join(output_dir, 'pairplot.png')
pairplot.savefig(pairplot_path)
plt.show()
print(f"Pairplot saved to '{pairplot_path}'.")

# Correlation heatmap
plt.figure(figsize=(10, 8))
corr = data[cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")

# Save correlation heatmap plot
heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
plt.savefig(heatmap_path)
plt.show()
print(f"Correlation heatmap saved to '{heatmap_path}'.")
