import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import pandas as pd

RESULTS_DIR = os.path.join(os.getcwd(), "results")
ZERO_DIR = os.path.join(RESULTS_DIR, "run001")
FIXED_DIR = os.path.join(RESULTS_DIR, "fixedshot_002")
EXP_DIR = os.path.join(RESULTS_DIR, "dynamicshot_002")
subset_labels = []
data_dict = {}

for root, dirs, files in os.walk(EXP_DIR):
    base_root = os.path.basename(os.path.normpath(root))
    if "qwen" in base_root:
        acc_list = []
        if "latest" in base_root:
            base_root = base_root.replace("latest", "7b")
        for dir in sorted(dirs):
            if "flight" not in dir:
                if dir not in subset_labels:
                    subset_labels.append(dir)
                with open(os.path.join(root, dir, "execution_acc_{}_dev.txt".format(dir[4:]))) as f:
                    filestr = f.read()
                    result = re.findall(r"Acc: \S*\n", filestr)
                    acc = result[0]
                    acc = acc[6:].strip("\n")
                    acc_list.append(float(acc))
        data_dict[base_root] = acc_list[:]

df = pd.DataFrame(data_dict)
df["Subset"] = subset_labels
dfm = df.melt(id_vars="Subset", var_name="Model", value_name="Accuracy")

print(df)

plt.figure(figsize=(10, 6))
#sns.barplot(data=dfm, x="Subset", y="Accuracy", hue="Model")
sns.barplot(data=df)

plt.title("Dynamic Accuracy by Model (NL similarity)")
plt.ylabel("Execution Accuracy")
plt.ylim(0, 1.0)
plt.yticks([x * 0.1 for x in range(11)])
plt.grid()
plt.tight_layout()

plt.show()