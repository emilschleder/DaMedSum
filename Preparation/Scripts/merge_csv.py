import pandas as pd

# Read the first CSV file
file1 = pd.read_csv("data/new/netdoktor_texts.csv")

# Read the second CSV file
file2 = pd.read_csv("data/new/sundheddk_texts.csv")

# Concatenate the two dataframes
merged_file = pd.concat([file1, file2])

# Save the merged dataframe to a new CSV file
merged_file.to_csv("data/summaries.csv", index=False)