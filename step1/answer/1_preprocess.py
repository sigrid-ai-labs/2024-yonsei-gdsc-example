import pandas as pd

csv_path = "./step1/data/example.csv"

df = pd.read_csv(csv_path)


def map_to_label(row):
    category = row["category"]
    if category in ["Information Seeking", "Coding"]:
        return "gpt-o1"
    elif category in ["Writing", "Language"]:
        return "Claude"
    else:
        return "gemini-mini"


df["llm"] = df.apply(map_to_label, axis=1)

print(df.head())

df.to_csv("./step1/data/labeled_data.csv", index=False)
