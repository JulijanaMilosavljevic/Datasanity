import pandas as pd
from datasanity import check_dataset

df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")
report = check_dataset(df, target="survived")

print(report.to_dict())
html = report.to_html()

with open("datasanity_report.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Saved: datasanity_report.html")
