import csv

articles = []
summary = ""
text = ""
current_section = None
prefixes = tuple(("Hvad", "Hvordan", "Hvor", "Hvorfor", "Operation for skelen"))

def append_article():
    if len(summary) < len(text):
        articles.append({"source": "Netdoktor", "text": text, "summary": summary})

with open("data/new/Netdoktor_all_text.txt", "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()

        if line.startswith("Fakta om"):
            if current_section in ["summary", "text"]:
                append_article()

            current_section = "summary"
            summary = line

        elif line.startswith(prefixes) and current_section == "summary":
            current_section = "text"
            text = line
        
        elif not line or line.startswith("Læs mere om"):
            if current_section == "text":
                append_article()

            current_section = None
            summary = ""
            text = ""
        elif current_section is not None:
            if current_section == "summary":
                summary += " " + line  
            elif current_section == "text":
                text += " " + line 

if current_section in ["summary", "text"]:
    append_article()

print(len(articles))
    
# Writing to a CSV file
csv_columns = ['source', 'text', 'summary']
csv_file = "netdoktor_texts.csv"

with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()
    for article in articles:
        writer.writerow(article)
