# Extract Text from PDF
import pdfplumber

all_text = ""

with pdfplumber.open("file.pdf") as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            all_text += text + "\n"

with open("output.txt", "w", encoding="utf-8") as f:
    f.write(all_text)
