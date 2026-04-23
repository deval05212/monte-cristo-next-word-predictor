# Filtering data from output.txt

# Read file
with open('output.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Step 1: Remove all existing line breaks
text = text.replace('\n', ' ')

# Step 2: Normalize spaces
text = ' '.join(text.split())

# Step 3: Add new line after every period
text = text.replace('. ', '.\n')

# Step 4: Save cleaned text
with open('cleaned_data.txt', 'w', encoding='utf-8') as f:
    f.write(text)

print("Done! Cleaned file saved as cleaned_data.txt")
