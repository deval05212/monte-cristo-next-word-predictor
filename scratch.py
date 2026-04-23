from keras.preprocessing.text import Tokenizer

with open("cleaned_data.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
total_words = len(tokenizer.word_index) + 1
print("Total words:", total_words)

seq_lengths = [len(line.split()) for line in lines]
print("Max seq length:", max(seq_lengths))
print("Avg seq length:", sum(seq_lengths) / len(seq_lengths))
