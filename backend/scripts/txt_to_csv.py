import os
import csv

def text_to_csv(input_file, output_file, max_tokens_per_row=256):
    with open(input_file, 'r', encoding='utf-8') as infile:
        content = infile.read().split()

    rows = [content[i:i + max_tokens_per_row] for i in range(0, len(content), max_tokens_per_row)]

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Text'])  # Header for the single column

        for row in rows:
            csvwriter.writerow([' '.join(row)])

# Replace 'path/to/directory' with the actual path to your directory containing 'merged_txt.txt'
directory_path = 'merged_text.txt'

# input_file_path = os.path.join(directory_path, 'merged_txt.txt')
output_file_path = 'merged.csv'
# print(directory_path)
text_to_csv(directory_path, output_file_path, max_tokens_per_row=256)


