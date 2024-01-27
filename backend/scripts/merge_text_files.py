import os

# Specify the directory containing the text files
folder_path = 'cleaned'

# Specify the output file
output_file_path = 'backend/notebooks/merged_text.txt'

# Open the output file in append mode
with open(output_file_path, 'a', encoding='utf-8') as output_file:
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file has a .txt extension
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            # Read the content of the file and write it to the output file
            with open(file_path, 'r', encoding='utf-8') as input_file:
                file_content = input_file.read()
                output_file.write(file_content)

print("Merge complete. Check", output_file_path)
