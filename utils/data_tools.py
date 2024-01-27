"""
Convert .txt to csv

"""

import csv
from sklearn.model_selection import train_test_split

def convert_txt_to_csv(input_txt, output_csv):
    with open(input_txt, 'r', encoding='utf-8') as infile, open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
       
        reader = infile.readlines()
        data = [line.strip().split() for line in reader]
        csv_writer = csv.writer(outfile)
        csv_writer.writerows(data)

def split_data(input_csv, output_train_csv, output_test_csv, output_val_csv, test_size=0.2, val_size=0.1, random_seed=42):
    with open(input_csv, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        data = list(csv_reader)
        
    train_data, test_val_data = train_test_split(data, test_size=(test_size + val_size), random_state=random_seed)
    test_data, val_data = train_test_split(test_val_data, test_size=(val_size / (test_size + val_size)), random_state=random_seed)

    with open(output_train_csv, 'w', encoding='utf-8', newline='') as train_file:
        csv_writer = csv.writer(train_file)
        csv_writer.writerows(train_data)

    with open(output_test_csv, 'w', encoding='utf-8', newline='') as test_file:
        csv_writer = csv.writer(test_file)
        csv_writer.writerows(test_data)

    with open(output_val_csv, 'w', encoding='utf-8', newline='') as val_file:
        csv_writer = csv.writer(val_file)
        csv_writer.writerows(val_data)

if __name__ == "__main__":
    input_txt_file = '/home/biniyam_ajaw/finetuning/data/dataset.txt'
    output_csv_file = '/home/biniyam_ajaw/finetuning/data/output_data.csv'
    output_train_csv = '/home/biniyam_ajaw/finetuning/data/train_data.csv'
    output_test_csv = '/home/biniyam_ajaw/finetuning/data/test_data.csv'
    output_val_csv = '/home/biniyam_ajaw/finetuning/data/val_data.csv'

    convert_txt_to_csv(input_txt_file, output_csv_file)
    split_data(output_csv_file, output_train_csv, output_test_csv, output_val_csv)
    print("Conversion to CSV and data split completed.")

