import os
import csv
import sys

def make_sub_csv(csv_path, out_name, str_token):
    """
    Extracts rows of train.csv to new file, using str_token
    To split beauty, fashion and mobile
    """
    with open(train_csv_path, encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        print(header)
        with open(out_name, 'w') as out:
            writer = csv.writer(out)
            writer.writerow(header)
            for row in reader:
                if str_token in row[3]:
                    writer.writerow(row)


if __name__=='__main__':
    train_csv_path = 'data/train.csv'
    for mode in ['mobile_image', 'beauty_image', 'fashion_image']:
        make_sub_csv(train_csv_path, 'train_' + mode + '.csv', mode)