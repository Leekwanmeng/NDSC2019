import os
import csv
import sys

def make_sub_csv(train_txt_path, out_name, str_token):
    with open(train_txt_path, encoding='utf-8', newline='') as f:
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
    train_txt_path = 'data/train.csv'
    for mode in ['mobile_image', 'beauty_image', 'fashion_image']:
        make_sub_csv(train_txt_path, 'train_' + mode + '.csv', mode)