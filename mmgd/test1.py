import csv

def get_file_keys(file_path,id_field):
    keys = []
    with open(file_path) as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            print(row)
            keys.append(row[id_field])
    return keys

node_keys=get_file_keys('db/nodes/ids.txt','nodeId')

print(node_keys)