import csv
import numpy as np


for file_path in ["./trauma_trainV2.csv", "./trauma_valV2.csv"]:
    proposal = {}
    with open(file_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        for line in csv_reader:
            video_id = line[0]
            timestamp = int(line[1])
            img_key = f'{video_id},{timestamp:04d}'
            box = [ min(float(x), 1) for x in line[2:6]]
            box = [max(0, x ) for x in box]
            box = np.array(box)
            print(box)
            if img_key in proposal:
                proposal[img_key].append(box)
            else:
                proposal[img_key] = [box]

    for key, value in proposal.items():
        proposal[key] = np.array(value)



    import pickle

    pickle_file_path = "{}.pkl".format(file_path)

    # Save the 'lines' list to the pickle file
    with open(pickle_file_path, "wb") as pickle_file:
        pickle.dump(proposal, pickle_file)
        

