import argparse
import warnings
import os
import re
import torch
import sys
# sys.path.insert(0, r'C:\\Users\\12716\Desktop\\model')
# from server.settings import BASE_DIR


def entity_index(path):
    index_entity = {}
    with open(path, encoding='utf-8') as file_obj:
        for line in file_obj:
            line = line.strip()
            line = line.split(",")
            index_entity[int(line[0])] = line[1]
    return index_entity


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(
        description="""TCMPG-GAE for prescriptions generation""")
    parser.add_argument('-device', type=str,
                        help="choose the cpu or cuda", default="cpu")
    parser.add_argument('-symptoms', type=str,
                        help="input the symptoms.")
    parser.add_argument('-filepath', type=str,
                        help="input the symptoms.")
    args = parser.parse_args()
    device = args.device
    symptoms = args.symptoms
    filepath = args.filepath

    # index_herb = entity_index(path='./data/index_herb')
    # index_symptom = entity_index(path='./data/index_symptom')
    index_herb = entity_index(path=os.path.join(filepath, 'server/data/index_herb'))
    index_symptom = entity_index(path=os.path.join(filepath, 'server/data/index_symptom'))
    symptom_index = dict(zip(index_symptom.values(), index_symptom.keys()))

    if device == 'cpu':
        mymodel = torch.load(os.path.join(filepath, 'server/data/best_model'),
                           map_location=torch.device('cpu'))
        symptom_emb = torch.load(
            os.path.join(filepath, 'server/data/best_symptom_emb.pt'), map_location=torch.device('cpu'))

    symptoms = symptoms.split(' ')
    test_symptoms_index = []
    for symptom in symptoms:
        if symptom != "":
            test_symptoms_index.append(symptom_index[symptom])


    test_emb_p = torch.sum(symptom_emb[test_symptoms_index], dim=0)
    test_emb_p = mymodel(test_emb_p)

    values = torch.topk(test_emb_p, 10).values
    indices = torch.topk(test_emb_p, 10).indices
    
    prescriptions = ""
    for h in range(10):
        dose = values[h]/torch.sum(values)
        dose = round(dose.item() * 100, 1)
        if h == 0:
            prescriptions += index_herb[int(indices[h])] + " " + str(dose) + "%"
        else:
            prescriptions += "," + index_herb[int(indices[h])] + " " + str(dose) + "%"
    print(prescriptions)

