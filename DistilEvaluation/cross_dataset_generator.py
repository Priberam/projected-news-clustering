from dataset_loader import LoadLinkingDataset
import json

dataset = LoadLinkingDataset("G:\\Corpora\\clustering\\original",["deu","eng","spa"])
static_dataset_f = open("static_dataset.dev","w",encoding="utf8")
gold_dataset_path = "G:\\Corpora\\clustering\\processed_clusters\\dataset.test.json"

mono_clusters = {}


for k in dataset["linking"].keys():
    for b_id in dataset["linking"][k]:
        connection_type = dataset["linking"][k][b_id]
        for art_k in dataset["bags"][k]["articles"]:
            for art_b in dataset["bags"][b_id]["articles"]:
                static_dataset_f.write(art_k + "\t" + art_b + "\t" + connection_type + "\n")


with open(gold_dataset_path, "r", encoding="utf8") as json_file:
    data = json.load(json_file)
    for doc in data:
        if(doc["cluster"] not in mono_clusters.keys()):
            mono_clusters[doc["cluster"]] = []
        mono_clusters[doc["cluster"]].append(doc["id"])

for cluster in mono_clusters.keys():
    for doc in mono_clusters[cluster]:
        for target_doc in mono_clusters[cluster]:
            if(doc == target_doc):
                continue
            static_dataset_f.write(doc + "\t" + target_doc + "\t" + "positive" + "\n")
    
static_dataset_f.close()