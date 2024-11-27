from transformers import BertTokenizerFast
from transformers import pipeline, AutoModelForTokenClassification, BertTokenizerFast 
import os
import torch
import sys
import json
import argparse
import pathlib

MODEL_PATH="./model"
bokmal_label="B"
nynorsk_label="N"

os.environ["TOKENIZERS_PARALLELISM"]="false"
enc_max_length=512

int_segmentation_device=-1  # -1 for cpu or gpu id
segmentation_device="cpu"

if torch.cuda.is_available():
    if torch.cuda.device_count()>0:
        int_segmentation_device=0
        segmentation_device="cuda:0"        

segmentation_tokenizer=None
segmentation_model=None
segmentation_classifier=None
batch_size=8

model_config=json.load(open(MODEL_PATH+"/config.json","r"))
ID2LABEL=model_config["id2label"]
ID2LABEL={i:"bm" if ID2LABEL[i]==bokmal_label else "nn" if ID2LABEL[i]==nynorsk_label else "" for i in ID2LABEL}

def load_model():
    global int_segmentation_device
    global segmentation_device
    global segmentation_tokenizer
    global segmentation_model
    global segmentation_classifier
    global int_segmentation_device
    global MODEL_PATH

    segmentation_tokenizer =  BertTokenizerFast.from_pretrained('NbAiLab/nb-bert-base')
    segmentation_tokenizer.model_max_length=512

    segmentation_model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    segmentation_model.to(segmentation_device)
    segmentation_model.eval()

    segmentation_classifier = pipeline("ner", model=segmentation_model, tokenizer=segmentation_tokenizer, device=int_segmentation_device)

    torch.no_grad()

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def identify_language_single(text):
    global segmentation_classifier
    global segmentation_tokenizer
    global segmentation_device
    global segmentation_model
    global model_config
    global bokmal_label
    global nynorsk_label
    global ID2LABEL
    num_bm=0
    num_nn=0
    content=[segmentation_classifier(i.strip().replace("\n", " "), aggregation_strategy="max", ignore_labels=[]) for i in text.split("\n\n") if i.strip()!=""]
    num_bm=sum([sum([ 1 for j in i if j["entity_group"]==bokmal_label]) for i in content])
    num_nn=sum([sum([ 1 for j in i if j["entity_group"]==nynorsk_label]) for i in content])
    return ID2LABEL[str(model_config["label2id"][bokmal_label])] if num_bm>=num_nn else ID2LABEL[str(model_config["label2id"][nynorsk_label])]

def identify_language(one_batch):
    global segmentation_classifier
    global segmentation_tokenizer
    global segmentation_device
    global int_segmentation_device
    global segmentation_model
    global ID2LABEL

    num_bm=0
    num_nn=0
    encoded_input = segmentation_tokenizer(one_batch,padding=True, truncation=True, max_length=512,  return_tensors="pt").to(segmentation_model.device)
    outputs = segmentation_model(**encoded_input)
    labels = outputs.logits.argmax(-1)
    non_zeros=[torch.nonzero(labels[i]).size()[0] for i in range(len(labels))]
    sums=torch.sum(labels, dim=1)

    # Remove allocated GPU memory
    if int_segmentation_device!=-1:
        encoded_input=encoded_input.to("cpu")
        labels=labels.to("cpu")
        torch.cuda.empty_cache()

    return [ ID2LABEL["2"] if sums[i]/non_zeros[i]>=1.5 else ID2LABEL["1"]  for i in range(len(sums))]

def get_file_content(f):
    c=None
    with open(f,"r") as ff:
        c=ff.read(3000).replace("\n"," ").replace("\r","")
    return c

def get_file_content_full(f):
    c=None
    with open(f,"r") as ff:
        c=ff.read().replace("\n"," ").replace("\r","")
    return c

def main():
    global use_single_new_line
    global use_only_beginning
    global batch_size
 
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", dest="filename",
                    help="single file to process", metavar="FILE")
    parser.add_argument("-d", "--dir", dest="dirname",
                    help="directory to process each file in it recursively. This option uses only the beginning of each file in identification.", metavar="FILE")
    parser.add_argument('-b','--batch-size', action="store", default="8",type=str, required=False, help='Batch size for the GPU tasks. Only processed in -o mode.')

    args = parser.parse_args()

    if args.batch_size is not None:
        try:
            batch_size=int(args.batch_size)
        except:
            pass

    if args.filename is not None:
        if os.path.isfile(args.filename):
            load_model()
            lang="err"
            try:
                lang=identify_language_single(get_file_content_full(args.filename))
            except:
                pass
            print(args.filename + "\t" + lang)
        else:
            eprint("The file " + args.filename + " could not be found.")
            exit(1)

    elif args.dirname is not None:
        files = pathlib.Path(args.dirname)
        model_loaded=False
        b=0
        my_batch=[]
        for f in files.iterdir():
            if not model_loaded:
                model_loaded = True
                load_model()
            if b == batch_size:
                langs=identify_language([i["content"] for i in my_batch])
                for i in range(len(my_batch)):
                    print(my_batch[i]["f_name"] + "\t" + langs[i])
                my_batch=[]
                b=0
            my_batch.append({"f_name":str(f), "content":get_file_content(f) })
            b+=1
        if b>0:
            langs=identify_language([i["content"] for i in my_batch])
            for i in range(len(my_batch)):
                print(my_batch[i]["f_name"] + "\t" + langs[i])
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
