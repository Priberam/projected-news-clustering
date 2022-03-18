import signal
import torch
import json
import nltk
from sentence_transformers import SentenceTransformer
import time
import os
import sys
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import threading
import ulid
from queue import Queue
import uvicorn
from fastapi.responses import JSONResponse



class Item(BaseModel):
    id: str
    text: str
    title: str
    main_repr: Optional[str] = None
    title_repr: Optional[str] = None
    paragraph_repr: Optional[str] = None
    title_paragraph_repr: Optional[str] = None


class Response(BaseModel):
    main_repr: Optional[List[float]] = None
    title_repr: Optional[List[float]] = None
    paragraph_repr: Optional[List[float]] = None
    title_paragraph_repr: Optional[List[float]] = None




class Document:
    def __init__(self, text, ulid_request, wait_for_doc, sentences_array):
        self.text = text
        self.ulid_request = ulid_request
        self.sentences_array = sentences_array
        self.wait_for_doc = wait_for_doc
        self.representation = []


class Consumer:
    def __init__(self, queue):
        self.queue = queue

    def thread_function(self):
        while True:
            #paragraph_batch = []
            sentence_batch = []
            targets = []
            new_doc = self.queue.get()
            #paragraph_batch += new_doc.paragraph_array

            for s in new_doc.sentences_array:
                sentence_batch += s

            targets.append(new_doc)
            
            maxN = 16
            iN = 0
            try:
                while iN < maxN - 1:
                    next_doc = self.queue.get_nowait()
                    #paragraph_batch += next_doc.paragraph_array
                    for s in next_doc.sentences_array:
                        sentence_batch += s

                    targets.append(next_doc)
                    iN += 1
            except:
                pass
            
            batch_reprs = torch.Tensor(model.encode(sentence_batch, batch_size=32))
            batch_cursor = 0

            print("Amount of processed docs:" + str(len(targets)))
            print("Sentence batch size:" + str(len(sentence_batch)))


            for target in targets:
                for s in target.sentences_array:
                    paragraph_repr = batch_reprs[batch_cursor:batch_cursor + len(s)]
                    target.representation.append(torch.sum(paragraph_repr, dim=0) / len(paragraph_repr))
                    batch_cursor += len(s)

                #target.representation = batch_reprs[batch_cursor:batch_cursor + len(target.paragraph_array)]
                #batch_cursor += len(target.paragraph_array)
                target.representation = torch.stack(target.representation)
                target.wait_for_doc.set()
                print("Processed document " + target.ulid_request.str + " in queue.")


    def start_thread(self):
        self.consumer_thread = threading.Thread(target = self.thread_function)
        self.consumer_thread.start()





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
# req_i = 0

# max_paragraphs = int(os.environ["MAX_PARAGRAPHS"])
max_paragraphs = 40

# max_sentence_characters = int(os.environ["MAX_SENTENCE_CHARACTERS"])
max_sentence_characters = 4096
max_tokens = model.get_max_seq_length()





print("Starting server...")
app = FastAPI()
print("Distiluse model service started.")

queue = Queue()

consumer = Consumer(queue)
consumer.start_thread()

@app.post("/")
async def post(item: Item):
  
    ulid_request = ulid.new()

    title = item.title
    text = item.text
    response = {}
    start_time = time.time()
    paragraph_reprs = []

    # title_array = self.getParagraphs(title)
    # first_paragraph = self.getParagraphs(text, num=1)
    paragraphs_array = getParagraphs(text)
    if(len(paragraphs_array) > 0):
        paragraphs_array.insert(0, title + " \n " + paragraphs_array[0])
    else:
        paragraphs_array.append(title)
    sentences_array = [getSentences(paragraph) for paragraph in paragraphs_array]


    wait_for_doc = asyncio.Event()

    new_doc = Document(text, ulid_request, wait_for_doc, sentences_array)

    
    queue.put(new_doc)
    print("Added document " + new_doc.ulid_request.str + " to queue.")

    await new_doc.wait_for_doc.wait()

    doc_reprs = new_doc.representation


    paragraph_reprs = (
        doc_reprs[1:] if len(doc_reprs) > 1 else torch.Tensor()
    )

    title_paragraph_repr = (
        doc_reprs[0] if len(doc_reprs) > 0 else torch.Tensor()
    )


    if item.main_repr is not None:
        if(len(paragraph_reprs) > 1):
            representation = torch.mean(
                torch.cat([title_paragraph_repr.unsqueeze(0), paragraph_reprs[1:]], dim=0),
                dim=0,
            )
        else:
            representation = title_paragraph_repr
        response["main_repr"] = representation.tolist()

    if item.title_repr is not None:
        response["title_repr"] = title_repr.tolist()

    if item.paragraph_repr is not None and len(paragraph_reprs) > 0:
        response["paragraph_repr"] = paragraph_reprs[0].tolist()

    if item.title_paragraph_repr is not None and len(title_paragraph_repr) > 0:
        response["title_paragraph_repr"] = title_paragraph_repr.tolist()

    print(item.id + ": " + str(time.time() - start_time) + " s")

    return JSONResponse(content=response)


def getParagraphs(text, num=max_paragraphs):
    paragraphs = []
    par_pieces = text.split("\n")
    for par in par_pieces:
        if len(par.strip()) > 0:
            paragraphs.append(par)
            if len(paragraphs) >= num:
                return paragraphs
    return paragraphs


def getSentences(paragraph):
    sentences = [paragraph]
    n_tokens = len(model.tokenize(paragraph))
    if n_tokens > model.get_max_seq_length():
        paragraph = paragraph[:max_sentence_characters]
        sentences = nltk.sent_tokenize(paragraph)
    return sentences



def getSentenceEmbeddings(paragraph):
    sentences = [paragraph]
    n_tokens = len(model.tokenize(paragraph))
    if n_tokens > model.get_max_seq_length():
        paragraph = paragraph[:max_sentence_characters]
        sentences = nltk.sent_tokenize(paragraph)
    representation = torch.Tensor(model.encode(sentences))
    if len(representation) > 1:
        return torch.sum(representation, dim=0) / len(representation)
    return representation[0]



'''

@app.post("/", response_model=Response)
async def post(item: Item):
    # global req_i
    # req_i += 1
    # if req_i % 100 == 0:
    #    print("Reloading model on req_i: ", req_i)
    #    model = SentenceTransformer('distiluse-base-multilingual-cased')

    # id_doc = item["id"]
    title = item.title
    text = item.text
    response = Response()
    start_time = time.time()
    paragraph_reprs = []

    # title_array = self.getParagraphs(title)
    # first_paragraph = self.getParagraphs(text, num=1)
    paragraphs_array = getParagraphs(text)

    title_repr = getSentenceEmbeddings(title)

    title_paragraph_repr = getSentenceEmbeddings(
        title + "\n" + (paragraphs_array[0] if len(paragraphs_array) > 0 else "")
    )
    first_par_repr = getSentenceEmbeddings(
        paragraphs_array[0] if len(paragraphs_array) > 0 else ""
    )

    if len(paragraphs_array) > 0:
        paragraph_reprs = torch.stack(
            [getSentenceEmbeddings(x) for x in paragraphs_array]
        )

    paragraph_reprs = (
        paragraph_reprs[1:] if len(paragraph_reprs) > 1 else torch.Tensor()
    )

    if item.main_repr is not None:
        representation = torch.mean(
            torch.cat([title_paragraph_repr.unsqueeze(0), paragraph_reprs], dim=0),
            dim=0,
        )
        response.main_repr = representation.tolist()

    if item.title_repr is not None:
        response.title_repr = title_repr.tolist()

    if item.paragraph_repr is not None and len(first_par_repr) > 0:
        response.paragraph_repr = first_par_repr.tolist()

    # if("paragraph_repr" in features and len(first_paragraph) > 0):
    #    response["paragraph_repr"] = first_par_repr.tolist()

    if item.title_paragraph_repr is not None and len(title_paragraph_repr) > 0:
        response.title_paragraph_repr = title_paragraph_repr.tolist()

    print(item.id + ": " + str(time.time() - start_time))
    return response
'''
