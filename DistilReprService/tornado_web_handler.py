import tornado
import tornado.web
import signal
import torch
import json
import nltk
from sentence_transformers import SentenceTransformer
import time
import os
import sys
import asyncio
from queue import Queue
import threading
import ulid



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


batch_size = 32





class Server(tornado.web.RequestHandler):

    def post(self):
        # global req_i
        # req_i += 1
        # if req_i % 100 == 0:
        #    print("Reloading model on req_i: ", req_i)
        #    model = SentenceTransformer('distiluse-base-multilingual-cased')

        request_ulid = ulid.new()

        jbody = json.loads(self.request.body.decode("utf-8"))
        # id_doc = jbody["id"]
        title = jbody["title"]
        text = jbody["text"]
        features = jbody.keys()
        response = {}
        start_time = time.time()
        paragraph_reprs = []

        # title_array = self.getParagraphs(title)
        # first_paragraph = self.getParagraphs(text, num=1)
        paragraphs_array = self.getParagraphs(text)

        title_repr = self.getSentenceEmbeddings(title)

        print(title_repr)

        title_paragraph_repr = self.getSentenceEmbeddings(
            title + "\n" + (paragraphs_array[0] if len(paragraphs_array) > 0 else "")
        )
        first_par_repr = self.getSentenceEmbeddings(
            paragraphs_array[0] if len(paragraphs_array) > 0 else ""
        )

        if len(paragraphs_array) > 0:
            paragraph_reprs = torch.stack(
                [self.getSentenceEmbeddings(x) for x in paragraphs_array]
            )

        paragraph_reprs = (
            paragraph_reprs[1:] if len(paragraph_reprs) > 1 else torch.Tensor()
        )

        if "main_repr" in features:
            representation = torch.mean(
                torch.cat([title_paragraph_repr.unsqueeze(0), paragraph_reprs], dim=0),
                dim=0,
            )
            response["main_repr"] = representation.tolist()

        if "title_repr" in features:
            response["title_repr"] = title_repr.tolist()

        if "paragraph_repr" in features and len(first_par_repr) > 0:
            response["paragraph_repr"] = first_par_repr.tolist()

        # if("paragraph_repr" in features and len(first_paragraph) > 0):
        #    response["paragraph_repr"] = first_par_repr.tolist()

        if "title_paragraph_repr" in features and len(title_paragraph_repr) > 0:
            response["title_paragraph_repr"] = title_paragraph_repr.tolist()

        print(jbody["id"] + ": " + str(time.time() - start_time))
        self.write(json.dumps(response))




    async def conc_post(self):
        ulid_request = ulid.new()

        jbody = json.loads(self.request.body.decode("utf-8"))
        title = jbody["title"]
        text = jbody["text"]
        features = jbody.keys()
        response = {}
        start_time = time.time()

        paragraphs_array = self.getParagraphs(title + "\n" + text)

        new_doc = Document(text, ulid_request, paragraphs_array)
        wait_for_doc = asyncio.Event()

        await producer.addToQueue(new_doc)

        #check if storage_doc_temp is already filled (another process has already called the consumer?)
        #if not, call consumer and await wait_for_doc.wait()?

        await consumer.processQueue(wait_for_doc)

        await wait_for_doc.wait()

        doc_reprs = new_doc.representation



        title_repr = (
            doc_reprs[0].unsqueeze(0) if len(doc_reprs) > 0 else torch.Tensor()
        )

        paragraph_reprs = (
            doc_reprs[1:] if len(doc_reprs) > 1 else torch.Tensor()
        )

        title_paragraph_repr = torch.mean(
                torch.cat([doc_reprs[0].unsqueeze(0), doc_reprs[1].unsqueeze(0)], dim=0),
                dim=0,
        )


        if "main_repr" in features:
            representation = torch.mean(
                doc_reprs,
                dim=0,
            )
            response["main_repr"] = representation.tolist()

        if "title_repr" in features:
            response["title_repr"] = title_repr.tolist()

        if "paragraph_repr" in features and len(paragraph_reprs) > 0:
            response["paragraph_repr"] = paragraph_reprs[0].tolist()

        # if("paragraph_repr" in features and len(first_paragraph) > 0):
        #    response["paragraph_repr"] = first_par_repr.tolist()

        if "title_paragraph_repr" in features and len(title_paragraph_repr) > 0:
            response["title_paragraph_repr"] = title_paragraph_repr.tolist()


        print(jbody["id"] + ": " + str(time.time() - start_time))
        self.write(json.dumps(response))


    def getParagraphs(self, text, num=max_paragraphs):
        paragraphs = []
        par_pieces = text.split("\n")
        for par in par_pieces:
            if len(par.strip()) > 0:
                paragraphs.append(par)
                if len(paragraphs) >= num:
                    return paragraphs
        return paragraphs

    def getSentenceEmbeddings(self, paragraph):
        sentences = [paragraph]
        n_tokens = len(model.tokenize(paragraph))
        if n_tokens > model.get_max_seq_length():
            paragraph = paragraph[:max_sentence_characters]
            sentences = nltk.sent_tokenize(paragraph)
        representation = torch.Tensor(model.encode(sentences))
        if len(representation) > 1:
            return torch.sum(representation, dim=0) / len(representation)
        return representation[0]








# equivalent to a struct; have the data organized
class Document:
    def __init__(self, text, ulid_request, paragraphs):
        self.text = text
        self.ulid_request = ulid_request
        self.paragraph_array = paragraphs
        self.representation = []



class Producer:
    def __init__(self, queue):
        self.queue = queue

    async def addToQueue(self, document):
        await self.queue.put(document)
        print("Added document " + document.ulid_request.str + " to queue.")


class Consumer:
    def __init__(self, queue):
        self.queue = queue

    async def processQueue(self, wait_for_doc):
        representations = []
        target_doc = await self.queue.get()
        target_doc.representation = torch.Tensor(model.encode(target_doc.paragraph_array))
        wait_for_doc.set()
        print("Processed document " + target_doc.ulid_request.str + " in queue.")



def make_app():
    return tornado.web.Application(
        [
            (r"/", Server),
        ]
    )

queue = asyncio.Queue()

producer = Producer(queue)
consumer = Consumer(queue)

producer_thread = threading.Thread()
consumer_thread = threading.Thread()

producer_thread.start()
consumer_thread.start()

producer_thread.join()
consumer_thread.join()


print("Starting server...")
app = make_app()
app.listen(8050)
print("Distiluse model service started.")
tornado.ioloop.IOLoop.instance().start()



'''

    def post(self):
        # global req_i
        # req_i += 1
        # if req_i % 100 == 0:
        #    print("Reloading model on req_i: ", req_i)
        #    model = SentenceTransformer('distiluse-base-multilingual-cased')

        request_ulid = ulid.new()

        jbody = json.loads(self.request.body.decode("utf-8"))
        # id_doc = jbody["id"]
        title = jbody["title"]
        text = jbody["text"]
        features = jbody.keys()
        response = {}
        start_time = time.time()
        paragraph_reprs = []

        # title_array = self.getParagraphs(title)
        # first_paragraph = self.getParagraphs(text, num=1)
        paragraphs_array = self.getParagraphs(text)

        title_repr = self.getSentenceEmbeddings(title)

        title_paragraph_repr = self.getSentenceEmbeddings(
            title + "\n" + (paragraphs_array[0] if len(paragraphs_array) > 0 else "")
        )
        first_par_repr = self.getSentenceEmbeddings(
            paragraphs_array[0] if len(paragraphs_array) > 0 else ""
        )

        if len(paragraphs_array) > 0:
            paragraph_reprs = torch.stack(
                [self.getSentenceEmbeddings(x) for x in paragraphs_array]
            )

        paragraph_reprs = (
            paragraph_reprs[1:] if len(paragraph_reprs) > 1 else torch.Tensor()
        )

        if "main_repr" in features:
            representation = torch.mean(
                torch.cat([title_paragraph_repr.unsqueeze(0), paragraph_reprs], dim=0),
                dim=0,
            )
            response["main_repr"] = representation.tolist()

        if "title_repr" in features:
            response["title_repr"] = title_repr.tolist()

        if "paragraph_repr" in features and len(first_par_repr) > 0:
            response["paragraph_repr"] = first_par_repr.tolist()

        # if("paragraph_repr" in features and len(first_paragraph) > 0):
        #    response["paragraph_repr"] = first_par_repr.tolist()

        if "title_paragraph_repr" in features and len(title_paragraph_repr) > 0:
            response["title_paragraph_repr"] = title_paragraph_repr.tolist()

        print(jbody["id"] + ": " + str(time.time() - start_time))
        self.write(json.dumps(response))


'''