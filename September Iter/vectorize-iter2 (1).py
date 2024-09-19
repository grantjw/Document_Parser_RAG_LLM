import logging
import time
import pickle
from uuid import uuid4
from typing import List
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores.utils import node_to_metadata_dict
import sys

#change collection name
milvus_name = "_gk_2024_Q2_Buyout"
#pickle name change
FILENAME = 'simple_dir_data_2024_Q2_Buyout.pickle'

# Constants 
MILVUS_HOST = "stepstone-milvus.milvus.svc.cluster.local"
MILVUS_PORT = 19530
DIM = 1024
COLLECTION_NAME = f"quarterly_reports_{DIM}{milvus_name}"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 20
EMBED_MODEL = "Alibaba-NLP/gte-large-en-v1.5"
EMBED_BATCH_SIZE = 10
LOG_LEVEL = "INFO"
DIRECTORY_PATH = '/home/jovyan/shared/projects/Sep_Iter/Data/Simple_Dir_Data/'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s %(threadName)s : %(message)s')
logger = logging.getLogger(__name__)
print(sys.executable)  # Output the path of the current Python executable

# Custom vector store class
class CakeAiMilvusVectorStore(MilvusVectorStore):
    def add(self, nodes: List['BaseNode']) -> List[str]:
        insert_list = []
        insert_ids = []  # Collecting IDs of inserted nodes
        for node in nodes:
            entry = node_to_metadata_dict(node)
            entry["id"] = uuid4().hex
            entry[self.embedding_field] = node.embedding
            insert_list.append(entry)

        try:
            print(f"inserting into milvus collection: {self.collection_name}")
            self._milvusclient.insert(collection_name=self.collection_name, data=insert_list, batch_size=EMBED_BATCH_SIZE, progress_bar=True, timeout=None)
        except Exception as e:
            logging.error("Milvus insert exception:" + str(e), exc_info=True)
            time.sleep(5)
            raise e

        return insert_ids

# Configure vector store
vector_store = CakeAiMilvusVectorStore(
    overwrite=False,
    doc_id_field="id_",
    uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}",
    collection_name=COLLECTION_NAME,
    dim=DIM,
)

# Settings for processing
Settings.node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBED_MODEL,
    max_length=CHUNK_SIZE,  # Ensure this matches the required length
    embed_batch_size=EMBED_BATCH_SIZE,
    trust_remote_code=True
)

# Configure storage context with the custom vector store
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Vectorizer initialization and operation
class Vectorizer:
    def __init__(self):
        logging.info("Vectorizer Actor created")
        self.vector_store = vector_store

    def vectorize(self, documents):
        logging.info("vectorize: start")
        try:
            total_docs = len(documents)
            VectorStoreIndex.from_documents(documents, storage_context=storage_context)
            logging.info("vectorize: successfully processed all documents")
        except Exception as e:
            logging.exception(f"vectorize: encountered an error: {e}")
            raise e
        return total_docs

# Load documents from pickle
try:
    with open(DIRECTORY_PATH + FILENAME, 'rb') as file:
        documents = pickle.load(file)
        logger.info('Data has been successfully loaded.')
except Exception as e:
    logger.error(f"Could not load pickled data: {e}")

# Example usage
vectorizer = Vectorizer()
total_processed = vectorizer.vectorize(documents)
print(f"Total documents processed: {total_processed}")
