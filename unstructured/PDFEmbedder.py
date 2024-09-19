from langchain_openai import AzureChatOpenAI
from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from typing import Optional, Any, List
from unstructured import partition
from unstructured.partition import auto, api
from unstructured.staging.base import elements_to_dicts, elements_from_base64_gzipped_json
import pandas as pd
# from llama_index.core.response.notebook_utils import display_source_node

class ProcessPDF:
    def __init__(
        self,
        llm: AzureChatOpenAI
    ) -> None:
        """Init params."""
        self._LLM = llm

    
    # Extract images with tables and charts and generate text representations of them.
    def extract_image(self, base64_image):
        messages=[
            # {"role": "system", "content": "You are a Financial Analyst."},
            {"role": "user", "content": [
                {"type": "text", "text": "Given the following image, if it is a table, please return a paragraph capturing all of the information in the table, in sentence form.  If it is a Brand Logo, please return any text that can be extracted from the logo (if there is no text, return whitespace).  If it is neither, return whitespace.  Do not return anything else except the paragraph, the text, or the whitespace."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"}
                }
            ]}
        ]
        ai_message = self._LLM.invoke(messages)
        return ai_message.content

    
    # Extract tables as html and generate text representations of them.
    def extract_table(self, table_as_html):
        messages=[
                # {"role": "system", "content": "You are a Financial Analyst."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Given the following html markup table, please return a paragraph capturing all of the information in the table, in sentence form. If there is no table, return whitespace.  Do not return anything else except the paragraph or the whitespace."},
                    {"type": "text", "text": f"{table_as_html}"}
                ]}
            ]
        ai_message = self._LLM.invoke(messages)
        return ai_message.content

    
    # Generate JSON of all text, images, and tables in pdf
    def partition_file_via_open_source(self, filename):
        print(f"processing {filename} using local library...", end="", flush=True)
        partition_params = {
            "filename": str(filename),
            "strategy": ["auto","hi_res","fast","ocr_only"][1],
            "hi_res_model_name": ["yolox","layout_v1.1.0"][0],
            "extract_image_block_types": ["Image", "Table"],
            "skip_infer_table_types": [],
            "extract_image_block_to_payload": True,
            # "coordinates": True
        }
        try:
            return elements_to_dicts(partition.auto.partition(**partition_params))
        except Exception as e:
            print(f"Failed to parse file due to error: {e}")

    
    # Create a Pandas DataFrame of the JSONified pdf doc, replacing images and tables with paragraphs
    def create_dataframe(self, element_dict):
        df = pd.DataFrame.from_dict(element_dict)
        df["page_number"] = df["metadata"].apply(lambda x: x["page_number"])
        df["extracted_text"] = df["metadata"].apply(lambda x: self.extract_table(x["text_as_html"]) if "text_as_html" in x else "")
        df["extracted_text"] = df["metadata"].apply(lambda x: self.extract_image(x["image_base64"]) if "image_base64" in x else "")
        df["extracted_text"] = df["extracted_text"].apply(lambda x: "" if x=="whitespace" else x)
        return df

    
    # Create a LIST of String representations of the pages in the pdf
    def create_block_text(self, element_df):
        element_df["combined_text"] = element_df["text"] + "[[" + element_df["extracted_text"] + "]]"
        element_df["combined_text"] = element_df["combined_text"].apply(lambda x: x.replace("[[]]", ""))
        last_page = element_df["page_number"].max()
        block_text = []
        for i in range(1, last_page+1):
            block_text.append(element_df[element_df["page_number"]==i]["combined_text"].str.cat(sep='\n'))
        return block_text

class VectorDBLoader:
    def __init__(
        self,
        vector_store: MilvusVectorStore,
        embed_model: Any
    ) -> None:
        """Init params."""
        self._VECTOR_STORE = vector_store
        self._EMBED_MODEL = embed_model

    def addToVectorDB(self, text_chunks, metadata=None):
        nodes = []
        for text_chunk in text_chunks:
            node = TextNode(
                text=text_chunk,
            )
            nodes.append(node)

        for node in nodes:
            node_embedding = self._EMBED_MODEL.get_text_embedding(
                node.get_content(metadata_mode="all")
            )
            node.embedding = node_embedding
            if metadata is not None:
                node.metadata = metadata

        self._VECTOR_STORE.add(nodes)
        # print(nodes)


class VectorDBRetriever(BaseRetriever):
    """Retriever over a Milvus vector store."""

    def __init__(
        self,
        vector_store: MilvusVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        """Init params."""
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        if query_bundle.embedding is None:
            query_embedding = self._embed_model.get_query_embedding(
                query_bundle.query_str
            )
        else:
            query_embedding = query_bundle.embedding

        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = self._vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        # return self._custom_filter(nodes_with_scores)
        return nodes_with_scores


    def _custom_filter(self, nodes_with_scores: List[NodeWithScore]) -> List[NodeWithScore]:
        all_nodes: Dict[str, NodeWithScore] = {}
        file_name_counts: Dict[str, int] = defaultdict(int)

        for node_with_score in nodes_with_scores:
            file_name = node_with_score.node.metadata["file_name"]
            if file_name_counts[file_name] < 5:
                all_nodes[node_with_score.node.get_content()] = node_with_score
                file_name_counts[file_name] += 1

        return sorted(all_nodes.values(), key=lambda x: x.score or 0.0, reverse=True)[:20]


