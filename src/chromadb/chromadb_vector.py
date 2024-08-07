# import json
# from typing import List, Dict

# import chromadb
# import pandas as pd
# from chromadb.config import Settings
# from chromadb.utils import embedding_functions

# from ..base import PlugBase
# from ..utils import deterministic_uuid

# default_ef = embedding_functions.DefaultEmbeddingFunction()


# class ChromaDB_VectorStore(PlugBase):
#     def __init__(self, config=None):
#         PlugBase.__init__(self, config=config)
#         if config is None:
#             config = {}

#         self.path = config.get("path", ".")
#         self.embedding_function = config.get("embedding_function", default_ef)
#         self.curr_client = config.get("client", "persistent")
#         self.collection_metadata = config.get("collection_metadata", None)
        
#         # Initialize collections dictionary
#         self.collections = config.get("collections", {})

#         self._initialize_client()
#         self._initialize_collections()

#     def _initialize_client(self):
#         if self.curr_client == "persistent":
#             self.chroma_client = chromadb.PersistentClient(
#                 path=self.path, settings=Settings(anonymized_telemetry=False)
#             )
#         elif self.curr_client == "in-memory":
#             self.chroma_client = chromadb.EphemeralClient(
#                 settings=Settings(anonymized_telemetry=False)
#             )
#         elif isinstance(self.curr_client, chromadb.api.client.Client):
#             self.chroma_client = self.curr_client
#         else:
#             raise ValueError(f"Unsupported client was set in config: {self.curr_client}")

#     def _initialize_collections(self):
#         self.collection_objects = {}
#         for key, collection_info in self.collections.items():
#             self.collection_objects[key] = self.chroma_client.get_or_create_collection(
#                 name=collection_info["name"],
#                 embedding_function=self.embedding_function,
#                 metadata=self.collection_metadata,
#             )

#     def add_collection(self, key: str, name: str, n_results: int = 10):
#         if key in self.collections:
#             raise ValueError(f"Collection key '{key}' already exists")
#         self.collections[key] = {"name": name, "n_results": n_results}
#         self.collection_objects[key] = self.chroma_client.get_or_create_collection(
#             name=name,
#             embedding_function=self.embedding_function,
#             metadata=self.collection_metadata,
#         )

#     def remove_collection(self, key: str) -> bool:
#         if key not in self.collections:
#             return False
#         collection_name = self.collections[key]["name"]
#         self.chroma_client.delete_collection(name=collection_name)
#         del self.collections[key]
#         del self.collection_objects[key]
#         return True

#     def generate_embedding(self, data: str, **kwargs) -> List[float]:
#         embedding = self.embedding_function([data])
#         if len(embedding) == 1:
#             return embedding[0]
#         return embedding

#     def add_item(self, collection_key: str, content: str, content_type: str, **kwargs) -> str:
#         if collection_key not in self.collection_objects:
#             raise ValueError(f"Unknown collection key: {collection_key}")

#         if collection_key == "sql":
#             question = kwargs.get("question")
#             if not question:
#                 raise ValueError("Question is required for SQL collection")
#             content_json = json.dumps({"question": question, "sql": content}, ensure_ascii=False)
#         else:
#             content_json = content

#         id = deterministic_uuid(content_json) + f"-{content_type}"
#         self.collection_objects[collection_key].add(
#             documents=content_json,
#             embeddings=self.generate_embedding(content_json),
#             ids=id,
#         )
#         return id

#     def get_training_data(self, **kwargs) -> pd.DataFrame:
#         df = pd.DataFrame()
#         for collection_key, collection in self.collection_objects.items():
#             data = collection.get()
#             if data is not None:
#                 documents = data["documents"]
#                 ids = data["ids"]
                
#                 if collection_key == "sql":
#                     documents = [json.loads(doc) for doc in documents]
#                     df_collection = pd.DataFrame({
#                         "id": ids,
#                         "question": [doc["question"] for doc in documents],
#                         "content": [doc["sql"] for doc in documents],
#                     })
#                 else:
#                     df_collection = pd.DataFrame({
#                         "id": ids,
#                         "question": [None for _ in documents],
#                         "content": documents,
#                     })
                
#                 df_collection["training_data_type"] = collection_key
#                 df = pd.concat([df, df_collection])
        
#         return df

#     def remove_training_data(self, id: str, **kwargs) -> bool:
#         for collection in self.collection_objects.values():
#             try:
#                 collection.delete(ids=id)
#                 return True
#             except:
#                 pass
#         return False

#     @staticmethod
#     def _extract_documents(query_results) -> list:
#         if query_results is None:
#             return []

#         if "documents" in query_results:
#             documents = query_results["documents"]

#             if len(documents) == 1 and isinstance(documents[0], list):
#                 try:
#                     documents = [json.loads(doc) for doc in documents[0]]
#                 except Exception as e:
#                     return documents[0]

#             return documents

#     def get_related_items(self, collection_key: str, question: str, **kwargs) -> list:
#         if collection_key not in self.collection_objects:
#             raise ValueError(f"Unknown collection key: {collection_key}")

#         n_results = self.collections[collection_key].get("n_results", 10)
#         return ChromaDB_VectorStore._extract_documents(
#             self.collection_objects[collection_key].query(
#                 query_texts=[question],
#                 n_results=n_results,
#             )
#         )

#     def list_collections(self) -> Dict[str, Dict[str, str]]:
#         return {key: {"name": info["name"]} for key, info in self.collections.items()}

import json
from typing import List, Dict

import chromadb
import pandas as pd
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from src.base import PlugBase
from src.utils import deterministic_uuid  # Make sure this is correctly imported

default_ef = embedding_functions.DefaultEmbeddingFunction()

class ChromaDB_VectorStore(PlugBase):
    def __init__(self, config=None):
        PlugBase.__init__(self, config=config)
        if config is None:
            config = {}

        self.path = config.get("path", ".")
        self.embedding_function = config.get("embedding_function", default_ef)
        self.curr_client = config.get("client", "persistent")
        self.collection_metadata = config.get("collection_metadata", None)
        
        # Initialize collections dictionary
        self.collections = config.get("collections", {})

        self._initialize_client()
        self._initialize_collections()

    def _initialize_client(self):
        if self.curr_client == "persistent":
            self.chroma_client = chromadb.PersistentClient(
                path=self.path, settings=Settings(anonymized_telemetry=False)
            )
        elif self.curr_client == "in-memory":
            self.chroma_client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False)
            )
        elif isinstance(self.curr_client, chromadb.api.client.Client):
            self.chroma_client = self.curr_client
        else:
            raise ValueError(f"Unsupported client was set in config: {self.curr_client}")

    def _initialize_collections(self):
        self.collection_objects = {}
        for key, collection_info in self.collections.items():
            self.collection_objects[key] = self.chroma_client.get_or_create_collection(
                name=collection_info["name"],
                embedding_function=self.embedding_function,
                metadata=self.collection_metadata,
            )

    def add_collection(self, key: str, name: str, n_results: int = 10):
        if key in self.collections:
            raise ValueError(f"Collection key '{key}' already exists")
        self.collections[key] = {"name": name, "n_results": n_results}
        self.collection_objects[key] = self.chroma_client.get_or_create_collection(
            name=name,
            embedding_function=self.embedding_function,
            metadata=self.collection_metadata,
        )

    def remove_collection(self, key: str) -> bool:
        if key not in self.collections:
            return False
        collection_name = self.collections[key]["name"]
        self.chroma_client.delete_collection(name=collection_name)
        del self.collections[key]
        del self.collection_objects[key]
        return True

    def generate_embedding(self, data: str, **kwargs) -> List[float]:
        embedding = self.embedding_function([data])
        if len(embedding) == 1:
            return embedding[0]
        return embedding

    def add_item(self, collection_key: str, content: str, content_type: str, **kwargs) -> str:
        if collection_key not in self.collection_objects:
            raise ValueError(f"Unknown collection key: {collection_key}")

        collection_info = self.collections[collection_key]
        content_json = self._prepare_content(collection_info, content, content_type, **kwargs)

        id = deterministic_uuid(content_json) + f"-{content_type}"
        self.collection_objects[collection_key].add(
            documents=content_json,
            embeddings=self.generate_embedding(content_json),
            ids=id,
        )
        return id

    def _prepare_content(self, collection_info: Dict, content: str, content_type: str, **kwargs) -> str:
        content_heads = collection_info.get("content_heads", [])
        content_dict = {head: kwargs.get(head) for head in content_heads}
        content_dict[content_type] = content
        return json.dumps(content_dict, ensure_ascii=False)

    def get_training_data(self, **kwargs) -> pd.DataFrame:
        df = pd.DataFrame()
        for collection_key, collection in self.collection_objects.items():
            data = collection.get()
            if data is not None:
                documents = data["documents"]
                ids = data["ids"]
                
                df_collection = self._process_collection_data(collection_key, documents, ids)
                df = pd.concat([df, df_collection])
        
        return df

    def _process_collection_data(self, collection_key: str, documents: List[str], ids: List[str]) -> pd.DataFrame:
        collection_info = self.collections[collection_key]
        content_heads = collection_info.get("content_heads", [])
        content_heads.append(collection_info.get("main_content_type", "content"))

        documents = [json.loads(doc) for doc in documents]
        df_collection = pd.DataFrame({
            "id": ids,
            **{head: [doc.get(head) for doc in documents] for head in content_heads}
        })
        
        df_collection["training_data_type"] = collection_key
        return df_collection

    def remove_training_data(self, id: str, **kwargs) -> bool:
        for collection in self.collection_objects.values():
            try:
                collection.delete(ids=id)
                return True
            except:
                pass
        return False

    @staticmethod
    def _extract_documents(query_results) -> list:
        if query_results is None:
            return []

        if "documents" in query_results:
            documents = query_results["documents"]

            if len(documents) == 1 and isinstance(documents[0], list):
                try:
                    documents = [json.loads(doc) for doc in documents[0]]
                except Exception as e:
                    return documents[0]

            return documents

    def get_related_items(self, collection_key: str, question: str, **kwargs) -> list:
        if collection_key not in self.collection_objects:
            raise ValueError(f"Unknown collection key: {collection_key}")

        n_results = self.collections[collection_key].get("n_results", 10)
        return ChromaDB_VectorStore._extract_documents(
            self.collection_objects[collection_key].query(
                query_texts=[question],
                n_results=n_results,
            )
        )
        
    def get_focused_related_items(self, collection_key: str, question: str, focus: str, n_results: int = 10, **kwargs) -> list:
        if collection_key not in self.collection_objects:
            raise ValueError(f"Unknown collection key: {collection_key}")

        collection_info = self.collections[collection_key]
        content_heads = collection_info.get("content_heads", [])
        
        if focus not in content_heads and focus != collection_info.get("main_content_type"):
            raise ValueError(f"Invalid focus: {focus}. Must be one of the content heads or main_content_type.")

        # Prepare the focused query
        focused_query = self._prepare_focused_query(question, focus)

        # Query the collection with the focused query
        query_results = self.collection_objects[collection_key].query(
            query_texts=[focused_query],
            n_results=n_results
        )

        # Extract and process the documents
        documents = ChromaDB_VectorStore._extract_documents(query_results)
        
        # Extract the focused content from each document
        focused_documents = [
            {
                **doc,
                "focused_content": doc.get(focus, ""),
                "relevance_score": self._calculate_relevance_score(doc.get(focus, ""), question)
            }
            for doc in documents
        ]

        # Sort the documents based on the relevance score
        sorted_documents = sorted(
            focused_documents,
            key=lambda x: x["relevance_score"],
            reverse=True
        )

        return sorted_documents
    
    def _prepare_focused_query(self, question: str, focus: str) -> str:
        return f"Regarding {focus}: {question}"

    def _calculate_relevance_score(self, focused_content: str, question: str) -> float:
        # This is a simple relevance calculation. You might want to use a more sophisticated method.
        question_words = set(question.lower().split())
        content_words = set(focused_content.lower().split())
        common_words = question_words.intersection(content_words)
        return len(common_words) / len(question_words)

    def list_collections(self) -> Dict[str, Dict[str, str]]:
        return {key: {"name": info["name"]} for key, info in self.collections.items()}