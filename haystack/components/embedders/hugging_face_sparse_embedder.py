from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document, SparseEmbedding
from typing import List, Dict, Optional, Union
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np

@component
class HuggingFaceSparseTextEmbedder:
    """
    HuggingFaceSparseTextEmbedder computes sparse embeddings using SPLADE models.
    
    Usage example:
    ```python
    embedder = HuggingFaceSparseTextEmbedder(
        model_name="prithivida/Splade_PP_en_v1"
    )
    
    result = embedder.run("Your text here")
    embedding = result["sparse_embedding"]
    ```
    """
    def __init__(
        self, 
        model_name: str = "prithivida/Splade_PP_en_v1",
        batch_size: int = 128,        
        top_k: Optional[int] = None,
        device: Optional[str] = None,
        is_called_from_document_embedder: bool = False
    ):
        """
        Create a HuggingFaceSparseTextEmbedder component.
        
        :param model_name: Name of the SPLADE model to use
        :param batch_size: Number of texts to process at once
        :param top_k: If set, only return top k features
        :param device: Device to run model on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.top_k = top_k
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.is_called_from_document_embedder = is_called_from_document_embedder
    
    def warm_up(self):
        # Initialize model and tokenizer    
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
    def to_dict(self) -> Dict:
        """Serialize this component to a dictionary."""
        return default_to_dict(
            self,
            model_name=self.model_name,
            batch_size=self.batch_size,
            top_k=self.top_k,
            device=self.device
        )
    
    @classmethod
    def from_dict(cls, data: Dict) -> "HuggingFaceSparseTextEmbedder":
        """Deserialize this component from a dictionary."""
        return default_from_dict(cls, data)

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode a batch of texts
        """
        
        if self.model is None or self.tokenizer is None:
            self.warm_up()
            
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            # Apply SPLADE activation
            sparse_emb = torch.max(
                torch.log(1 + torch.relu(logits)),
                dim=1
            )[0]
            
        return sparse_emb.cpu().numpy()

    def _get_sparse_features(self, embeddings: np.ndarray) -> List[Dict[str, float]]:
        vocab = self.tokenizer.get_vocab()
        inv_vocab = {v: k for k, v in vocab.items()}
        
        features = []
        sparse_embeddings = []
        for embedding in embeddings:
            
            if self.top_k:
                top_indices = np.argsort(embedding)[-self.top_k:]
            else:
                top_indices = np.where(embedding > 0)[0]
                
            
            sparse_embedding = SparseEmbedding(
                indices=[x for x in top_indices],
                values=[embedding[idx] for idx in top_indices]
            )
            sparse_embeddings.append(sparse_embedding)
            # feature_dict = {
            #     inv_vocab[idx]: float(embedding[idx]) 
            #     for idx in top_indices
            # }
            # features.append(feature_dict)
            
        return sparse_embeddings

    @component.output_types(sparse_embedding=SparseEmbedding)
    def run(self, text: Union[str, List[str]]) -> Dict[str, Union[SparseEmbedding, List[SparseEmbedding]]]:
        """
        Run the embedder on the input text.
        
        :param text: A string or list of strings to embed
        :returns: Dictionary containing sparse embeddings and optional features
        :raises TypeError: If the input is not a string or list of strings
        """
        if isinstance(text, str):            
            texts = [text]
        else:
            texts = text
            
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embeddings = self._encode_batch(batch)
            all_embeddings.append(embeddings)
            
        all_embeddings = np.vstack(all_embeddings)                
        sparse_embeddings = self._get_sparse_features(all_embeddings)
        # if isinstance(text, str):
        if not self.is_called_from_document_embedder:
            return {"sparse_embedding": sparse_embeddings[0]}   
        else:            
            return {"sparse_embedding": sparse_embeddings}
        
        
        


@component
class HuggingFaceSparseDocumentEmbedder:
    def __init__(
        self,
        model_name: str = "prithivida/Splade_PP_en_v1",
        batch_size: int = 128,
        top_k: Optional[int] = None,
        device: Optional[str] = None,
        meta_fields: Optional[List[str]] = None
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.top_k = top_k
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.meta_fields = meta_fields or []
        
        self.text_embedder = HuggingFaceSparseTextEmbedder(
            model_name=model_name,
            batch_size=batch_size,
            top_k=top_k,
            device=device,
            is_called_from_document_embedder=True
        )
    
    def warm_up(self):
        self.text_embedder.warm_up()
        
    def to_dict(self) -> Dict:
        return default_to_dict(
            self,
            model_name=self.model_name,
            batch_size=self.batch_size,
            top_k=self.top_k,
            device=self.device,
            meta_fields=self.meta_fields
        )
    
    @classmethod
    def from_dict(cls, data: Dict) -> "HuggingFaceSparseDocumentEmbedder":
        return default_from_dict(cls, data)

    def _get_document_text(self, doc: Document) -> str:
        text_parts = [doc.content or ""]
        for field in self.meta_fields:
            if field in doc.meta:
                text_parts.append(str(doc.meta[field]))
        return " ".join(text_parts)

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """Embed a list of documents."""
        texts = [self._get_document_text(doc) for doc in documents]
        
        embeddings_output = self.text_embedder.run(texts)
        embeddings = embeddings_output["sparse_embedding"]
        features = embeddings_output.get("features")
        
        for idx, doc in enumerate(documents):
            doc.sparse_embedding = embeddings[idx]
            # if features:
            #     doc.meta["sparse_features"] = features[idx]
                
        return {"documents": documents}