from langchain_core.documents import BaseDocumentCompressor, Document
from colbert.search import Searcher
from typing import Sequence, Optional

class ColBERTDocumentCompressor(BaseDocumentCompressor):
    """Document compressor that uses ColBERT for reranking."""

    def __init__(self, searcher: Searcher, top_n: int = 3):
        self.searcher = searcher
        self.top_n = top_n

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Rerank documents using ColBERT.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        doc_texts = [doc.page_content for doc in documents]
        results = self.searcher.search(query, doc_texts)
        reranked_docs = [documents[idx] for idx in results[:self.top_n]]
        return reranked_docs