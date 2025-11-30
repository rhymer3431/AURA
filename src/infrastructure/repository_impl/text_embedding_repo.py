from domain.node.repository.text_embedding_port import TextEmbeddingPort


class TextEmbeddingRepository(TextEmbeddingPort):
    def __init__(self, adapter: TextEmbeddingPort):
        self.adapter = adapter

    def encode(self, text: str):
        return self.adapter.encode(text)
