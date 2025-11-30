from domain.node.repository.text_embedding_port import TextEmbeddingPort


class ClipTextAdapter(TextEmbeddingPort):
    def encode(self, text: str):
        raise NotImplementedError("CLIP text adapter not implemented yet.")
