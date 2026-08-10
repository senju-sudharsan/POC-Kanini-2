"""Page-aware document chunking for the Phase 3 RAG pipeline."""

from hashlib import sha256
import re

from poc_kanini.models.documents import ProcessedDocument, ProcessedPage
from poc_kanini.rag.models import DocumentChunk


class DocumentChunker:
    """Convert Phase 2 processed documents into citation-ready chunks."""

    def __init__(
        self,
        target_characters: int = 1200,
        max_characters: int = 1800,
        overlap_characters: int = 150,
    ) -> None:
        if target_characters <= 0:
            raise ValueError("target_characters must be positive.")
        if max_characters < target_characters:
            raise ValueError("max_characters must be >= target_characters.")
        if overlap_characters < 0 or overlap_characters >= target_characters:
            raise ValueError(
                "overlap_characters must be >= 0 and smaller than target_characters."
            )

        self.target_characters = target_characters
        self.max_characters = max_characters
        self.overlap_characters = overlap_characters

    def chunk(self, document: ProcessedDocument) -> list[DocumentChunk]:
        """Create page-aware chunks while preserving document provenance."""

        chunks: list[DocumentChunk] = []
        chunk_index = 0

        for page in document.pages:
            blocks = self._extract_blocks(page)

            for text in self._group_blocks(blocks):
                chunks.append(
                    self._create_chunk(
                        document=document,
                        page=page,
                        text=text,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

        return chunks

    def _extract_blocks(self, page: ProcessedPage) -> list[str]:
        """Prefer Phase 2 structural information, with text fallback."""

        blocks: list[str] = []

        for value in (
            page.structure.headings
            + page.structure.paragraphs
            + page.structure.list_items
            + page.structure.table_lines
        ):
            value = value.strip()
            if value:
                blocks.append(value)

        if not blocks:
            fallback = page.normalized_text.strip() or page.text.strip()
            if fallback:
                blocks = self._split_sentences(fallback)

        return blocks

    def _group_blocks(self, blocks: list[str]) -> list[str]:
        """Group related blocks into bounded chunks."""

        chunks: list[str] = []
        current = ""

        for block in blocks:
            pieces = self._split_large_block(block)

            for piece in pieces:
                candidate = f"{current}\n\n{piece}".strip() if current else piece

                if len(candidate) <= self.target_characters:
                    current = candidate
                    continue

                if current:
                    chunks.append(current)

                overlap = self._get_overlap(current)
                current = f"{overlap}\n\n{piece}".strip() if overlap else piece

                if len(current) > self.max_characters:
                    chunks.append(current)
                    current = ""

        if current:
            chunks.append(current)

        return chunks

    def _split_large_block(self, text: str) -> list[str]:
        """Split oversized blocks into manageable pieces."""

        if len(text) <= self.max_characters:
            return [text]

        sentences = self._split_sentences(text)

        if len(sentences) <= 1:
            return self._split_by_words(text)

        pieces: list[str] = []
        current = ""

        for sentence in sentences:
            candidate = f"{current} {sentence}".strip() if current else sentence

            if len(candidate) <= self.target_characters:
                current = candidate
            else:
                if current:
                    pieces.append(current)

                if len(sentence) <= self.max_characters:
                    current = sentence
                else:
                    pieces.extend(self._split_by_words(sentence))
                    current = ""

        if current:
            pieces.append(current)

        return pieces

    def _split_by_words(self, text: str) -> list[str]:
        """Bound very large text without cutting words."""

        words = text.split()
        pieces: list[str] = []
        current: list[str] = []
        length = 0

        for word in words:
            added_length = len(word) + (1 if current else 0)

            if current and length + added_length > self.target_characters:
                pieces.append(" ".join(current))
                current = [word]
                length = len(word)
            else:
                current.append(word)
                length += added_length

        if current:
            pieces.append(" ".join(current))

        return pieces

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Lightweight sentence splitting for the chunking boundary."""

        return [
            value.strip()
            for value in re.split(r"(?<=[.!?])\s+", text.strip())
            if value.strip()
        ]

    def _get_overlap(self, text: str) -> str:
        if not text or self.overlap_characters == 0:
            return ""

        return text[-self.overlap_characters :].lstrip()

    def _create_chunk(
        self,
        document: ProcessedDocument,
        page: ProcessedPage,
        text: str,
        chunk_index: int,
    ) -> DocumentChunk:
        chunk_id = sha256(
            (
                f"{document.metadata.document_id}:"
                f"{page.page_number}:"
                f"{chunk_index}:"
                f"{text}"
            ).encode("utf-8")
        ).hexdigest()

        return DocumentChunk(
            chunk_id=chunk_id,
            document_id=document.metadata.document_id,
            filename=document.metadata.filename,
            document_type=document.document_type,
            page_number=page.page_number,
            chunk_index=chunk_index,
            text=text,
            metadata={
                "document_id": document.metadata.document_id,
                "filename": document.metadata.filename,
                "document_type": document.document_type,
                "page_number": str(page.page_number),
                "chunk_index": str(chunk_index),
                "extraction_method": page.extraction_method,
                "ocr_status": page.ocr_status,
            },
        )