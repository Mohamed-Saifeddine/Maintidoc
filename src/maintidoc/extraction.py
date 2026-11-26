from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from pathlib import Path

import pdfplumber
import pymupdf
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader

from maintidoc.config import Settings
from maintidoc.embeddings import OllamaVisionCaptioner
from maintidoc.models import DocumentElement, ElementType

ProgressCallback = Callable[[str, float], None]


def document_id_for(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", path.stem).strip("-").lower()
    return f"{stem}-{digest}"


def table_to_markdown(table: list[list[object | None]]) -> str:
    rows = [[str(cell or "").replace("\n", " ").strip() for cell in row] for row in table]
    if len(rows) < 2 or not any(any(cell for cell in row) for row in rows):
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    header = rows[0]
    body = rows[1:]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


class MultimodalPDFExtractor:
    """Extraction séparée du texte, des tableaux et des images d'un PDF."""

    def __init__(self, config: Settings) -> None:
        self.config = config
        self.splitter = SentenceSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        self.captioner = OllamaVisionCaptioner(config) if config.enable_vlm_captions else None

    def extract(
        self,
        pdf_path: Path,
        progress: ProgressCallback | None = None,
    ) -> tuple[list[DocumentElement], int]:
        path = pdf_path.resolve()
        if not path.exists() or path.suffix.lower() != ".pdf":
            raise FileNotFoundError(f"PDF introuvable : {path}")

        doc_id = document_id_for(path)
        output_dir = self.config.extracted_dir / doc_id
        output_dir.mkdir(parents=True, exist_ok=True)

        if progress:
            progress("Extraction du texte avec LlamaIndex", 0.05)
        text_elements, page_texts = self._extract_text(path, doc_id)

        if progress:
            progress("Extraction des tableaux", 0.35)
        table_elements = self._extract_tables(path, doc_id)

        if progress:
            progress("Extraction des images", 0.60)
        image_elements, page_count = self._extract_images(
            path, doc_id, output_dir, page_texts, progress
        )

        elements = text_elements + table_elements + image_elements
        return elements, page_count

    def _extract_text(
        self, path: Path, doc_id: str
    ) -> tuple[list[DocumentElement], dict[int, str]]:
        documents = PyMuPDFReader().load(file_path=path)
        elements: list[DocumentElement] = []
        page_texts: dict[int, str] = {}
        for page_index, document in enumerate(documents, start=1):
            page = int(document.metadata.get("page_label") or page_index)
            page_texts[page] = document.text
            nodes = self.splitter.get_nodes_from_documents([document])
            for chunk_index, node in enumerate(nodes, start=1):
                content = node.get_content().strip()
                if not content:
                    continue
                elements.append(
                    DocumentElement(
                        document_id=doc_id,
                        file_name=path.name,
                        page=page,
                        element_id=f"page_{page:03d}_text_{chunk_index:03d}",
                        element_type=ElementType.TEXT,
                        content=content,
                    )
                )
        return elements, page_texts

    def _extract_tables(self, path: Path, doc_id: str) -> list[DocumentElement]:
        elements: list[DocumentElement] = []
        with pymupdf.open(path) as pymupdf_document, pdfplumber.open(path) as pdfplumber_document:
            for page_number, page in enumerate(pymupdf_document, start=1):
                detected_tables = [table.extract() for table in page.find_tables().tables]
                if not detected_tables:
                    detected_tables = (
                        pdfplumber_document.pages[page_number - 1].extract_tables() or []
                    )
                for table_index, table in enumerate(detected_tables, start=1):
                    markdown = table_to_markdown(table)
                    if len(markdown) < 20:
                        continue
                    elements.append(
                        DocumentElement(
                            document_id=doc_id,
                            file_name=path.name,
                            page=page_number,
                            element_id=f"page_{page_number:03d}_table_{table_index:03d}",
                            element_type=ElementType.TABLE,
                            content=markdown,
                        )
                    )
        return elements

    def _extract_images(
        self,
        path: Path,
        doc_id: str,
        output_dir: Path,
        page_texts: dict[int, str],
        progress: ProgressCallback | None,
    ) -> tuple[list[DocumentElement], int]:
        pdf = pymupdf.open(path)
        elements: list[DocumentElement] = []
        seen_hashes: set[str] = set()
        image_count = 0
        for page_index, page in enumerate(pdf, start=1):
            for image_index, image in enumerate(page.get_images(full=True), start=1):
                if image_count >= self.config.max_images_per_document:
                    break
                extracted = pdf.extract_image(image[0])
                data = extracted["image"]
                width = int(extracted.get("width", 0))
                height = int(extracted.get("height", 0))
                digest = hashlib.sha256(data).hexdigest()
                if digest in seen_hashes or len(data) < 1500 or width < 120 or height < 120:
                    continue
                seen_hashes.add(digest)
                extension = extracted.get("ext", "png")
                image_path = (
                    output_dir / f"page_{page_index:03d}_image_{image_index:03d}.{extension}"
                )
                image_path.write_bytes(data)
                context = page_texts.get(page_index, "")
                description = f"Image extraite de la page {page_index}. Contexte : {context[:1000]}"
                if self.captioner:
                    try:
                        description = self.captioner.describe(image_path, context)
                    except Exception as exc:  # L'indexation CLIP reste possible si le VLM échoue.
                        description += f" [Description VLM indisponible : {exc}]"
                elements.append(
                    DocumentElement(
                        document_id=doc_id,
                        file_name=path.name,
                        page=page_index,
                        element_id=f"page_{page_index:03d}_image_{image_index:03d}",
                        element_type=ElementType.IMAGE,
                        content=description,
                        image_path=image_path,
                        metadata={"width": width, "height": height},
                    )
                )
                image_count += 1
                if progress:
                    progress(
                        f"Image {image_count} extraite",
                        min(
                            0.85,
                            0.60 + image_count / max(self.config.max_images_per_document, 1) * 0.25,
                        ),
                    )
        page_count = len(pdf)
        pdf.close()
        return elements, page_count
