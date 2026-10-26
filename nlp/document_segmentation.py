"""
Document Segmentation and Chunking Module for SyllabusLens.
Divides lecture notes (PDF pages, PPTX slides, TXT units) into structured, referenced sections.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


def truncate_title(text: str, limit: int = 90) -> str:
    """Shorten a heading at a word boundary, never mid-word."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:-") + "\u2026"


def clean_heading(heading: str) -> str:
    """Remove the internal 'Sec N:' numbering used for plain-text sections (display only)."""
    return re.sub(r"^Sec \d+:\s*", "", heading or "").strip()


@dataclass
class NoteSection:
    """Represents a coherent section or chunk of academic lecture notes."""
    section_id: str
    doc_name: str
    page_or_slide: str
    heading: str
    text: str
    word_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.word_count == 0 and self.text:
            self.word_count = len(self.text.split())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_id": self.section_id,
            "doc_name": self.doc_name,
            "page_or_slide": self.page_or_slide,
            "heading": self.heading,
            "text": self.text,
            "word_count": self.word_count
        }


class DocumentSegmenter:
    """
    Splits lecture materials into manageable, referenced chunks
    preserving page numbers, slide numbers, or unit headers.
    """

    def __init__(self, target_chunk_words: int = 250, chunk_overlap_words: int = 30):
        self.target_chunk_words = target_chunk_words
        self.chunk_overlap_words = chunk_overlap_words

    def segment_pptx_slides(self, doc_name: str, slides_data: List[Dict[str, Any]]) -> List[NoteSection]:
        """
        Segments PPTX slides.
        Each item in slides_data is expected to have:
        {'slide_num': int, 'title': str, 'text': str}
        """
        sections: List[NoteSection] = []
        for s in slides_data:
            s_num = s.get("slide_num", len(sections) + 1)
            title = (s.get("title") or "").strip()
            content = (s.get("text") or "").strip()

            if not content:
                continue

            heading = f"Slide {s_num}: {title}" if title else f"Slide {s_num}"
            full_text = f"{title}\n{content}" if title and title not in content else content

            # If slide is reasonably sized, keep it as single section; else split with overlap
            words = full_text.split()
            if len(words) <= self.target_chunk_words * 1.5:
                sections.append(NoteSection(
                    section_id=f"{doc_name}_Slide_{s_num}",
                    doc_name=doc_name,
                    page_or_slide=f"Slide {s_num}",
                    heading=heading,
                    text=full_text,
                    metadata={"slide_num": s_num, "title": title}
                ))
            else:
                chunks = self._chunk_words(words, self.target_chunk_words, self.chunk_overlap_words)
                for c_idx, c_text in enumerate(chunks, 1):
                    sections.append(NoteSection(
                        section_id=f"{doc_name}_Slide_{s_num}_Part_{c_idx}",
                        doc_name=doc_name,
                        page_or_slide=f"Slide {s_num} (Part {c_idx})",
                        heading=f"{heading} (Part {c_idx})",
                        text=c_text,
                        metadata={"slide_num": s_num, "part": c_idx}
                    ))

        return sections

    def segment_pdf_pages(self, doc_name: str, pages_text: List[str]) -> List[NoteSection]:
        """
        Segments PDF pages.
        Preserves page numbering and subdivides large pages into overlapping chunks.
        """
        sections: List[NoteSection] = []
        for p_idx, page_str in enumerate(pages_text, 1):
            text = (page_str or "").strip()
            if not text:
                continue

            # Attempt to extract a representative top heading from first non-empty line
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            top_line = lines[0] if lines else f"Page {p_idx}"
            heading = f"Page {p_idx}: {top_line[:50]}..." if len(top_line) > 50 else f"Page {p_idx}: {top_line}"

            words = text.split()
            if len(words) <= self.target_chunk_words * 1.4:
                sections.append(NoteSection(
                    section_id=f"{doc_name}_Page_{p_idx}",
                    doc_name=doc_name,
                    page_or_slide=f"Page {p_idx}",
                    heading=heading,
                    text=text,
                    metadata={"page_num": p_idx}
                ))
            else:
                chunks = self._chunk_words(words, self.target_chunk_words, self.chunk_overlap_words)
                for c_idx, c_text in enumerate(chunks, 1):
                    sections.append(NoteSection(
                        section_id=f"{doc_name}_Page_{p_idx}_Part_{c_idx}",
                        doc_name=doc_name,
                        page_or_slide=f"Page {p_idx} (Part {c_idx})",
                        heading=f"{heading} (Part {c_idx})",
                        text=c_text,
                        metadata={"page_num": p_idx, "part": c_idx}
                    ))

        return sections

    def segment_txt_document(self, doc_name: str, full_text: str) -> List[NoteSection]:
        """
        Segments plain text or Markdown notes.
        Detects Unit/Chapter/Lecture headers or Markdown headers (`#`, `##`).
        """
        sections: List[NoteSection] = []
        if not full_text or not full_text.strip():
            return []

        # Split on markdown headers or unit indicators
        header_split_pattern = re.compile(
            r'(?m)^(?:#{1,4}\s+|(?:\b(?:Unit|Chapter|Lecture|Module|Section|Topic)\s+\d+[:\.\-]?\s*))'
        )

        matches = list(header_split_pattern.finditer(full_text))

        if len(matches) > 1:
            indices = [m.start() for m in matches] + [len(full_text)]
            for i in range(len(indices) - 1):
                chunk_str = full_text[indices[i]:indices[i+1]].strip()
                if not chunk_str:
                    continue
                first_line = chunk_str.splitlines()[0].strip()
                clean_title = truncate_title(re.sub(r'^[#\s]+', '', first_line))
                sec_idx = i + 1

                sections.append(NoteSection(
                    section_id=f"{doc_name}_Sec_{sec_idx}",
                    doc_name=doc_name,
                    page_or_slide=f"Section {sec_idx}",
                    heading=f"Sec {sec_idx}: {clean_title}",
                    text=chunk_str,
                    metadata={"section_idx": sec_idx}
                ))
        else:
            # Fallback to paragraph or word-window chunking
            paragraphs = [p.strip() for p in re.split(r'\n\s*\n', full_text) if p.strip()]
            cur_buffer = []
            cur_words = 0
            chunk_num = 1

            for p in paragraphs:
                p_words = len(p.split())
                if cur_words + p_words > self.target_chunk_words and cur_buffer:
                    chunk_text = "\n\n".join(cur_buffer)
                    sections.append(NoteSection(
                        section_id=f"{doc_name}_Part_{chunk_num}",
                        doc_name=doc_name,
                        page_or_slide=f"Part {chunk_num}",
                        heading=f"{doc_name} (Part {chunk_num})",
                        text=chunk_text,
                        metadata={"part": chunk_num}
                    ))
                    chunk_num += 1
                    cur_buffer = [p]
                    cur_words = p_words
                else:
                    cur_buffer.append(p)
                    cur_words += p_words

            if cur_buffer:
                chunk_text = "\n\n".join(cur_buffer)
                sections.append(NoteSection(
                    section_id=f"{doc_name}_Part_{chunk_num}",
                    doc_name=doc_name,
                    page_or_slide=f"Part {chunk_num}",
                    heading=f"{doc_name} (Part {chunk_num})",
                    text=chunk_text,
                    metadata={"part": chunk_num}
                ))

        return sections

    def _chunk_words(self, words: List[str], chunk_size: int, overlap: int) -> List[str]:
        """Utility window chunker for long sequences of words."""
        chunks = []
        start = 0
        step = max(1, chunk_size - overlap)
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            if end >= len(words):
                break
            start += step
        return chunks
