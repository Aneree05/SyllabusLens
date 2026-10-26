"""
Question Paper Segmentation Module for SyllabusLens.
Extracts individual questions, subquestions, and associated marks from raw exam paper text.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple


@dataclass
class QuestionItem:
    """Represents a segmented question from an examination paper."""
    question_id: str
    raw_text: str
    clean_text: str
    marks: Optional[str] = None
    parent_id: Optional[str] = None
    sub_label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "raw_text": self.raw_text,
            "clean_text": self.clean_text,
            "marks": self.marks or "",
            "parent_id": self.parent_id or "",
            "sub_label": self.sub_label or ""
        }


class QuestionSegmenter:
    """
    Parses exam question paper text into individual questions and subquestions.
    Handles varied numbering formats (1., Q1, Question 1, 1(a), 1.1, (i), (ii), etc.).
    """

    def __init__(self):
        # Patterns for extracting marks e.g. [5 marks], (10M), [10], (5 Marks)
        self.marks_pattern = re.compile(
            r'\[\s*(\d+\s*(?:marks?|m|pts?)?)\s*\]|\(\s*(\d+\s*(?:marks?|m|pts?))\s*\)',
            re.IGNORECASE
        )

        # Primary question header regex:
        # e.g.:
        # "1.", "1)", "1:", "Q1.", "Q.1", "Q1:", "Question 1:", "Question 1.", "1.1"
        self.main_q_pattern = re.compile(
            r'^(?:(?:Question|Ques|Q)\.?\s*(\d+[a-zA-Z]?)|(\d{1,3})\s*[\.\:\)]|'
            r'(\d{1,3})(?=\((?:[a-z]|[ivxIVX]{1,5})\)))\s*(.*)$',
            re.IGNORECASE
        )

        # Subquestion pattern e.g. "(a)", "(b)", "a)", "a.", "(i)", "(ii)", "i)"
        # Bare "a." labels are only accepted when followed by a capital/digit, so that
        # continuation text such as "e.g. ..." or "i.e. ..." is not read as a sub-question.
        self.sub_q_pattern = re.compile(
            r'^(?:\(([a-z]|[ivxIVX]{1,5})\)|([a-z]|[ivxIVX]{1,5})\)|([a-z])\.(?=\s+[A-Z0-9]))\s*(.*)$'
        )

    def extract_marks(self, text: str) -> Tuple[str, Optional[str]]:
        """Extracts mark indicators from text and returns (clean_text, marks)."""
        match = self.marks_pattern.search(text)
        marks = None
        if match:
            marks = match.group(1) or match.group(2)
            # Clean marks from question text
            text = self.marks_pattern.sub('', text)
        return text.strip(), marks

    def segment(self, raw_text: str) -> List[QuestionItem]:
        """
        Segments raw question paper text into QuestionItem objects.
        If structured numbering is absent, falls back to paragraph splitting.
        """
        if not raw_text or not raw_text.strip():
            return []

        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        questions: List[QuestionItem] = []

        current_main_num: Optional[str] = None
        current_sub_label: Optional[str] = None
        current_buffer: List[str] = []
        current_marks: Optional[str] = None

        def flush_current():
            nonlocal current_buffer, current_main_num, current_sub_label, current_marks
            if not current_buffer:
                return
            full_text = " ".join(current_buffer).strip()
            clean_body, detected_marks = self.extract_marks(full_text)
            final_marks = detected_marks or current_marks

            if current_main_num and current_sub_label:
                qid = f"Q{current_main_num}.{current_sub_label}"
                parent = f"Q{current_main_num}"
            elif current_main_num:
                qid = f"Q{current_main_num}"
                parent = None
            else:
                qid = f"Q{len(questions) + 1}"
                parent = None

            questions.append(QuestionItem(
                question_id=qid,
                raw_text=full_text,
                clean_text=clean_body,
                marks=final_marks,
                parent_id=parent,
                sub_label=current_sub_label
            ))
            current_buffer = []
            current_marks = None

        # If document does not contain numbered questions, segment by paragraphs directly
        has_numbered_questions = any(self.main_q_pattern.match(l) for l in lines)
        if not has_numbered_questions:
            paragraphs = [p.strip() for p in re.split(r'\n\s*\n', raw_text) if p.strip()]
            for idx, p in enumerate(paragraphs, 1):
                clean_body, detected_marks = self.extract_marks(p)
                questions.append(QuestionItem(
                    question_id=f"Q{idx}",
                    raw_text=p,
                    clean_text=clean_body,
                    marks=detected_marks
                ))
            return questions

        for line in lines:
            # Check for main question starter
            main_match = self.main_q_pattern.match(line)
            if main_match:
                flush_current()
                q_num = main_match.group(1) or main_match.group(2) or main_match.group(3)
                current_main_num = str(q_num)
                current_sub_label = None
                rest = main_match.group(4).strip()

                # Check if the line immediately contains a subquestion like "(a) What is..."
                sub_in_line = self.sub_q_pattern.match(rest)
                if sub_in_line:
                    current_sub_label = sub_in_line.group(1) or sub_in_line.group(2) or sub_in_line.group(3)
                    rest = sub_in_line.group(4).strip()

                if rest:
                    current_buffer.append(rest)
                continue

            # Check for standalone subquestion starter like "(a) Explain..."
            sub_match = self.sub_q_pattern.match(line)
            if sub_match and (current_main_num is not None or current_buffer):
                flush_current()
                current_sub_label = sub_match.group(1) or sub_match.group(2) or sub_match.group(3)
                rest = sub_match.group(4).strip()
                if rest:
                    current_buffer.append(rest)
                continue

            # Regular continuation line
            if current_buffer or current_main_num:
                current_buffer.append(line)

        flush_current()
        return self._drop_instruction_stems(questions)

    @staticmethod
    def _drop_instruction_stems(questions: List[QuestionItem]) -> List[QuestionItem]:
        """
        A bare stem such as "1. Answer:" that is followed by sub-questions carries no content
        and would be scored as an (always unmatched) question. Short stems (<= 3 words) are dropped;
        longer stems are kept because they provide context.
        """
        parents_with_children = {q.parent_id for q in questions if q.parent_id}
        return [q for q in questions
                if not (q.question_id in parents_with_children and len(q.clean_text.split()) <= 3)]
