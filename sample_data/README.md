# Sample Academic Evaluation Dataset

This directory contains synthetic benchmark data for testing and evaluating **SyllabusLens**:

1. `sample_question_paper.txt`:
   - A representative 4th-year B.Tech End-Semester Examination Paper in Natural Language Processing.
   - Includes Part A (short concept questions) and Part B (in-depth descriptive questions).
   - Question 7 is intentionally an out-of-syllabus Quantum Computing question to test negative retrieval handling.

2. `sample_lecture_notes_nlp.txt`:
   - Comprehensive lecture notes for CS801 covering 6 units:
     - Unit 1: Foundations & Text Normalization
     - Unit 2: Vector Space Models & TF-IDF Retrieval
     - Unit 3: POS Tagging & NER
     - Unit 4: Distributed Word Representations (Word2Vec, GloVe)
     - Unit 5: Neural Architectures (RNN, LSTM, Transformers, Attention)
     - Unit 6: Evaluation & Academic Syllabus Coverage

3. `ground_truth_relevance.json`:
   - Labeled query-to-section relevance judgments for computing Precision@k, Recall@k, MRR, MAP, and coverage classification matrices.
