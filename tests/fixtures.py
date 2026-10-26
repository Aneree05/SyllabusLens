"""
SYNTHETIC TEST DATA. Hand-written for automated tests only; not real course material.
Notes use '# ' headings so each heading becomes one indexed section.
"""

NLP = dict(
    description="Computational linguistics: tokenization, vector space models, parsing and embeddings.",
    topics=["Tokenization", "TF-IDF weighting", "Syntactic parsing", "Word embeddings", "Hidden Markov models"],
    files={"notes.txt": """# Tokenization
Tokenization splits raw text into tokens such as words and subwords. Whitespace tokenization fails on punctuation and contractions. Subword tokenization like byte pair encoding handles unknown words.
# TF-IDF weighting
TF-IDF weighting multiplies term frequency by inverse document frequency. Terms that are frequent in a document but rare in the corpus receive high TF-IDF weights. Cosine similarity compares TF-IDF vectors.
# Syntactic parsing
Syntactic parsing builds a parse tree for a sentence using a context free grammar. The CYK algorithm and shift reduce parsers are common parsing methods. Dependency parsing links heads and dependents.
# Word embeddings
Word embeddings such as word2vec and GloVe map words to dense vectors. Skip gram and continuous bag of words are training objectives. Embeddings capture semantic similarity between words.
"""},
    paper1="""NLP End Semester Examination - Time allowed: 3 hours - Maximum marks: 50
1. Explain tokenization and why whitespace tokenization fails on punctuation. [5 marks]
2. Describe TF-IDF weighting and how cosine similarity compares document vectors. [10 marks]
3. Explain how a parse tree is built by syntactic parsing with a context free grammar. [10 marks]
""",
    paper2="""NLP End Semester Examination 2024 - Time allowed: 3 hours - Maximum marks: 50
1. Discuss word embeddings such as word2vec and the skip gram objective. [10 marks]
2. Describe TF-IDF weighting of terms using inverse document frequency. [5 marks]
3. Explain subword tokenization with byte pair encoding. [5 marks]
""",
)

OS = dict(
    description="Operating system concepts: scheduling, deadlocks and virtual memory.",
    topics=["CPU scheduling", "Deadlocks", "Virtual memory and paging"],
    files={"notes.txt": """# CPU scheduling
CPU scheduling decides which process runs next. Round robin scheduling uses a time quantum. Shortest job first minimises average waiting time.
# Deadlocks
A deadlock occurs when processes wait for each other forever. The four Coffman conditions are mutual exclusion, hold and wait, no preemption and circular wait. The banker algorithm avoids deadlock.
# Virtual memory and paging
Virtual memory uses paging to map virtual pages to physical frames. A page fault triggers page replacement such as LRU or FIFO. The page table stores the mapping and the TLB caches it.
"""},
    paper1="""OS End Semester Examination - Time allowed: 3 hours - Maximum marks: 50
1. Explain virtual memory paging and page replacement using LRU. [10 marks]
2. State the Coffman conditions for deadlock. [5 marks]
""",
)

DBMS = dict(
    description="Database systems: relational model, SQL and normalization.",
    topics=["Normalization", "SQL joins", "Transactions"],
    files={"notes.txt": """# Normalization
Normalization removes redundancy using functional dependencies. Third normal form and BCNF are common targets.
# SQL joins
An inner join returns matching rows. Outer joins keep unmatched rows with null values.
# Transactions
A transaction has ACID properties. Two phase locking enforces serializability.
"""},
)
