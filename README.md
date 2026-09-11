# Candidate Selection Tool - NLP Resume Screener

an NLP-powered tool designed to help recruiters and hiring managers evaluate how well a candidate's resume matches a job description using lexical, semantic, skill, experience, and requirement matching.

## Features
- **Multi-modal Input:** Provide the Job Description via manual text entry, PDF upload, or TXT file.
- **Advanced NLP Scoring:** Uses a hybrid approach combining:
    - **Lexical Matching:** TF-IDF Vectorization for keyword overlap.
    - **Semantic Analysis:** Sentence Transformers (all-MiniLM-L6-v2) for deep contextual matching.
    - **Skill Extraction:** Rule-based extraction and normalization of programming, data science, and DevOps skills.
    - **Experience Tracking:** Automatic extraction of required vs. candidate years of experience.
    - **Education Matching:** Verification of degrees and fields of study.

## How it Works
The tool calculates a weighted final score based on:
1. Required Skills (30%)
2. Semantic Requirement Matching (20%)
3. Experience Alignment (15%)
4. Lexical Similarity (10%)
5. Preferred Skills (10%)
6. Overall Semantic Similarity (10%)
7. Education Match (5%)
