import streamlit as st
import pdfplumber
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# PAGE CONFIG
st.set_page_config(
    page_title="Candidate Selection Tool",
    page_icon="📄",
    layout="centered"
)

st.title("Candidate Selection Tool")
st.subheader("NLP Based Resume Screening")
st.caption(
    "Evaluate how well a candidate's resume matches a job description "
    "using lexical, semantic, skill, experience, and requirement matching."
)

# PDF / TEXT EXTRACTION
def extract_pdf_text(uploaded_file):
    """Extract text from a PDF file."""
    text = []

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)

    return "\n".join(text)


def clean_text(text):
    """Basic text normalization."""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# SENTENCE TRANSFORMER
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_embedding_model()


# SKILL NORMALIZATION
# Canonical skill -> possible aliases
SKILL_ALIASES = {

    # Programming
    "python": ["python", "python3"],
    "java": ["java"],
    "javascript": ["javascript", "js"],
    "typescript": ["typescript", "ts"],
    "c++": ["c++", "cpp"],
    "c#": ["c#", "c sharp"],
    "go": ["golang", "go"],
    "rust": ["rust"],

    # Data / ML
    "machine learning": [
        "machine learning",
        "ml"
    ],
    "deep learning": [
        "deep learning",
        "dl"
    ],
    "artificial intelligence": [
        "artificial intelligence",
        "ai"
    ],
    "natural language processing": [
        "natural language processing",
        "nlp"
    ],
    "computer vision": [
        "computer vision",
        "cv"
    ],

    # ML frameworks
    "tensorflow": ["tensorflow"],
    "pytorch": ["pytorch"],
    "keras": ["keras"],
    "scikit-learn": [
        "scikit-learn",
        "sklearn",
        "scikit learn"
    ],
    "opencv": ["opencv", "cv2"],
    "yolo": ["yolo", "yolov8", "yolov9", "yolov10", "yolov11"],

    # LLM / RAG
    "llm": [
        "large language model",
        "large language models",
        "llm",
        "llms"
    ],
    "rag": [
        "rag",
        "retrieval augmented generation",
        "retrieval-augmented generation"
    ],
    "langchain": ["langchain"],
    "chromadb": ["chromadb", "chroma db"],
    "sentence transformers": [
        "sentence transformers",
        "sentence-transformers"
    ],
    "hugging face": [
        "hugging face",
        "huggingface"
    ],

    # Databases
    "sql": ["sql"],
    "mysql": ["mysql"],
    "postgresql": [
        "postgresql",
        "postgres"
    ],
    "mongodb": ["mongodb", "mongo db"],
    "redis": ["redis"],

    # Cloud
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "gcp": [
        "gcp",
        "google cloud",
        "google cloud platform"
    ],

    # DevOps
    "docker": ["docker"],
    "kubernetes": [
        "kubernetes",
        "k8s"
    ],
    "git": ["git"],
    "github": ["github"],
    "ci/cd": [
        "ci/cd",
        "continuous integration",
        "continuous deployment"
    ],

    # Web
    "react": ["react", "reactjs", "react.js"],
    "node.js": ["node.js", "nodejs", "node js"],
    "flask": ["flask"],
    "django": ["django"],
    "fastapi": ["fastapi"],

    # Mobile
    "flutter": ["flutter"],
    "dart": ["dart"],

    # Other
    "linux": ["linux"],
    "rest api": [
        "rest api",
        "restful api",
        "rest apis"
    ],
    "api": ["api", "apis"],
}


def normalize_skill(skill):
    """Convert aliases to canonical skill names."""
    skill = skill.lower().strip()

    for canonical, aliases in SKILL_ALIASES.items():
        if skill in aliases:
            return canonical

    return skill


def extract_skills(text):
    """
    Extract known skills from text and normalize them.
    """
    text_lower = text.lower()

    found = set()

    for canonical, aliases in SKILL_ALIASES.items():

        for alias in aliases:

            # Escape alias so characters such as + and . are safe
            pattern = r"(?<!\w)" + re.escape(alias.lower()) + r"(?!\w)"

            if re.search(pattern, text_lower):
                found.add(canonical)
                break

    return found


# JD SECTION DETECTION
PREFERRED_KEYWORDS = [
    "preferred",
    "nice to have",
    "nice-to-have",
    "bonus",
    "plus",
    "desirable",
    "preferred qualifications",
    "preferred skills"
]

REQUIRED_KEYWORDS = [
    "required",
    "requirements",
    "required qualifications",
    "required skills",
    "must have",
    "minimum qualifications",
    "qualifications"
]


def classify_jd_skills(jd_text):
    """
    Identify required vs preferred skills based on nearby section
    headings and wording.

    This is intentionally rule-based for Version 1.
    """

    lines = jd_text.split("\n")

    current_section = "general"

    required_text = []
    preferred_text = []

    for line in lines:

        line_clean = line.strip().lower()

        if not line_clean:
            continue

        # Detect section heading
        if any(keyword in line_clean for keyword in PREFERRED_KEYWORDS):
            current_section = "preferred"

        elif any(keyword in line_clean for keyword in REQUIRED_KEYWORDS):
            current_section = "required"

        if current_section == "preferred":
            preferred_text.append(line)

        elif current_section == "required":
            required_text.append(line)

    required_skills = extract_skills(" ".join(required_text))
    preferred_skills = extract_skills(" ".join(preferred_text))

    # Remove overlap
    preferred_skills -= required_skills

    # If the JD has no recognizable sections,
    # treat all detected skills as required.
    if not required_skills and not preferred_skills:
        required_skills = extract_skills(jd_text)

    return required_skills, preferred_skills

# EXPERIENCE EXTRACTION
def extract_required_years(jd_text):
    """
    Extract the largest explicit number of years required by the JD.

    Examples:
        "3+ years experience"
        "5 years of experience"
        "minimum 2 years"
    """

    patterns = [
        r"(\d+)\s*\+\s*years?",
        r"(\d+)\s*years?\s+(?:of\s+)?experience",
        r"minimum\s+of\s+(\d+)\s*years?",
        r"at\s+least\s+(\d+)\s*years?"
    ]

    years = []

    for pattern in patterns:
        matches = re.findall(pattern, jd_text.lower())

        for match in matches:
            years.append(float(match))

    return max(years) if years else None


def extract_candidate_years(resume_text):
    """
    Estimate explicit years of experience from the resume.

    Version 1 uses explicit statements only.
    """

    patterns = [
        r"(\d+)\s*\+\s*years?\s+(?:of\s+)?experience",
        r"(\d+)\s*years?\s+(?:of\s+)?experience",
        r"(\d+)\s*years?\s+in"
    ]

    years = []

    for pattern in patterns:
        matches = re.findall(pattern, resume_text.lower())

        for match in matches:
            years.append(float(match))

    return max(years) if years else None


def calculate_experience_score(jd_text, resume_text):

    required_years = extract_required_years(jd_text)
    candidate_years = extract_candidate_years(resume_text)

    # No explicit requirement
    if required_years is None:
        return 1.0, required_years, candidate_years

    # Requirement exists but resume doesn't explicitly state years
    if candidate_years is None:
        return 0.5, required_years, candidate_years

    if candidate_years >= required_years:
        return 1.0, required_years, candidate_years

    # Partial credit
    score = candidate_years / required_years

    return min(score, 1.0), required_years, candidate_years


# EDUCATION MATCHING
def calculate_education_score(jd_text, resume_text):

    jd_lower = jd_text.lower()
    resume_lower = resume_text.lower()

    degree_terms = [
        "bachelor",
        "bachelor's",
        "bsc",
        "b.sc",
        "bs",
        "master",
        "master's",
        "msc",
        "m.sc",
        "ms",
        "phd",
        "doctorate"
    ]

    field_terms = [
        "computer science",
        "computer engineering",
        "software engineering",
        "information technology",
        "information systems",
        "data science",
        "artificial intelligence",
        "machine learning",
        "electrical engineering"
    ]

    jd_degrees = [
        term for term in degree_terms
        if term in jd_lower
    ]

    jd_fields = [
        term for term in field_terms
        if term in jd_lower
    ]

    # No obvious education requirement
    if not jd_degrees and not jd_fields:
        return 1.0

    degree_match = (
        any(term in resume_lower for term in jd_degrees)
        if jd_degrees
        else True
    )

    field_match = (
        any(term in resume_lower for term in jd_fields)
        if jd_fields
        else True
    )

    if degree_match and field_match:
        return 1.0

    if degree_match or field_match:
        return 0.5

    return 0.0

# TF-IDF MATCH
def calculate_lexical_similarity(resume_text, jd_text):

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=15000
    )

    X = vectorizer.fit_transform([
        resume_text,
        jd_text
    ])

    return float(
        cosine_similarity(X[0], X[1])[0, 0]
    )


# CHUNKING
def split_into_chunks(text, max_words=80):

    words = text.split()

    chunks = []

    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])

        if chunk.strip():
            chunks.append(chunk)

    return chunks


# SEMANTIC REQUIREMENT MATCHING
def calculate_semantic_requirement_score(
    jd_text,
    resume_text,
    model
):
    """
    Instead of embedding the entire JD and entire resume only,
    compare JD chunks against resume chunks.

    For each JD chunk, take the best matching resume chunk.
    """

    jd_chunks = split_into_chunks(jd_text)
    resume_chunks = split_into_chunks(resume_text)

    if not jd_chunks or not resume_chunks:
        return 0.0

    jd_embeddings = model.encode(
        jd_chunks,
        normalize_embeddings=True
    )

    resume_embeddings = model.encode(
        resume_chunks,
        normalize_embeddings=True
    )

    similarities = cosine_similarity(
        jd_embeddings,
        resume_embeddings
    )

    # For each JD chunk, find its strongest resume match
    best_matches = similarities.max(axis=1)

    return float(np.mean(best_matches))


# REQUIRED SKILL MATCHING
def calculate_skill_scores(
    jd_required,
    jd_preferred,
    resume_skills
):

    required_matched = sorted(
        jd_required.intersection(resume_skills)
    )

    required_missing = sorted(
        jd_required - resume_skills
    )

    preferred_matched = sorted(
        jd_preferred.intersection(resume_skills)
    )

    preferred_missing = sorted(
        jd_preferred - resume_skills
    )

    # Required score
    if jd_required:
        required_score = (
            len(required_matched) /
            len(jd_required)
        )
    else:
        required_score = 1.0

    # Preferred score
    if jd_preferred:
        preferred_score = (
            len(preferred_matched) /
            len(jd_preferred)
        )
    else:
        preferred_score = 1.0

    return {
        "required_score": required_score,
        "preferred_score": preferred_score,
        "required_matched": required_matched,
        "required_missing": required_missing,
        "preferred_matched": preferred_matched,
        "preferred_missing": preferred_missing
    }


# FINAL SCORING ENGINE
def getResult(JD_txt, resume_txt):

    JD_txt = clean_text(JD_txt)
    resume_txt = clean_text(resume_txt)

    lexical_score = calculate_lexical_similarity(
        resume_txt,
        JD_txt
    )

    embeddings = model.encode(
        [resume_txt, JD_txt],
        normalize_embeddings=True
    )

    semantic_score = float(
        cosine_similarity(
            [embeddings[0]],
            [embeddings[1]]
        )[0, 0]
    )

    required_skills, preferred_skills = classify_jd_skills(
        JD_txt
    )

    resume_skills = extract_skills(resume_txt)

    skill_results = calculate_skill_scores(
        required_skills,
        preferred_skills,
        resume_skills
    )

    requirement_semantic_score = (
        calculate_semantic_requirement_score(
            JD_txt,
            resume_txt,
            model
        )
    )

    experience_score, required_years, candidate_years = (
        calculate_experience_score(
            JD_txt,
            resume_txt
        )
    )

    education_score = calculate_education_score(
        JD_txt,
        resume_txt
    )

    final_score = (
        0.30 * skill_results["required_score"] +
        0.10 * skill_results["preferred_score"] +
        0.20 * requirement_semantic_score +
        0.10 * lexical_score +
        0.10 * semantic_score +
        0.15 * experience_score +
        0.05 * education_score
    )

    return {
        "final_score": final_score,

        "lexical_score": lexical_score,
        "semantic_score": semantic_score,
        "requirement_semantic_score": requirement_semantic_score,

        "required_skill_score": skill_results["required_score"],
        "preferred_skill_score": skill_results["preferred_score"],

        "required_matched": skill_results["required_matched"],
        "required_missing": skill_results["required_missing"],

        "preferred_matched": skill_results["preferred_matched"],
        "preferred_missing": skill_results["preferred_missing"],

        "resume_skills": sorted(resume_skills),

        "experience_score": experience_score,
        "required_years": required_years,
        "candidate_years": candidate_years,

        "education_score": education_score
    }


# USER INTERFACE
st.write("### Step 1: Provide Job Description")

jd_method = st.radio(
    "Choose JD input method:",
    ("Type Manually", "Upload PDF", "Upload TXT")
)

job_description = ""

if jd_method == "Type Manually":

    job_description = st.text_area(
        "Paste Job Description Here",
        height=250
    )

elif jd_method == "Upload PDF":

    uploadedJD = st.file_uploader(
        "Upload Job Description PDF",
        type="pdf"
    )

    if uploadedJD:
        job_description = extract_pdf_text(uploadedJD)

elif jd_method == "Upload TXT":

    uploadedJDTxt = st.file_uploader(
        "Upload Job Description TXT",
        type="txt"
    )

    if uploadedJDTxt:
        job_description = uploadedJDTxt.read().decode(
            "utf-8",
            errors="ignore"
        )


# RESUME INPUT
st.write("### Step 2: Upload Resume")

uploadedResume = st.file_uploader(
    "Upload Resume PDF",
    type="pdf"
)

resume = ""

if uploadedResume:
    resume = extract_pdf_text(uploadedResume)

# PROCESS
if st.button("Process", type="primary"):

    if job_description and resume:

        result = getResult(
            job_description,
            resume
        )

        # FINAL SCORE
        final_score = result["final_score"] * 100

        st.write("## 📊 Overall Result")

        st.metric(
            "Resume Relevance Score",
            f"{final_score:.2f}%"
        )

        # COMPONENT SCORES
        st.write("### Score Breakdown")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Required Skills",
                f"{result['required_skill_score'] * 100:.2f}%"
            )

            st.metric(
                "Semantic Requirements",
                f"{result['requirement_semantic_score'] * 100:.2f}%"
            )

            st.metric(
                "Experience",
                f"{result['experience_score'] * 100:.2f}%"
            )

        with col2:

            st.metric(
                "Preferred Skills",
                f"{result['preferred_skill_score'] * 100:.2f}%"
            )

            st.metric(
                "Lexical Similarity",
                f"{result['lexical_score'] * 100:.2f}%"
            )

            st.metric(
                "Document Semantic Similarity",
                f"{result['semantic_score'] * 100:.2f}%"
            )

        # REQUIRED SKILLS
        st.write("### 🛠 Required Skills")

        if result["required_matched"]:
            st.success(
                "Matched: " +
                ", ".join(result["required_matched"])
            )

        if result["required_missing"]:
            st.error(
                "Missing: " +
                ", ".join(result["required_missing"])
            )

        if not result["required_matched"] and not result["required_missing"]:
            st.info("No explicit required skills detected.")

        # PREFERRED SKILLS
        st.write("### ⭐ Preferred Skills")

        if result["preferred_matched"]:
            st.success(
                "Matched: " +
                ", ".join(result["preferred_matched"])
            )

        if result["preferred_missing"]:
            st.warning(
                "Missing: " +
                ", ".join(result["preferred_missing"])
            )
        # EXPERIENCE
        st.write("### 💼 Experience")

        if result["required_years"] is not None:

            st.write(
                f"Required experience: "
                f"**{result['required_years']:.0f}+ years**"
            )

            if result["candidate_years"] is not None:

                st.write(
                    f"Detected candidate experience: "
                    f"**{result['candidate_years']:.0f} years**"
                )

            else:

                st.warning(
                    "Could not detect an explicit number of "
                    "experience years in the resume."
                )

        else:

            st.info(
                "No explicit years-of-experience requirement detected."
            )

        # EDUCATION
        st.write("### 🎓 Education")

        st.write(
            f"Education match: "
            f"**{result['education_score'] * 100:.2f}%**"
        )

        # RESUME SKILLS
        with st.expander("Detected Resume Skills"):

            if result["resume_skills"]:
                st.write(
                    ", ".join(result["resume_skills"])
                )
            else:
                st.write("No known skills detected.")

    else:

        st.error(
            "Please provide both the Job Description "
            "and the Resume."
        )
