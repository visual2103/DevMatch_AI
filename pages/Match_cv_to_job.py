import streamlit as st
import numpy as np
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from utils.common_functions import (load_docx_from_folder, spacy_tokenizer, progress_bar_update, nlp)
from utils.explanation import generate_explanation_with_llm_cv_to_job
from concurrent.futures import ThreadPoolExecutor
import re
from src.db.queries import (
    get_job_id_by_filename,
    get_job_industry,
    get_cv_industry_scores,
    get_cv_industries,
    get_job_industry_scores
)

cv_folder = './DataSet/cv'
job_folder = './DataSet/job_descriptions'


def get_keyword_matching_scores(custom_skills, text):
    matches = 0
    for skill, weight in custom_skills:
        if re.search(rf'\b{re.escape(skill.lower())}\b', text.lower()):
            matches += weight
    return matches / 100


def get_matching_scores_between_cv_and_job_descriptions(cv_text, job_texts, progress_bar, status_text):
    if not job_texts:
        return []
    def compute_batch_similarity(batch, cv_doc):
        return [cv_doc.similarity(nlp(job_text)) for job_text in batch]

    def chunk_list(lst, n):
        return [lst[i:i+n] for i in range(0, len(lst), n)]

    cv_text_preprocessed = " ".join(spacy_tokenizer(cv_text))
    job_texts_preprocessed = [" ".join(spacy_tokenizer(job_text)) for job_text in job_texts]

    progress_bar_update(30, progress_bar, status_text)

    cv_doc = nlp(cv_text_preprocessed)
    batch_size = 50
    chunks = chunk_list(job_texts_preprocessed, batch_size)
    num_threads = min(8, len(chunks))

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = executor.map(lambda batch: compute_batch_similarity(batch, cv_doc), chunks)

    similarities_embeddings = [sim for batch in results for sim in batch]

    progress_bar_update(70, progress_bar, status_text)

    combined_texts = [cv_text_preprocessed] + job_texts_preprocessed
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_texts)

    cv_vector = tfidf_matrix[0]
    job_vectors = tfidf_matrix[1:]

    similarities_tfidf = cosine_similarity(cv_vector, job_vectors)

    progress_bar_update(80, progress_bar, status_text)

    scaler = MinMaxScaler()
    similarities_tfidf_scaled = scaler.fit_transform(
        np.array(similarities_tfidf).reshape(-1, 1)
    ).flatten()

    progress_bar_update(90, progress_bar, status_text)

    similarities_embeddings = np.array(similarities_embeddings)
    final_scores = 0.6 * similarities_embeddings + 0.4 * similarities_tfidf_scaled

    return final_scores


def filter_jobs_by_industry(db_path, jd_folder, selected_industry):
    jd_texts, jd_filenames, _ = load_docx_from_folder(jd_folder, is_cv=False)

    industry_scores = get_job_industry_scores(db_path, jd_filenames, selected_industry)

    filtered_indices = [i for i, score in enumerate(industry_scores) if score > 0]

    if not filtered_indices:
        return [], [], np.array([])

    return (
        [jd_texts[i] for i in filtered_indices],
        [jd_filenames[i] for i in filtered_indices],
        industry_scores[filtered_indices]
    )


st.title("📌 Match CV → Job")
progress_text = "Operation in progress. Please wait. ⏳"
uploaded_file = st.file_uploader("Upload a CV", type=["docx"])

st.subheader("🛠️ Define Required Skills and Weights")
skill_count = st.number_input("How many skills do you want to assess?", min_value=1, max_value=20, value=3)

custom_skills = []
total_weight = 0

if skill_count:
    with st.form("skill_form"):
        for i in range(skill_count):
            col1, col2 = st.columns([2, 1])
            with col1:
                skill_name = st.text_input(f"Skill #{i + 1} name", key=f"skill_name_{i}")
            with col2:
                weight = st.number_input(f"Weight % for Skill #{i + 1}", min_value=0, max_value=100,
                                         key=f"skill_weight_{i}")

            if skill_name:
                custom_skills.append((skill_name.strip(), weight))
                total_weight += weight

        if total_weight != 100:
            st.error(f"Total weights must sum to 100% (current sum: {total_weight}%)")
        else:
            st.success("Weights sum to 100% - Ready to proceed!")

        submitted = st.form_submit_button("✅ Done selecting skills")

progress_bar = st.progress(0)
status_text = st.empty()

col1, col2, col3 = st.columns([1,2,1])
with col2:
    start_button = st.button(
        "Find Best Jobs for My CV",
        type="primary",
        use_container_width=True
    )

if start_button:
    if not uploaded_file:
        st.error("❌ Please upload a CV first!")
    elif total_weight != 100:
        st.error(f"❌ Total weight must be exactly 100%. Current sum: {total_weight}%.")
    else:
        cv_filename = uploaded_file.name
        cv_industries = get_cv_industries("./data/cvs_metadata.sqlite", cv_filename)
        if not cv_industries:
            st.error("❌ Industry for the uploaded CV not found in database.")
        else:
            selected_industry = cv_industries[0]  # primul industry din listă
            st.success(f"Industry detected: {selected_industry}")

        doc = Document(uploaded_file)
        cv_text = ' '.join([para.text for para in doc.paragraphs])
        cv_text = cv_text.replace('\n', ' ').replace('  ', ' ')

        # Load JD
        jd_texts, jd_filenames, industry_scores = filter_jobs_by_industry(
            "./data/cvs_metadata.sqlite",
            job_folder,
            selected_industry
        )

        progress_bar_update(10, progress_bar, status_text)


        progress_bar_update(30, progress_bar, status_text)

        cv_job_matching_scores = get_matching_scores_between_cv_and_job_descriptions(
            cv_text, jd_texts, progress_bar, status_text
        )

        skill_scores = np.array([
            get_keyword_matching_scores(custom_skills, job_text)
            for job_text in jd_texts
        ])


        progress_bar_update(70, progress_bar, status_text)

        scaler = MinMaxScaler()
        skill_scores_norm = scaler.fit_transform(skill_scores.reshape(-1, 1)).flatten()

        final_scores = (
            0.1 * industry_scores +
            0.3 * skill_scores_norm +
            0.6 * cv_job_matching_scores
        )

        best_job_idx = np.argmax(final_scores)

        progress_bar_update(90, progress_bar, status_text)

        st.markdown("---")
        st.subheader("🎯 Best Job Match")
        st.write(f"**Job Title:** {jd_filenames[best_job_idx]}")
        st.write(jd_texts[best_job_idx])

        st.markdown("---")
        st.subheader("Matched Skills:")
        # Skills matched
        common_skills = [
            skill for skill, _ in custom_skills
            if re.search(rf'\b{re.escape(skill.lower())}\b', jd_texts[best_job_idx].lower())
        ]
        if common_skills:
            st.write(", ".join(common_skills))
        else:
            st.write("No matched skills found.")

        st.markdown("---")
        st.header("Selection Explanation:")

        st.write(
            f"**Industry Knowledge Score:** {industry_scores[best_job_idx] * 100:.2f}% | "
            f"**Technical Skills Score:** {skill_scores_norm[best_job_idx] * 100:.2f}% | "
            f"**CV to Job Matching Score:** {cv_job_matching_scores[best_job_idx] * 100:.2f}% | "
            f"**Overall Match Score:** {final_scores[best_job_idx] * 100:.2f}%"
        )


        st.markdown("---")
        explanation_text = generate_explanation_with_llm_cv_to_job(
            cv_filename="Uploaded CV",
            domain_score=industry_scores[best_job_idx],
            skills_score=skill_scores_norm[best_job_idx],
            matching_score=cv_job_matching_scores[best_job_idx],
            matched_skills=common_skills,
            domain_selected=selected_industry
        )

        st.subheader("Explanation for the Best Match")
        st.write(explanation_text)

        progress_bar_update(100, progress_bar, status_text)
        st.balloons()