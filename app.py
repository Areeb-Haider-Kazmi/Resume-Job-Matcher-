import streamlit as st
from utils import (
    extract_text_from_pdf,
    compute_similarity_bert,
    extract_keywords
)

st.set_page_config(page_title="Resume & JD Matcher", layout="centered")
st.title("📄 Resume & Job Description Matcher (BERT powered)")

st.markdown("Upload your resume and job description. This app uses BERT from Hugging Face to compute match similarity.")

resume_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
jd_text = st.text_area("Paste Job Description", height=200)

if st.button("Match"):
    if resume_file and jd_text.strip() != "":
        with st.spinner("Analyzing using BERT..."):
            resume_text = extract_text_from_pdf(resume_file)
            score = compute_similarity_bert(resume_text, jd_text)

            st.success(f"✅ Match Score: **{score:.2f}%**")

            # Match interpretation
            if score > 75:
                st.balloons()
                st.info("Excellent Match! 🚀")
            elif score > 50:
                st.warning("Moderate match. Customize your resume.")
            else:
                st.error("Low match. Consider improving alignment.")

            # Matched keywords
            st.subheader("🔑 Matched Keywords")
            keywords = extract_keywords(resume_text, jd_text)
            if keywords:
                st.markdown(f"`{', '.join(keywords[:20])}`")
            else:
                st.markdown("_No major keyword overlaps found._")
    else:
        st.warning("Please upload both Resume and Job Description.")
