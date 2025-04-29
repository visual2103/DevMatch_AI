import streamlit as st

st.set_page_config(
    page_title="CV & Job Matcher",
    layout="centered"
)

st.title("Welcome to CV & Job Matcher")
st.markdown("""
### Your AI-Powered Career Matchmaker

This application helps you find the perfect match between job descriptions and CVs using advanced AI technology.

#### 🎯 What you can do:

1. **Match Job to CVs** 📋
   - Upload a job description
   - Find the best matching CVs from our database
   - Get detailed matching scores and explanations

2. **Match CV to Jobs** 📄
   - Upload your CV
   - Find the most suitable jobs
   - See how well your skills match each position

#### 🚀 Getting Started:
Use the sidebar to navigate between these features and start matching!
""")

st.markdown("---")

st.info("👈 Select a feature from the sidebar to begin!")