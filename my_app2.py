import streamlit as st
import pandas as pd
import os
import tempfile
import numpy as np
import time
import logging
import warnings

# Set page configuration
st.set_page_config(page_title="Resume-to-Job Matcher", layout="wide")

# App title and description
st.title("Resume-to-Job Matcher")
st.write("Upload your resume and find suitable job matches.")

# Sidebar for algorithm selection and CSV path input
with st.sidebar:
    st.header("Configuration")
    
    # Algorithm selection
    algorithm = st.radio(
        "Choose matching approach:",
        ["SBERT", "TF-IDF", "LSH with Cache"]
    )
    
    # Path to the CSV file input (only used for SBERT and LSH)
    jobs_csv_path = st.text_input("Path to job descriptions CSV file", value="job_descriptions.csv")
    
    # Number of results to show
    top_n = st.slider("Number of job matches to display", min_value=1, max_value=20, value=5)

# Main area - File uploader for PDF resume only
uploaded_resume = st.file_uploader("Upload Resume (PDF)", type="pdf")

# Processing button
if st.button("Find Matching Jobs"):
    if uploaded_resume is None:
        st.warning("Please upload a resume file.")
    elif not os.path.exists(jobs_csv_path) and algorithm != "TF-IDF":
        st.warning(f"CSV file not found at path: {jobs_csv_path}")
    else:
        # Save uploaded PDF resume to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_resume:
            temp_resume.write(uploaded_resume.getvalue())
            resume_path = temp_resume.name
            
        try:
            with st.spinner(f"Processing with {algorithm}..."):
                # SBERT implementation
                if algorithm == "SBERT":
                    import re
                    import string
                    import joblib
                    import torch
                    from sentence_transformers import SentenceTransformer, util
                    import fitz  # PyMuPDF
                    
                    # Configure logging
                    warnings.filterwarnings('ignore')
                    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename='job_recommender_sbert.log')
                    logger = logging.getLogger()
                    
                    # --- Utility Functions ---
                    def extract_resume_text(pdf_path):
                        try:
                            doc = fitz.open(pdf_path)
                            return " ".join([page.get_text() for page in doc])
                        except Exception as e:
                            logger.error(f"PDF extraction failed: {e}")
                            return ""

                    def clean_text(text):
                        if not isinstance(text, str):
                            return ""
                        text = text.lower()
                        text = re.sub(r'\n', ' ', text)
                        text = text.translate(str.maketrans('', '', string.punctuation))
                        return re.sub(r'\s+', ' ', text).strip()

                    def weight_text(text, weight):
                        if not isinstance(text, str):
                            return ""
                        weighted = text * int(weight)
                        frac = weight - int(weight)
                        if frac:
                            words = text.split()
                            weighted += " " + " ".join(words[:int(len(words) * frac)])
                        return weighted

                    # --- SBERT Caching ---
                    def build_or_load_sbert_embeddings(jobs_df, cache_dir="./sbert_cache"):
                        os.makedirs(cache_dir, exist_ok=True)
                        vec_path = os.path.join(cache_dir, "job_embeddings.npy")
                        model_path = os.path.join(cache_dir, "sbert_model.pkl")

                        if os.path.exists(vec_path) and os.path.exists(model_path):
                            logger.info("Loading cached SBERT embeddings...")
                            return np.load(vec_path), joblib.load(model_path), jobs_df

                        weights = {'Job Title': 3.0, 'Role': 2.5, 'skills': 2.0, 'Job Description': 1.0, 'Company': 0.8}
                        jobs_df['Weighted_Text'] = ""
                        for field, weight in weights.items():
                            if field in jobs_df.columns:
                                jobs_df['Weighted_Text'] += jobs_df[field].fillna('').apply(lambda x: weight_text(str(x), weight) + " ")
                        jobs_df['Weighted_Text_Clean'] = jobs_df['Weighted_Text'].apply(clean_text)

                        model = SentenceTransformer('all-MiniLM-L6-v2')
                        job_embeddings = model.encode(jobs_df['Weighted_Text_Clean'].tolist(), convert_to_numpy=True, show_progress_bar=True)

                        np.save(vec_path, job_embeddings)
                        joblib.dump(model, model_path)
                        return job_embeddings, model, jobs_df

                    # --- Similarity Matching ---
                    def match_resume_sbert(resume_text, job_embeddings, model, jobs_df, top_n=10):
                        resume_clean = clean_text(resume_text)
                        resume_embedding = model.encode(resume_clean, convert_to_tensor=True)
                        job_embeddings_tensor = torch.tensor(job_embeddings).to(resume_embedding.device)

                        similarities = util.pytorch_cos_sim(resume_embedding, job_embeddings_tensor)[0].cpu().numpy()
                        jobs_df = jobs_df.copy()
                        jobs_df['similarity'] = similarities
                        return jobs_df.sort_values('similarity', ascending=False).head(top_n)

                    # --- Main Recommender ---
                    def run_job_recommender_sbert(resume_path, jobs_csv_path, top_n=10):
                        start = time.time()
                        logger.info("Loading data...")
                        jobs_df = pd.read_csv(jobs_csv_path)
                        resume_text = extract_resume_text(resume_path)
                        if not resume_text:
                            logger.warning("Resume extraction failed.")
                            return pd.DataFrame({'Message': ['Resume extraction failed']})

                        job_embeddings, model, processed_df = build_or_load_sbert_embeddings(jobs_df)
                        results = match_resume_sbert(resume_text, job_embeddings, model, processed_df, top_n)
                        logger.info(f"Total execution time: {time.time() - start:.2f}s")
                        return results
                    
                    results = run_job_recommender_sbert(resume_path, jobs_csv_path, top_n)
                
                # TF-IDF implementation - Modified to use Snowflake
                elif algorithm == "TF-IDF":
                    import os
                    import re
                    import string
                    import time
                    import logging
                    import warnings
                    import pandas as pd
                    import numpy as np
                    from collections import Counter
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from sklearn.metrics.pairwise import cosine_similarity
                    from sklearn.preprocessing import normalize
                    from scipy.sparse import save_npz, load_npz
                    import joblib
                    import snowflake.connector
                    import fitz
                    
                    warnings.filterwarnings('ignore')
                    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename='job_recommender_optimized.log')
                    logger = logging.getLogger()
                    
                    # --- Utility Functions ---
                    def extract_resume_text(pdf_path):
                        try:
                            doc = fitz.open(pdf_path)
                            return " ".join([page.get_text() for page in doc])
                        except Exception as e:
                            logger.error(f"PDF extraction failed: {e}")
                            return ""

                    def clean_text(text):
                        if not isinstance(text, str):
                            return ""
                        text = text.lower()
                        text = re.sub(r'\n', ' ', text)
                        text = text.translate(str.maketrans('', '', string.punctuation))
                        return re.sub(r'\s+', ' ', text).strip()

                    def weight_text(text, weight):
                        if not isinstance(text, str):
                            return ""
                        weighted = text * int(weight)
                        frac = weight - int(weight)
                        if frac:
                            words = text.split()
                            weighted += " " + " ".join(words[:int(len(words) * frac)])
                        return weighted
                    
                    # --- TF-IDF Caching ---
                    def build_or_load_tfidf(jobs_df, cache_dir="./cache"):
                        os.makedirs(cache_dir, exist_ok=True)
                        vec_path, model_path = f"{cache_dir}/job_vectors.npz", f"{cache_dir}/vectorizer.pkl"
                        
                        if os.path.exists(vec_path) and os.path.exists(model_path):
                            logger.info("Loading cached TF-IDF...")
                            return load_npz(vec_path), joblib.load(model_path), jobs_df
                            
                        weights = {
                            'Job Title': 3.0, 'Role': 2.5, 'skills': 2.0, 'Job Description': 1.0, 'Company': 0.8
                        }
                        jobs_df['Weighted_Text'] = ""
                        for field, weight in weights.items():
                            if field in jobs_df.columns:
                                jobs_df['Weighted_Text'] += jobs_df[field].fillna('').apply(lambda x: weight_text(str(x), weight) + " ")
                        jobs_df['Weighted_Text_Clean'] = jobs_df['Weighted_Text'].apply(clean_text)
                        
                        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, min_df=2, max_df=0.85, ngram_range=(1, 2), sublinear_tf=True)
                        job_vectors = vectorizer.fit_transform(jobs_df['Weighted_Text_Clean'])
                        
                        save_npz(vec_path, job_vectors)
                        joblib.dump(vectorizer, model_path)
                        return job_vectors, vectorizer, jobs_df
                    
                    # --- Similarity Matching ---
                    def match_resume(resume_text, job_vectors, vectorizer, jobs_df, top_n=10):
                        resume_clean = clean_text(resume_text)
                        resume_vector = vectorizer.transform([resume_clean])
                        similarities = cosine_similarity(resume_vector, job_vectors)[0]
                        
                        jobs_df = jobs_df.copy()
                        jobs_df['similarity'] = similarities
                        top_jobs = jobs_df.sort_values('similarity', ascending=False).head(top_n)
                        return top_jobs
                    
                    # --- Main Recommender ---
                    def run_job_recommender(resume_path, jobs_df, top_n=10):
                        start = time.time()
                        logger.info("Loading data...")
                        resume_text = extract_resume_text(resume_path)
                        if not resume_text:
                            logger.warning("Resume extraction failed.")
                            return pd.DataFrame({'Message': ['Resume extraction failed']})
                            
                        job_vectors, vectorizer, processed_df = build_or_load_tfidf(jobs_df)
                        results = match_resume(resume_text, job_vectors, vectorizer, processed_df, top_n)
                        logger.info(f"Total execution time: {time.time() - start:.2f}s")
                        return results
                    
                    # Connect to Snowflake using given credentials
                    conn = snowflake.connector.connect(
                        user='MUDIT',
                        password='Testing@123123',
                        account='BCEMHHI-LB94703',
                        warehouse='COMPUTE_WH',
                        database='JOB_RECOMMENDATIONS',
                        schema='JOB_DATA',
                        role='ACCOUNTADMIN'
                    )
                    
                    # Execute query and fetch results into a pandas DataFrame
                    st.write("Connected to Snowflake successfully.")
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM JOB_DATASET")  # Replace with your actual table name
                    #jobs_df = cursor.fetch_pandas_all()
                    
                    jobs_df = pd.read_csv(jobs_csv_path)
                    cursor.close()
                    
                    # Run TF-IDF recommender with Snowflake data
                    results = run_job_recommender(resume_path, jobs_df, top_n)
                
                # LSH implementation
                else:  # LSH with Cache
                    import os
                    import re
                    import string
                    import time
                    import logging
                    import warnings
                    import pandas as pd
                    import numpy as np
                    import joblib
                    from datasketch import MinHash, MinHashLSH
                    import fitz
                    
                    warnings.filterwarnings('ignore')
                    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename='job_recommender_lsh.log')
                    logger = logging.getLogger()
                    
                    # --- Utility Functions ---
                    def extract_resume_text(pdf_path):
                        try:
                            doc = fitz.open(pdf_path)
                            text = " ".join([page.get_text() for page in doc])
                            doc.close()
                            return text
                        except Exception as e:
                            logger.error(f"PDF extraction failed: {e}")
                            print(f"Error extracting text from {pdf_path}: {e}")
                            return ""

                    def clean_text(text):
                        if not isinstance(text, str):
                            return ""
                        text = text.lower()
                        text = re.sub(r'\n', ' ', text)
                        text = text.translate(str.maketrans('', '', string.punctuation))
                        return re.sub(r'\s+', ' ', text).strip()

                    def weight_text(text, weight):
                        if not isinstance(text, str):
                            return ""
                        weighted = text * int(weight)
                        frac = weight - int(weight)
                        if frac:
                            words = text.split()
                            weighted += " " + " ".join(words[:int(len(words) * frac)])
                        return weighted

                    def get_shingles(text, k=5):
                        if len(text) < k:
                            return set()
                        return set([text[i:i+k] for i in range(len(text)-k+1)])
                    
                    # --- MinHash LSH Embedding with Caching ---
                    def build_or_load_lsh_index(jobs_df, cache_dir="./lsh_cache", num_perm=128, force_rebuild=False):
                        os.makedirs(cache_dir, exist_ok=True)
                        lsh_path = os.path.join(cache_dir, "lsh_index.pkl")
                        minhash_path = os.path.join(cache_dir, "minhashes.pkl")
                        df_path = os.path.join(cache_dir, "processed_jobs_df.pkl")
                        
                        # Check if all cache files exist
                        cache_exists = all(os.path.exists(path) for path in [lsh_path, minhash_path, df_path])
                        
                        if cache_exists and not force_rebuild:
                            print(f"CACHE HIT: Loading cached LSH index from {os.path.abspath(cache_dir)}")
                            logger.info("Loading cached LSH index and MinHashes...")
                            start_time = time.time()
                            try:
                                lsh = joblib.load(lsh_path)
                                minhashes = joblib.load(minhash_path)
                                cached_df = joblib.load(df_path)
                                load_time = time.time() - start_time
                                print(f"Cache loaded in {load_time:.2f} seconds")
                                
                                # Verify if cached DataFrame structure matches the current one
                                if list(cached_df.columns) == list(jobs_df.columns) and len(cached_df) == len(jobs_df):
                                    print("DataFrame structure matches cache. Using cached data.")
                                    return lsh, minhashes, cached_df
                                else:
                                    print("WARNING: DataFrame structure has changed. Rebuilding cache...")
                            except Exception as e:
                                print(f"Error loading cache: {e}. Rebuilding...")
                        else:
                            if force_rebuild:
                                print("Force rebuilding LSH index as requested.")
                            else:
                                print(f"CACHE MISS: Building new LSH index (files not found in {os.path.abspath(cache_dir)})")
                                
                        logger.info("Building new LSH index...")
                        print("Building LSH index from scratch. This may take some time...")
                        start_time = time.time()
                        
                        # Process text with weights
                        weights = {
                            'Job Title': 3.0, 'Role': 2.5, 'skills': 2.0, 'Job Description': 1.0, 'Company': 0.8
                        }
                        
                        jobs_df['Weighted_Text'] = ""
                        for field, weight in weights.items():
                            if field in jobs_df.columns:
                                jobs_df['Weighted_Text'] += jobs_df[field].fillna('').apply(lambda x: weight_text(str(x), weight) + " ")
                        jobs_df['Weighted_Text_Clean'] = jobs_df['Weighted_Text'].apply(clean_text)
                        
                        lsh = MinHashLSH(threshold=0.3, num_perm=num_perm)
                        minhashes = {}
                        
                        total_rows = len(jobs_df)
                        for i, (idx, row) in enumerate(jobs_df.iterrows()):
                            if i % 100 == 0:
                                print(f"Processing {i}/{total_rows} jobs ({i/total_rows*100:.1f}%)...")
                            shingles = get_shingles(row['Weighted_Text_Clean'])
                            m = MinHash(num_perm=num_perm)
                            for shingle in shingles:
                                m.update(shingle.encode('utf8'))
                            lsh.insert(str(idx), m)
                            minhashes[str(idx)] = m
                            
                        build_time = time.time() - start_time
                        print(f"LSH index built in {build_time:.2f} seconds")
                        
                        # Save to cache
                        print("Saving LSH index to cache...")
                        try:
                            joblib.dump(lsh, lsh_path, compress=3)
                            joblib.dump(minhashes, minhash_path, compress=3)
                            joblib.dump(jobs_df, df_path, compress=3)
                            print(f"Cache saved to {os.path.abspath(cache_dir)}")
                        except Exception as e:
                            print(f"Warning: Failed to save cache: {e}")
                            
                        return lsh, minhashes, jobs_df
                    
                    # --- Similarity Matching ---
                    def match_resume_lsh(resume_text, lsh, minhashes, jobs_df, num_perm=128, top_n=10):
                        print("Processing resume text...")
                        resume_clean = clean_text(resume_text)
                        shingles = get_shingles(resume_clean)
                        
                        if not shingles:
                            print("Warning: No shingles generated from resume. Text might be too short or empty.")
                            return pd.DataFrame({'Message': ['No valid content found in resume']})
                            
                        m = MinHash(num_perm=num_perm)
                        for sh in shingles:
                            m.update(sh.encode('utf8'))
                            
                        print("Finding matches in LSH index...")
                        result_ids = lsh.query(m)
                        
                        if not result_ids:
                            print("No matches found in LSH. Try lowering the threshold.")
                            return pd.DataFrame({'Message': ['No matches found']})
                            
                        print(f"Found {len(result_ids)} potential matches. Calculating similarities...")
                        similarities = []
                        
                        for idx in result_ids:
                            jaccard_sim = m.jaccard(minhashes[idx])
                            similarities.append((int(idx), jaccard_sim))
                            
                        similarities.sort(key=lambda x: x[1], reverse=True)
                        top_ids = [idx for idx, _ in similarities[:top_n]]
                        top_scores = [score for _, score in similarities[:top_n]]
                        
                        result_df = jobs_df.loc[top_ids].copy()
                        result_df['similarity'] = top_scores
                        return result_df
                    
                    # --- Main Recommender ---
                    def run_job_recommender_lsh(resume_path, jobs_df, cache_dir="./lsh_cache", top_n=10, force_rebuild=False, num_perm=128):
                        print(f"\n--- Job Recommender LSH Started ---")
                        start = time.time()
                        
                        print(f"Extracting text from resume: {resume_path}")
                        resume_text = extract_resume_text(resume_path)
                        if not resume_text:
                            logger.warning("Resume extraction failed.")
                            return pd.DataFrame({'Message': ['Resume extraction failed']})
                            
                        print(f"Building or loading LSH index (Force rebuild: {force_rebuild})")
                        lsh, minhashes, processed_df = build_or_load_lsh_index(
                            jobs_df,
                            cache_dir=cache_dir,
                            num_perm=num_perm,
                            force_rebuild=force_rebuild
                        )
                        
                        print(f"Finding matches for resume...")
                        results = match_resume_lsh(resume_text, lsh, minhashes, processed_df, top_n=top_n, num_perm=num_perm)
                        
                        total_time = time.time() - start
                        print(f"\n--- Job Recommender completed in {total_time:.2f} seconds ---")
                        return results
                    
                    # Use the CSV for LSH
                    jobs_df = pd.read_csv(jobs_csv_path)
                    results = run_job_recommender_lsh(
                        resume_path=resume_path,
                        jobs_df=jobs_df,
                        top_n=top_n,
                        force_rebuild=False,
                        cache_dir="./lsh_cache",
                        num_perm=128
                    )
            
            # Display results
            st.header("Top Job Matches")
            
            if 'Message' in results.columns and len(results) == 1:
                st.warning(results['Message'].iloc[0])
            else:
                # Format the similarity score to be more readable
                if 'similarity' in results.columns:
                    display_df = results.copy()
                    display_df['similarity'] = display_df['similarity'].apply(lambda x: f"{x:.4f}")
                    
                    # Show the results table
                    st.dataframe(display_df[['Job Title', 'Company', 'Role', 'location', 'similarity']])
                
                # Detailed job information in expandable sections
                st.subheader("Detailed Job Information")
                for i, (_, row) in enumerate(results.iterrows()):
                    job_title = row.get('Job Title', f"Job {i+1}")
                    company = row.get('Company', 'Unknown')
                    
                    with st.expander(f"{job_title} at {company}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Job Details**")
                            st.markdown(f"**Job Title:** {row.get('Job Title', 'N/A')}")
                            st.markdown(f"**Company:** {row.get('Company', 'N/A')}")
                            st.markdown(f"**Role:** {row.get('Role', 'N/A')}")
                            st.markdown(f"**Location:** {row.get('location', 'N/A')}")
                            st.markdown(f"**Match Score:** {row.get('similarity', 0):.4f}")
                        
                        with col2:
                            if 'skills' in row and pd.notna(row['skills']):
                                st.markdown("**Required Skills**")
                                for skill in str(row['skills']).split()[:10]:
                                    st.markdown(f"- {skill}")
                        
                        if 'Job Description' in row and pd.notna(row['Job Description']):
                            st.markdown("**Job Description**")
                            st.write(row['Job Description'])
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        
        finally:
            # Clean up temporary resume file
            if 'resume_path' in locals():
                os.unlink(resume_path)
