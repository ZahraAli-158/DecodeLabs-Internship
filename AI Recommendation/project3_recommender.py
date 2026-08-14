"""
Project 3: AI Recommendation Logic
DecodeLabs - AI Agent Fellowship 2026
Zahra | BS-AI | The University of Faisalabad

CAPSTONE: Tech Stack Recommender
Goal: Map a user's raw skills/interests to the best-matching job role using
      Content-Based Filtering (TF-IDF + Cosine Similarity).

Architecture: IPO Framework (Input -> Process -> Output)
Pipeline (4 steps, as per slides): Ingestion -> Scoring -> Sorting -> Filtering
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# STEP 1 (INPUT / INGESTION): LOAD THE ITEM DATASET
# =========================================================
print("=" * 65)
print("STEP 1: INGESTION -> LOADING JOB ROLES DATASET")
print("=" * 65)

df = pd.read_csv("raw_skills.csv")
print(f"Total job roles (items) loaded: {len(df)}")
print("\nSample of dataset:")
print(df.head())

# =========================================================
# WHY CONTENT-BASED FILTERING? (as explained in slides)
# =========================================================
# Collaborative filtering needs historical user-behavior data (e.g. "users who
# picked X also picked Y"). We don't have that here (Cold Start problem for a
# brand-new tool). Content-Based Filtering solves this: it maps user input
# directly to item attributes (job role skill-sets), no history required.

# =========================================================
# STEP 2 (PROCESS): CAPTURE USER STATE (MINIMUM 3 INPUTS)
# =========================================================
print("\n" + "=" * 65)
print("STEP 2: INGESTION -> CAPTURING USER PROFILE (min 3 skills)")
print("=" * 65)

def get_user_skills():
    """Take user input interactively. Falls back to a demo profile if run
    non-interactively (e.g. no terminal input available)."""
    try:
        raw = input("Apni kam se kam 3 skills/interests likhein (comma se separate karein): ")
        skills = [s.strip() for s in raw.split(",") if s.strip()]
        if len(skills) < 3:
            print("Kam se kam 3 skills zaroori hain, demo profile use ki ja rahi hai.")
            return ["Python", "Cloud Computing", "Automation"]
        return skills
    except (EOFError, OSError):
        print("(No interactive input detected -> using demo profile)")
        return ["Python", "Cloud Computing", "Automation"]

user_skills = get_user_skills()
user_profile_text = " ".join(user_skills)
print(f"User Profile Captured: {user_skills}")

# =========================================================
# STEP 3 (PROCESS): VECTOR MAPPING WITH TF-IDF
# =========================================================
print("\n" + "=" * 65)
print("STEP 3: SCORING -> TF-IDF VECTOR MAPPING")
print("=" * 65)
print("Machine numbers hi samajhti hai, isliye text (skills) ko numerical")
print("vectors mein convert kar rahe hain, jahan generic/common skills ka")
print("weight kam hoga aur unique/specific skills ka weight zyada hoga.")

# Combine item descriptions + the user profile into ONE shared vocabulary
# space (crucial: item features and user features MUST share the same
# vocabulary, otherwise the similarity math fails - as slides warn).
corpus = df["skills"].tolist() + [user_profile_text]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

# Last row = the user profile vector; everything before it = job role vectors
job_vectors = tfidf_matrix[:-1]
user_vector = tfidf_matrix[-1]

print(f"Vocabulary size (unique skill-terms across dataset): {len(vectorizer.get_feature_names_out())}")

# =========================================================
# STEP 4 (PROCESS): COSINE SIMILARITY SCORING
# =========================================================
print("\n" + "=" * 65)
print("STEP 4: SCORING -> COSINE SIMILARITY (industry standard)")
print("=" * 65)
print("Euclidean distance vector ki 'size' se sensitive hoti hai, isliye hum")
print("Cosine Similarity use karte hain jo sirf 'orientation/angle' dekhti hai.")
print("Score 1 = perfect match, Score 0 = koi common skill nahi.")

similarity_scores = cosine_similarity(user_vector, job_vectors).flatten()
df["match_score"] = similarity_scores

# =========================================================
# STEP 5 (OUTPUT): SORTING
# =========================================================
print("\n" + "=" * 65)
print("STEP 5: SORTING -> RANKING ALL JOB ROLES BY MATCH SCORE")
print("=" * 65)

ranked = df.sort_values(by="match_score", ascending=False).reset_index(drop=True)
print(ranked[["job_role", "match_score"]].round(4).to_string(index=False))

# =========================================================
# STEP 6 (OUTPUT): FILTERING -> TOP-N RECOMMENDATIONS
# =========================================================
TOP_N = 3
print("\n" + "=" * 65)
print(f"STEP 6: FILTERING -> TOP {TOP_N} RECOMMENDATIONS (choice overload se bachao)")
print("=" * 65)

top_matches = ranked.head(TOP_N)

for i, row in top_matches.iterrows():
    match_pct = row["match_score"] * 100
    print(f"\n#{i + 1}: {row['job_role']}  —  {match_pct:.1f}% match")
    print(f"    Required skills: {row['skills']}")

# =========================================================
# COLD START CHECK (as explained in slides)
# =========================================================
print("\n" + "=" * 65)
print("COLD START CHECK")
print("=" * 65)
if top_matches["match_score"].max() == 0:
    print("⚠️  User profile ka koi bhi skill dataset ke vocabulary se match nahi hua.")
    print("    Yeh 'Cold Start' problem hai — fallback ke taur par 'Trending' ya")
    print("    'Popular roles' dikhana chahiye (jaise Data Scientist / Full Stack Developer).")
else:
    print("✅ User profile ko dataset ke against successfully score kiya gaya.")

# =========================================================
# FINAL SUMMARY
# =========================================================
print("\n" + "=" * 65)
print("PROJECT 3 COMPLETE - SUMMARY")
print("=" * 65)
print(f"User Input:      {user_skills}")
print(f"Algorithm:       Content-Based Filtering (TF-IDF + Cosine Similarity)")
print(f"Dataset Size:    {len(df)} job roles")
print(f"Top Recommendation: {top_matches.iloc[0]['job_role']} ({top_matches.iloc[0]['match_score']*100:.1f}% match)")
print("Badge Requirement: Complete ✅")
