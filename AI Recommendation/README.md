# Project: AI Recommendation Logic
**DecodeLabs — AI Agent Internship 2026**
**Author:** Zahra | BS Artificial Intelligence | The University of Faisalabad

**Capstone:** Tech Stack Recommender

---

## 📌 Overview

Yeh project ek **Content-Based Recommendation Engine** banata hai jo user ke diye gaye skills/interests ko job roles ke saath match karke best-fit career paths suggest karta hai — bilkul waise jaise Netflix/Amazon items recommend karte hain, lekin humne pure similarity math se banaya hai, bina history data ke.

Pipeline **IPO Framework (Input → Process → Output)** aur uske andar 4-step ranking pipeline follow karta hai: **Ingestion → Scoring → Sorting → Filtering**.

---

## ✅ Requirements Fulfilled

| # | Requirement (from Project Brief) | Status |
|---|---|---|
| 1 | Take user input (choices or interests) | ✅ Minimum 3 skills liye jate hain |
| 2 | Match preferences using logic or similarity | ✅ TF-IDF + Cosine Similarity |
| 3 | Display recommended items | ✅ Top 3 job roles, match % ke saath |

**Bonus concepts bhi cover kiye:**
- Content-Based Filtering (vs Collaborative Filtering — dono ka explanation code comments mein hai)
- TF-IDF Weighting (generic skills ka weight kam, specific skills ka zyada)
- Cosine Similarity (Euclidean distance ke bajaye, kyunke yeh vector magnitude se independent hai)
- Cold Start problem detection aur fallback logic
- 4-Step Ranking Pipeline (Ingestion, Scoring, Sorting, Filtering)

---

## 🗂️ Files in this Project

| File | Description |
|---|---|
| `project3_recommender.py` | Terminal version — core engine |
| `app.py` | **Streamlit web app version (recommended)** — with UI, synonym matching, bar chart |
| `raw_skills.csv` | Dataset — 15 job roles, har ek apne required skills ke saath |
| `README.md` | Yeh file |

---

## 🌐 Streamlit Web App (Improved Version)

Terminal version ke ilawa ek **web-based version** bhi hai jisme yeh improvements hain:

1. **Web UI** — dropdown se skills select karein ya khud type karein, terminal ki zaroorat nahi
2. **Synonym Matching** — "ML" → "Machine Learning", "JS" → "JavaScript", "K8s" → "Kubernetes" waghera automatically normalize ho jate hain
3. **"Why this match?" Explanation** — har recommendation ke saath yeh dikhta hai ke exactly kaunsi skills match hui
4. **Bar Chart Visualization** — Top-N roles ka match % graphically dikhta hai
5. **Adjustable Top-N** — slider se decide karein kitni recommendations chahiye (1 se 10 tak)

### Run karne ka tareeka:
```bash
pip install streamlit pandas scikit-learn
cd path/to/downloaded/folder
streamlit run app.py
```
Yeh automatically browser mein `http://localhost:8501` open kar dega.

---

## ⚙️ How to Run

### 1. Requirements install karein
```bash
pip install pandas scikit-learn
```

### 2. Script run karein (same folder mein `raw_skills.csv` hona zaroori hai)
```bash
python project3_recommender.py
```

### 3. Jab prompt aaye, apni skills comma se separate karke likhein
```
Apni kam se kam 3 skills/interests likhein (comma se separate karein): Python, Machine Learning, Statistics
```

### 4. Output dekhein
- Terminal mein Step 1 se Step 6 tak sara process print hoga
- End mein Top 3 job role recommendations aayenge, match percentage ke saath

**Note:** Agar aap script ko bina input diye (non-interactively) run karengi, to yeh automatically ek demo profile (`Python, Cloud Computing, Automation`) use kar legi, taake script kabhi crash na ho.

---

## 🧠 Methodology (Pipeline Steps)

1. **Ingestion (Input)** — `raw_skills.csv` load kiya (15 job roles) aur user se minimum 3 skills liye
2. **Vector Mapping** — Job roles ke skills aur user ka profile, dono ko **same shared vocabulary** mein TF-IDF vectors mein convert kiya (zaroori hai warna similarity math fail ho jati hai)
3. **Scoring** — Har job role ke vector aur user ke vector ke beech **Cosine Similarity** calculate ki (0 se 1 tak score, jahan 1 = perfect match)
4. **Sorting** — Sab job roles ko match score ke hisab se descending order mein sort kiya
5. **Filtering** — Sirf **Top 3** roles show kiye, taake "choice overload" na ho

---

## 🔬 Why TF-IDF + Cosine Similarity?

- **TF-IDF:** Simple binary matching (1/0) mein generic skills (jaise "Python") aur rare/specific skills (jaise "Kubernetes") ka same weight hota hai. TF-IDF generic/common skills ko penalize karta hai aur unique skills ko zyada importance deta hai.
- **Cosine Similarity:** Euclidean distance vector ki magnitude (size) se sensitive hoti hai — agar ek job role ki description lambi ho aur dusri choti, to comparison ghalat ho sakta hai. Cosine Similarity sirf **angle/orientation** dekhti hai, isliye yeh industry-standard approach hai (Netflix, Amazon jese systems isi par based hain).

---

## 📊 Example Result

**Input:** `Python, Cloud Computing, Automation`

| Rank | Job Role | Match Score |
|---|---|---|
| 1 | DevOps Engineer | 41.5% |
| 2 | Cloud Architect | 40.9% |
| 3 | QA/Test Engineer | 13.0% |

Result logically sahi hai — DevOps aur Cloud Architect dono roles Cloud Computing aur Automation heavy hote hain.

---

## ❄️ Cold Start Handling

Agar user ki di gayi skills dataset ke vocabulary se bilkul match na karein (match score = 0 sab jagah), script automatically warn karti hai aur **Trending/Popular roles** fallback suggest karti hai — jaisa slides mein "Bypassing the Cold Start" section mein explain kiya gaya tha.

---

## 🔑 Key Learning

- Content-Based Filtering ka sabse bada faida yeh hai ke isse **history data** ki zaroorat nahi hoti — naye users bhi turant recommendations pa sakte hain
- Item features aur user features ko **exact same vocabulary** mein map karna zaroori hai, warna similarity math ghalat result dega
- Recommendation ka goal sirf "match dhoondna" nahi, balke **choice overload se bachana** bhi hai — isliye Top-N filtering zaroori hai

---

## 🚀 Next Steps (Future Improvements)

- Skills ko synonyms ke saath normalize karna (e.g. "ML" = "Machine Learning")
- Collaborative filtering add karna jab enough user interaction data mil jaye
- Web interface (Streamlit) banana taake user GUI se skills select kar sake
