# skill_gap_analyzer.py

from flask import Flask, render_template_string, request
import pandas as pd
import re, string
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
#  Step 1: Data Loading 
import traceback

try:
    job_df = pd.read_csv("data/job_postings.csv", low_memory=False)
    resume_df = pd.read_csv("data/UpdatedResumeDataSet.csv")
except Exception as e:
    print(f"CSV loading error: {e}")
    traceback.print_exc()
    exit()


desc_col = next((col for col in job_df.columns if 'desc' in col.lower()), None)
if not desc_col:
    print(" Could not find job description column in job_postings.csv")
    exit()

#  Step 2: Cleaning Functions
def clean_text(text):
    text = re.sub(r'<.*?>', '', str(text))
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return text

job_df['clean_description'] = job_df[desc_col].apply(clean_text)
resume_df['clean_resume'] = resume_df['Resume'].apply(clean_text)

#  Step 3: Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
job_vecs = vectorizer.fit_transform(job_df['clean_description'])

#  Step 4: Model Training with Resilience
def train_model():
    os.makedirs("models", exist_ok=True)
    X, y = [], []
    for _, row in resume_df.iterrows():
        resume_text = row.get('clean_resume')
        if not isinstance(resume_text, str) or not resume_text.strip():
            continue
        try:
            sim = cosine_similarity(vectorizer.transform([resume_text]), job_vecs).flatten()
            top_score = np.max(sim)
            X.append([top_score])
            y.append(1 if top_score > 0.3 else 0)
        except Exception as err:
            print(f"⚠️ Skipping row due to error: {err}")
    model = LogisticRegression().fit(X, y)
    pickle.dump(model, open("models/rank_model.pkl", "wb"))
    return model

rank_model = train_model()
skills = [
    "python", "machine learning", "flask", "sql", "communication", "leadership",
    "data analysis", "pandas", "tensorflow", "git", "docker", "linux",
    "cloud computing", "nlp", "deep learning", "statistics", "agile", "project management", "teamwork", "problem solving", "time management",
    "critical thinking", "adaptability", "creativity",  "negotiation", "customer service", "presentation skills", "networking"
]

#  Step 5: Radar Chart
def generate_chart(user_resume, job_desc):
    
    user_skills = [1 if skill in user_resume else 0 for skill in skills]
    job_skills = [1 if skill in job_desc else 0 for skill in skills]
    angles = np.linspace(0, 2 * np.pi, len(skills), endpoint=False).tolist()
    user_skills += user_skills[:1]
    job_skills += job_skills[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, user_skills, label='Your Resume')
    ax.plot(angles, job_skills, label='Job Role')
    ax.fill(angles, user_skills, alpha=0.25)
    ax.fill(angles, job_skills, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(skills)
    ax.legend()

    os.makedirs("static", exist_ok=True)
    chart_path = "static/skill_gap.png"
    plt.savefig(chart_path)
    plt.close()
    return chart_path

skill_resources = {
    "python": {
        "courses": [
            "https://www.coursera.org/learn/python",
            "https://www.udemy.com/course/python-for-data-science/"
        ],
        "projects": [
            "https://github.com/TheAlgorithms/Python"
        ]
    },
    "machine learning": {
        "courses": [
            "https://www.coursera.org/learn/machine-learning",
            "https://www.fast.ai/"
        ],
        "projects": [
            "https://github.com/ageron/handson-ml"
        ]
    },
    "flask": {
        "courses": [
            "https://www.udemy.com/course/python-flask-for-beginners/",
            "https://realpython.com/flask-by-example-part-1-project-setup/"
        ],
        "projects": [
            "https://github.com/miguelgrinberg/microblog"
        ]
    },
    "sql": {
        "courses": [
            "https://www.kaggle.com/learn/intro-to-sql",
            "https://www.codecademy.com/learn/learn-sql"
        ],
        "projects": [
            "https://github.com/sqlite/sqlite"
        ]
    },
    "communication": {
        "courses": [
            "https://www.coursera.org/learn/communication-skills",
            "https://www.linkedin.com/learning/communication-foundations"
        ],
        "projects": []
    }
    
}
skill_explanations = {
    "python": "This role involves scripting and automation.",
    "machine learning": "Core to building intelligent systems.",
    "flask": "Used for developing APIs in this job.",
    "communication": "Important for cross-functional collaboration.",
    "sql": "Data querying is frequently mentioned."
}

def generate_skill_bar_chart(skill_scores):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, len(skill_scores) * 0.6))

    skills = list(skill_scores.keys())
    scores = list(skill_scores.values())
    colors = ['#4caf50' if s >= 60 else '#f44336' for s in scores]

    bars = ax.barh(skills, scores, color=colors, edgecolor='black')
    ax.set_xlim(0, 100)
    ax.set_xlabel("Skill Strength (%)", fontsize=12, labelpad=10)
    ax.set_title("Skill Level Estimates", fontsize=14, pad=15)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height() / 2,
                f'{int(width)}%', va='center', fontsize=10)

    plt.tight_layout()
    chart_path = "static/skill_levels.png"
    plt.savefig(chart_path, dpi=120, bbox_inches='tight')
    plt.close()
    return chart_path

#  Step 6: Recommendations
def recommend_resources(resume_text, job_text):
    recommendations = []

    for skill in skill_resources:
        in_resume = skill in resume_text
        in_job = skill in job_text

        if in_job:
            # 🔍 Missing vs 🚀 Advancement label
            label = "🔍 Missing Skill" if not in_resume else "🚀 Advance Skill"
            reason = skill_explanations.get(skill, "Required or beneficial for this role.")
            recommendations.append(f"<strong>📌 {skill.upper()}</strong> <em>— {label}</em><br><small>🧾 {reason}</small><br>")

            # Courses
            for link in skill_resources[skill]["courses"]:
                name = link.split('//')[1]
                recommendations.append(f"&nbsp;&nbsp;&nbsp;📚 <a href='{link}' target='_blank'>{name}</a><br>")

            # Projects
            for proj in skill_resources[skill]["projects"]:
                name = proj.split('//')[1]
                recommendations.append(f"&nbsp;&nbsp;&nbsp;💻 <a href='{proj}' target='_blank'>{name}</a><br>")

    return recommendations

def estimate_relevant_skills(resume_text, job_text, all_skills):
    relevant = [skill for skill in all_skills if skill in job_text]
    scores = {}
    for skill in relevant:
        count = resume_text.count(skill)
        scores[skill] = min(100, count * 20)  # simple score: more mentions = more confidence
    return scores

# 🟩 Ranking Metrics Calculation
from sklearn.metrics import precision_score, recall_score, f1_score

def compute_ranking_metrics(resume_text, job_text, skill_list):
    y_true = [1 if skill in job_text else 0 for skill in skill_list]
    y_pred = [1 if skill in resume_text else 0 for skill in skill_list]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Compute NDCG
    relevance_scores = [1 if skill in job_text else 0 for skill in skill_list]
    predicted_scores = [resume_text.count(skill) for skill in skill_list]
    sorted_indices = np.argsort(predicted_scores)[::-1]
    dcg = sum([(2**relevance_scores[i] - 1) / np.log2(idx + 2) for idx, i in enumerate(sorted_indices)])
    idcg = sum([(2**r - 1) / np.log2(idx + 2) for idx, r in enumerate(sorted(relevance_scores, reverse=True))])
    ndcg = dcg / idcg if idcg > 0 else 0.0

    # Compute MAP
    relevant_skills = set(skill for skill in skill_list if skill in job_text)
    retrieved_skills = [skill for skill in skill_list if skill in resume_text]
    hits = 0
    sum_precisions = 0
    for i, skill in enumerate(retrieved_skills):
        if skill in relevant_skills:
            hits += 1
            sum_precisions += hits / (i + 1)
    map_score = sum_precisions / len(relevant_skills) if relevant_skills else 0.0

    return {
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1-Score": round(f1, 3),
        "NDCG": round(ndcg, 3),
        "MAP": round(map_score, 3)
    }


#  Step 7: Flask Routes
@app.route('/')
def home():
    return render_template_string('''
        <html>
        <head>
            <title>JobFit Engine</title>
            <style>
                body {
                    font-family: 'Segoe UI', sans-serif;
                    background-color: #f8f9fa;
                    color: #333;
                    text-align: center;
                    margin: 0;
                    padding: 20px;
                }
                h2 {
                    color: #0077cc;
                }
                .container {
                    max-width: 700px;
                    margin: auto;
                    background-color: #ffffff;
                    padding: 30px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    border-radius: 8px;
                }
                textarea {
                    width: 100%;
                    padding: 10px;
                    font-size: 1em;
                    border-radius: 5px;
                    border: 1px solid #ccc;
                }
                input[type="submit"] {
                    background-color: #0077cc;
                    color: white;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 5px;
                    font-size: 1em;
                    cursor: pointer;
                }
                input[type="file"] {
                    margin: 10px 0;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h2>💼 JobFit Engine</h2>
                <p>Identify your skill gaps and find the perfect job match.</p>
                <form method="POST" action="/analyze" enctype="multipart/form-data">
                    <textarea name="resume" rows="10" placeholder="Paste your resume here..."></textarea><br><br>
                    <input type="file" name="resume_file" accept=".txt"><br><br>
                    <input type="submit" value="Analyze Resume">
                </form>
            </div>
        </body>
        </html>
    ''')

@app.route('/analyze', methods=['POST'])
def analyze():
    #  Step 1: Handle resume input from uploaded .txt file or textarea
    resume_text = ""
    if 'resume_file' in request.files:
        uploaded_file = request.files['resume_file']
        if uploaded_file and uploaded_file.filename.endswith('.txt'):
            resume_text = uploaded_file.read().decode('utf-8')

    #  Fallback to textarea input
    if not resume_text.strip():
        resume_text = request.form.get('resume', '')

    #  Clean resume text
    resume_text = clean_text(resume_text)

    # 🔍 Similarity matching with job descriptions
    sim_scores = cosine_similarity(vectorizer.transform([resume_text]), job_vecs).flatten()
    top_indices = sim_scores.argsort()[-3:][::-1]
    matched_jobs = job_df.iloc[top_indices]
    top_job = matched_jobs.iloc[0]['clean_description']

    #  Generate radar chart comparing skills
    chart_path = generate_chart(resume_text, top_job)

    #  Estimate resume skill strength
    skill_scores = estimate_relevant_skills(resume_text, top_job, skills)
    level_chart_path = generate_skill_bar_chart(skill_scores)

    #🎓 Generate enriched recommendations
    recommendations = []
    skill_explanations = {
        "python": "This role involves scripting and automation.",
        "machine learning": "Core to building intelligent systems.",
        "flask": "Used for developing APIs in this job.",
        "sql": "Data querying is frequently mentioned.",
        "communication": "Important for cross-functional collaboration.",
        "leadership": "This role may involve team coordination.",
        "docker": "Used for containerizing models or apps.",
        "cloud computing": "Job demands cloud deployment experience."
        # add more if needed
    }

    for skill in skill_resources:
        in_resume = skill in resume_text
        in_job = skill in top_job

        if in_job:
            label = "🔍 Missing Skill" if not in_resume else "🚀 Advance Skill"
            reason = skill_explanations.get(skill, "Required or beneficial for this role.")
            recommendations.append(f"<strong>📌 {skill.upper()}</strong> <em>— {label}</em><br><small>🧾 {reason}</small><br>")

            for link in skill_resources[skill]["courses"]:
                name = link.split('//')[1]
                recommendations.append(f"&nbsp;&nbsp;&nbsp;📚 <a href='{link}' target='_blank'>{name}</a><br>")

            for proj in skill_resources[skill]["projects"]:
                name = proj.split('//')[1]
                recommendations.append(f"&nbsp;&nbsp;&nbsp;💻 <a href='{proj}' target='_blank'>{name}</a><br>")

    #  Simple career trajectory suggestion
    career_paths = {
        "ml engineer": "🔜 Next: Senior ML Engineer → AI Architect",
        "data analyst": "🔜 Next: BI Analyst → Data Scientist",
        "backend developer": "🔜 Next: Software Engineer → Technical Lead"
    }

    matched_title = matched_jobs.iloc[0]['job_title'].lower()
    suggestion = None
    for role in career_paths:
        if role in matched_title:
            suggestion = career_paths[role]
            break

    #  Render elegant output
    return render_template_string('''
        <html>
        <head>
            <title>JobFit Results</title>
            <style>
                body {
                    font-family: 'Segoe UI', sans-serif;
                    background-color: #f8f9fa;
                    color: #333;
                    padding: 20px;
                }
                h2, h3 {
                    color: #0077cc;
                    text-align: center;
                }
                ul {
                    list-style-type: none;
                    padding-left: 0;
                }
                li {
                    background: #fff;
                    margin: 10px auto;
                    padding: 10px 15px;
                    max-width: 700px;
                    border-radius: 5px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }
                img {
                    display: block;
                    margin: 20px auto;
                    border-radius: 10px;
                    box-shadow: 0 1px 5px rgba(0,0,0,0.1);
                }
                a {
                    color: #0077cc;
                    text-decoration: none;
                }
                .back-link {
                    text-align: center;
                    margin-top: 30px;
                }
            </style>
        </head>
        <body>
            <h2>🔍 Top Job Matches</h2>
            <ul>{% for job in jobs %}<li>{{job}}</li>{% endfor %}</ul>

            <h3>📊 Skill Gap Radar</h3>
            <img src="/static/skill_gap.png" width="400">

           <h3>🔎 Skill Level Estimates</h3>
           <img src="/static/skill_levels.png" width="500">
            {% if suggestion %}
                <h3>📈 Career Trajectory</h3>
                <p style="text-align:center;">{{ suggestion }}</p>
            {% endif %}

            <h3>🎓 Recommendations</h3>
            <ul>{% for rec in recs %}<li>{{rec|safe}}</li>{% endfor %}</ul>

            <div class="back-link">
                <a href="/">⬅️ Back to JobFit Engine</a>
            </div>
        </body>
        </html>
    ''', jobs=matched_jobs['job_title'].tolist(), recs=recommendations, levels=skill_scores, suggestion=suggestion, level_chart=level_chart_path)

if __name__ == '__main__':
    print("🚀 Launching Skill Gap Analyzer at http://127.0.0.1:5000")
    app.run(debug=True)