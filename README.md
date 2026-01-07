# resume_job_fit_classification
Capstone project for ML Zoomcamp: a supervised machine learning model that predicts candidate-job compatibility by classifying resumes as Good Fit or Bad Fit for a given job description.

## Business Goal
Companies spend countless hours screening resumes for open positions. This project aims to automate the initial screening process by predicting whether a candidate is a good fit for a specific job, helping HR teams save time and focus on top candidates.

**Key performance goals:**  
- **Positive class recall:** Ensure good candidates are not missed.  
- **Negative class precision:** Avoid wasting time on candidates unlikely to fit.

## Project Objective
- Build a machine learning pipeline that transforms resumes and job descriptions into features suitable for classification.
- Predict the probability of a candidate being a **Good Fit** for a job.
- Deploy the model as a **REST API** so it can be used in real HR systems or web applications.
- Optimize for **high recall on Good Fit candidates** and **high precision on No Fit candidates**.

## Dataset
- **Source:** [BWBayu Job CV Supervised Dataset](https://huggingface.co/datasets/bwbayu/job_cv_supervised)
- **Structure:** Each row contains:
  - `resume_text` – cleaned text of the candidate's resume
  - `job_description_text` – cleaned text of the job description
  - `label` – binary indicator: 1 = Good Fit, 0 = No Fit
- **Size:** ~ 31,203 rows
- **Notes:** Dataset was cleaned and preprocessed for text analysis.

## Data Exploration
**Class distribution:**

| Label     | Proportion |
|----------|------------|
| 1 (Good Fit) | 0.504      |
| 0 (No Fit)   | 0.496      |

**Resume and Job Description lengths (in characters):**

| Feature       | Count  | Mean     | Std       | Min   | 25%    | 50%    | 75%    | Max     |
|---------------|--------|---------|-----------|-------|--------|--------|--------|---------|
| resume_len    | 31203  | 3892.87 | 3108.46   | 112   | 1638   | 3502   | 5039   | 21044   |
| job_len       | 31203  | 1609.14 | 1310.82   | 142   | 655    | 1122   | 2127   | 6080    |

- **Unique counts:**
  - `resume_text`: 637
  - `job_description_text`: 452
  - `combined_text`: 31,203 (each row is unique)

## Modeling

### Feature Engineering
- **Separate TF-IDF vectorization** for:
  - Resume text
  - Job description text
  - → Used to calculate **cosine similarity** between resume and job description
- **Combined TF-IDF vectorization** for the concatenated text of resume + job description
  - → Used as features in the classification model

 ### 🛠 Modeling Experiments (Hyperparameter Tuning)

This table summarizes the **models tested, their tuned hyperparameters**, and the **main metrics** used to guide model selection.  

| Model               | Hyperparameters                                      | AUC  | Positive Recall (Good Fit) | Negative Precision (No Fit) |
|--------------------|------------------------------------------------------|------|----------------------------|-----------------------------|
| Logistic Regression | C = 1                                               | 0.91 | 0.96                       | 0.91                        |
| Decision Tree       | max_depth = 4, min_samples_leaf = 100               | 0.75 | 0.58                       | 0.65                        |
| Random Forest       | n_estimators = 50, max_depth = 30                  | 0.92 | 0.97                       | 0.94                        |
| XGBoost (final/test)| eta = 0.15, max_depth = 6, min_child_weight = 2, num_boost_round = 400 | 0.91 | 0.97                       | 0.96                        |

**Notes:**  
- Positive class recall (Good Fit) was prioritized to avoid missing strong candidates.  
- Negative class precision (No Fit) was important to reduce unnecessary interviews.  
- The **XGBoost classifier** was selected as the final model for deployment due to its strong balance between these metrics.  
- Hyperparameters for other models were tuned and evaluated on validation data.

## 📊 Results (Detailed Metrics)

The table below provides **per-class performance metrics** for each model, showing **precision, recall, and F1-score** for both classes, along with AUC. XGBoost metrics are reported on the **final test set**.  

### Model Performance Summary

| Model               | Dataset      | Class 0 Precision | Class 0 Recall | Class 0 F1 | Class 1 Precision | Class 1 Recall | Class 1 F1 | AUC  |
|--------------------|-------------|-----------------|---------------|------------|-----------------|---------------|------------|------|
| Logistic Regression | Validation  | 0.91            | 0.41          | 0.56       | 0.59            | 0.96          | 0.73       | 0.91 |
| Decision Tree       | Validation  | 0.65            | 0.78          | 0.71       | 0.73            | 0.58          | 0.65       | 0.75 |
| Random Forest       | Validation  | 0.94            | 0.44          | 0.60       | 0.63            | 0.97          | 0.77       | 0.92 |
| XGBoost (final)    | Test        | 0.96            | 0.53          | 0.68       | 0.65            | 0.97          | 0.78       | 0.91 |


**Key Takeaways:**  
- **XGBoost** offers the best combination of **high recall for Good Fit candidates** and **good precision for No Fit candidates**, aligning with the project's HR objectives.  
- Other models performed reasonably on validation but did not balance these metrics as effectively.  
- This demonstrates why XGBoost was chosen for **deployment as a REST API** for automated resume screening.


## Usage

This project exposes a **REST API** that predicts whether a candidate is a **Good Fit** or **No Fit** for a given job description based on resume content.

The service is fully containerized using **Docker** for easy setup and reproducibility.

---

### Prerequisites

- Docker installed
- Git installed

---

### Step 1: Clone the Repository

Open the terminal and run:

```bash
git clone https://github.com/zeinasamer/resume_job_fit_classification.git
cd resume_job_fit_classification

Or visit the repo: (https://github.com/zeinasamer/resume_job_fit_classification)

### Step 2: Build the Docker Image

Build the Docker image and install all dependencies defined in `pyproject.toml` and `uv.lock`.

```bash
docker build -t resume-job-fit-api .

### Step 3: Run the Docker Container

Run the container and expose the API on port **9696**.

```bash
docker run -p 9696:9696 resume-job-fit-api .

The API is now running and accessible at http://localhost:9696.

## System Architecture

This project is designed for real-world usage, allowing the machine learning model to be easily integrated into HR systems or web applications.

### Architecture Flow

1. **HR or Client System**
   - Sends a JSON request containing:
     - Candidate resume text
     - Job description text

2. **FastAPI REST API**
   - Receives the request at the `/predict` endpoint
   - Converts the incoming JSON payload into a Pandas DataFrame
   - Passes the data through the pre-trained machine learning pipeline
   - Returns a JSON response containing:
     - `fit_probability` — probability that the candidate is a good fit
     - `prediction` — categorical label (`"Good Fit"` or `"Bad Fit"`)

3. **Machine Learning Pipeline**
   - Applies TF-IDF vectorization to:
     - Resume text
     - Job description text
   - Computes cosine similarity between resume and job description
   - Uses an **XGBoost classifier** to predict candidate–job compatibility

4. **HR or Client System**
   - Receives the prediction response
   - Integrates results into dashboards, reports, or candidate selection workflows


## Tech Stack

- Python 3.12

- Libraries: Pandas, NumPy, Scikit-learn, XGBoost, Joblib, FastAPI, Uvicorn

- Deployment: Docker
