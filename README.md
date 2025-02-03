# Fake Job Postings Detection

## Overview
This project involves detecting fraudulent job postings using machine learning models. The dataset used is `fake_job_postings.csv`, which contains various attributes related to job postings, such as job title, location, company profile, description, requirements, benefits, and labels indicating whether a job posting is fraudulent or not. The goal is to analyze the data and build models to classify job postings as real or fake.

## Project Structure
1. **Importing Data**
   - Necessary libraries such as `pandas`, `numpy`, `seaborn`, `matplotlib`, and `sklearn` are imported.
   - The dataset is loaded using `pandas.read_csv()`.
   - Missing values are analyzed using `missingno`.

2. **Data Visualization**
   - Distribution of fraudulent vs. non-fraudulent job postings.
   - Distribution of job postings by employment type, required education, and experience.
   - Correlation heatmaps of numerical features.
   - Word cloud visualization of job descriptions.

3. **Data Preprocessing**
   - Categorical and numerical feature separation.
   - Handling missing values using `SimpleImputer`.
   - One-hot encoding categorical features.
   - Splitting data into training and test sets.

4. **Model Training and Evaluation**
   - **Random Forest Classifier:**
     - Trained with different numbers of trees.
     - Achieved an accuracy of **99.48%**.
     - Top keywords influencing fraudulent job postings were extracted.
   - **Naive Bayes Classifier:**
     - Achieved an accuracy of **98.43%**.
     - Precision: **0.93**, Recall: **0.50**, F1 Score: **0.65**.
   - **Decision Tree Classifier:**
     - Achieved an accuracy of **99.42%**.
     - Precision: **0.99**, Recall: **0.93**, F1 Score: **0.96**.

5. **Key Findings**
   - Fraudulent job postings often contain certain keywords in company profiles and descriptions.
   - Specific industries and job functions are more prone to fraudulent postings.
   - STEM vs. Non-STEM fraudulent job posting analysis showed varying trends.

## Usage
1. Install dependencies:
   ```sh
   pip install pandas numpy seaborn matplotlib scikit-learn nltk wordcloud missingno
   ```
2. Run the script:
   ```sh
   python fake_job_detection.py
   ```
3. Modify dataset file path if needed:
   ```python
   df = pd.read_csv('fake_job_postings.csv')
   ```

## Future Work
- Incorporate NLP techniques like TF-IDF and word embeddings.
- Experiment with deep learning models for classification.
- Deploy the model as a web service for real-time job posting verification.

## Author
**Vedant** - [GitHub](https://github.com/yourprofile)
