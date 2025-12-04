# Amazon Echo Review Sentiment Analyzer 🎧🗣️

This project uses real Amazon Echo Dot reviews to train a **multiclass sentiment analysis model** that classifies reviews as:

- 😢 **Negative**
- 😎 **Neutral**
- ✅ **Positive**

The model is built using TF-IDF vectorization and multinomial logistic regression, and includes a full training pipeline, an inference module, and a separate Streamlit web application for interactive predictions.

## Overview

The goal of this project is to analyze real Amazon Echo Dot product reviews and predict whether each review expresses negative, neutral, or positive sentiment. The system uses Natural Language Processing (NLP) techniques to convert raw review text into numerical features the machine learning model can understand.

The project is structured as a portfolio-ready, end-to-end ML system with:

- A full training script  
- Clean text preprocessing  
- TF-IDF feature extraction  
- Multiclass sentiment classification  
- A Streamlit web app for live testing  
- Modular Python package layout  

## Dataset

This project uses Amazon Echo Dot review data with the following fields:

- Review Text  
- Rating (1–5 stars)  
- Review Date  
- User Verified  
- Device Color  
- Configuration  
- Page URL  

### Sentiment Label Mapping

| Rating | Label | Meaning |
|--------|--------|-----------|
| 1–2 | 0 | Negative |
| 3 | 1 | Neutral |
| 4–5 | 2 | Positive |

Place your dataset here:

```
data/amazon_alexa_reviews.csv
```

If the dataset is large, include only a small sample in the repository.

## Model Details

The model is built using:

- Text preprocessing: lowercasing, cleaning, removing non-letters  
- TF-IDF vectorizer with unigrams and bigrams  
- Logistic Regression using the multinomial option  
- Balanced class weights to improve accuracy for neutral reviews  
- Three output probabilities for each sentiment class  

The model produces output in the form:

```
[Negative Probability, Neutral Probability, Positive Probability]
```

## Project Structure

```
alexa-sentiment-analyzer/
├── app.py                        # Streamlit web app
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── amazon_alexa_reviews.csv  # dataset sample (optional)
├── models/
│   ├── amazon_echo_sentiment_model.joblib
│   └── tf_vectorizer.joblib
├── src/
│   ├── __init__.py
│   ├── train.py                  # training script
│   └── inference.py              # prediction logic
└── notebooks/
    └── 01_exploration.ipynb      # optional EDA notebook
```

## Installation

```
git clone https://github.com/patdal1810/alexa-sentiment-analyzer.git
cd alexa-sentiment-analyzer
```

Create a virtual environment:

```
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

Install packages:

```
pip install -r requirements.txt
```

## Training the Model

```
python -m src.train
```

## Running the Streamlit App

```
streamlit run app.py
```

## Example Predictions

**Input:**  
“This device is terrible. Doesn’t work at all.”

**Output:**  
- Negative 😢  
- Confidence: 92%

---

**Input:**  
“It’s okay, not great but not terrible either.”

**Output:**  
- Neutral 😎  
- Confidence: 68%

---

**Input:**  
“I absolutely love my Echo Dot!”

**Output:**  
- Positive ✅  
- Confidence: 96%

## Future Improvements

- Add word clouds per sentiment  
- Use BERT/DistilBERT  
- Batch CSV prediction  
- Deploy to Streamlit Cloud or Hugging Face  
- Add SHAP/LIME explainability  
- Build a FastAPI backend  

## License

MIT License.

## Credits

Developed as a machine learning educational project using Amazon Alexa review data.
