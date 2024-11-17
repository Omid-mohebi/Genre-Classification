# Genre Classification Model

This repository contains the code and methodology for building a text classification model to predict the genre of a text based on their titles and descriptions. The model utilizes natural language processing (NLP) techniques and machine learning algorithms to classify the text into one of nine genres.
This project spesially trained on books title and discriptions but it will work for all kind of text data.

## Table of Contents

- Overview
- Setup
- Usage
- Model Details
- Results

## Overview

This project aims to classify books into nine genres: Thriller, Classics, Romance, Mystery, Science, Literature, Fantasy, Historical, and Fiction. The model takes the title and description of each book as input and predicts the genre based on these text features. 

### Data
- **Train Data**: `train.csv` contains columns for the book title, description, and associated genre labels.
- **Test Data**: `test.csv` contains columns for the book title and description, but no genre labels (for prediction).

### Libraries Used
- `pandas`: For data manipulation and loading.
- `numpy`: For numerical operations.
- `nltk`: For text processing (tokenization, stopword removal, lemmatization).
- `sklearn`: For machine learning (RandomForestClassifier).

## Setup

### Prerequisites
To run this code, you need to have Python 3.6+ installed along with the following dependencies:

- `pandas`
- `numpy`
- `nltk`
- `scikit-learn`

### Installing Dependencies
You can install the required dependencies using `pip`:

```bash
pip install pandas numpy nltk scikit-learn
```

Additionally, you'll need to download the NLTK corpora:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

### File Structure

```
/data
  ├── train.csv
  └── test.csv
```

- `train.csv`: The training data with titles, descriptions, and genre labels.
- `test.csv`: The test data with titles and descriptions.

## Usage

1. **Load the Data**:
   The training and testing data are loaded from CSV files.

   ```python
   train_data = pd.read_csv('../data/train.csv')
   test_data = pd.read_csv('../data/test.csv')
   ```

2. **Text Preprocessing**:
   - Tokenization: Splitting titles and descriptions into individual words.
   - Stopword Removal: Removing common words like "and", "the", etc.
   - Lemmatization: Reducing words to their base form (e.g., "running" -> "run").
   
   This is done through the `tokenize()` function.

3. **Feature Vectorization**:
   The processed text is converted into a feature vector using a dictionary (`word2idx`) that maps each unique word to a unique index. A bag-of-words approach is used to represent each text as a sparse vector of word counts.

4. **Model Training**:
   The features are fed into a `RandomForestClassifier` to train the model. The model is trained on the training data (`x_train_vector`) and their corresponding genre labels (`y_train`).

   ```python
   model = RandomForestClassifier(n_jobs=2, random_state=0)
   model.fit(x_train_vector, y_train)
   ```

5. **Model Evaluation**:
   - The model achieves 99% accuracy on the training data and 75% on the test data.

6. **Prediction**:
   After training, the model predicts the genre for the test data. The predictions are stored in a DataFrame for submission.

   ```python
   py = model.predict(x_test_vector)
   ```

   The result is a DataFrame where each row contains the predicted probability for each genre.

## Model Details

- **Algorithm**: `RandomForestClassifier`
- **Features**: Bag-of-Words representation of titles and descriptions.
- **Number of Genres**: 9 (Thriller, Classics, Romance, Mystery, Science, Literature, Fantasy, Historical, Fiction)

### Performance
- **Training Accuracy (ACC)**: 99%
- **Test Accuracy (F1)**: 75%

The training accuracy is very high due to overfitting on the training data. However, the model still performs reasonably well on unseen test data with an accuracy of 75%.

## Results

After training the model and predicting on the test set, the results are stored in a DataFrame with predicted genre probabilities:

```python
submission = pd.DataFrame(dicti)
```

The final submission format contains the predicted probabilities for each of the nine genres.
