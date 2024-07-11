# Natural Language Processing Checkpoint
This checkpoint is designed to test your understanding of the content from the Text Classification Cumulative Lab. 

Specifically, this will cover:

- Preprocessing and exploring text data using `nltk`
- Vectorizing text data using a bag-of-words approach
- Fitting machine learning models using vectorized text data

### Data Understanding

In this repository under the file path `data/movie_descriptions.csv` there is a CSV file containing the titles, genres, and descriptions for 5,000 films pulled from [IMDb](https://www.kaggle.com/hijest/genre-classification-dataset-imdb).

**The features of interest for this analysis will be:**

1. `desc`: The description of the film, which we will explore and then use as the features of our model
2. `genre`: The target for our predictive model


```python
# Run this cell without changes
import pandas as pd

# Import the data
data = pd.read_csv('data/movie_descriptions.csv')

# Output a sample
data = data.sample(1500, random_state=100)
data.head()
```


```python
# Run this cell without changes
data.genre.value_counts()
```

### Requirements

1. Initialize tokenizer and stemmer objects to prepare for text preprocessing
2. Write a function that implements standard "bag of words" text preprocessing
3. Initialize and fit a `CountVectorizer` from `sklearn`
3. Vectorize data using `CountVectorizer`
4. Fit a decision tree classifier on vectorized text data

## 1) Initialize Tokenizer, Stemmer, and Stopwords Objects

In our exploratory text analysis, we will:

* Standardize case
* Tokenize (split text into words)
* Remove stopwords
* Stem words

Three of those steps require that we import some functionality from `nltk`. In the cell below, create:

* An instance of `RegexpTokenizer` ([documentation here](https://www.nltk.org/api/nltk.tokenize.regexp.html#module-nltk.tokenize.regexp)) called `tokenizer`
  * The regex pattern should select all words with three or more characters. You can use the pattern `r"(?u)\w{3,}"`
* A list of stopwords (documentation [here](https://www.nltk.org/api/nltk.corpus.html#module-nltk.corpus) and [here](https://www.nltk.org/nltk_data/)) called `stopwords_list`
* An instance of `PorterStemmer` ([documentation here](https://www.nltk.org/api/nltk.stem.porter.html)) called `stemmer`


```python
# Run this line in a new cell if nltk isn't working
# !pip install nltk

# Replace None with appropriate code

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Create an intance of the RegexpTokenizer with the variable name `tokenizer`
# The regex pattern should select all words with three or more characters
tokenizer = None

# Create a list of stopwords in English
stopwords_list = None

# Create an instance of nltk's PorterStemmer with the variable name `stemmer`
stemmer = None

# your code here
raise NotImplementedError
```


```python
# Checking that variables are no longer None
assert tokenizer
assert stopwords_list
assert stemmer

# PUT ALL WORK FOR THE ABOVE QUESTION ABOVE THIS CELL
# THIS UNALTERABLE CELL CONTAINS HIDDEN TESTS
```

## 2) Write a Function That Implements Standard Text Preprocessing

In the cell below, complete the `preprocess_text` function so the inputted text is returned lower cased, tokenized, stopwords removed, and stemmed.

For example, if you input the text

```
This is an example sentence for preprocessing.
```

The result of `preprocess_text` should be this list of strings:

```python
['exampl', 'sentenc', 'preprocess']
```


```python
def preprocess_text(text, tokenizer, stopwords_list, stemmer):
    # Standardize case (lowercase the text)
    # your code here
    raise NotImplementedError
    
    # Tokenize text using `tokenizer`
    # your code here
    raise NotImplementedError
    
    # Remove stopwords using `stopwords_list`
    # your code here
    raise NotImplementedError
    
    # Stem the tokenized text using `stemmer`
    # your code here
    raise NotImplementedError
    
    # Return the preprocessed text
    # your code here
    raise NotImplementedError
    
preprocess_text("This is an example sentence for preprocessing.", tokenizer, stopwords_list, stemmer)
```


```python
from types import FunctionType

assert type(preprocess_text) == FunctionType
assert type(preprocess_text('Example text', tokenizer, stopwords_list, stemmer)) == list
# PUT ALL WORK FOR THE ABOVE QUESTION ABOVE THIS CELL
# THIS UNALTERABLE CELL CONTAINS HIDDEN TESTS
```

Now that the function has been created, use it to preprocess the entire dataset:


```python
# Run this cell without changes
# (This may take a while due to nested loops)
text_data = data.desc.apply(lambda x: preprocess_text(x, tokenizer, stopwords_list, stemmer))
text_data
```


```python
# Run this cell without changes
data["preprocessed_text"] = text_data
data
```

Now let's take a look at the top ten most frequent words for each genre.


```python
# Run this cell without changes
import matplotlib.pyplot as plt
import seaborn as sns

# Set up figure and axes
fig, axes = plt.subplots(nrows=7, figsize=(12, 12))

# Empty dict to hold words that have already been plotted and their colors
plotted_words_and_colors = {}
# Establish color palette to pull from
# (If you get an error message about popping from an empty list, increase this #)
color_palette = sns.color_palette('cividis', n_colors=38)

# Creating a plot for each unique genre
data_by_genre = [y for _, y in data.groupby('genre', as_index=False)]
for idx, genre_df in enumerate(data_by_genre):
    # Find top 10 words in this genre
    all_words_in_genre = genre_df.preprocessed_text.explode()
    top_10 = all_words_in_genre.value_counts()[:10]
    
    # Select appropriate colors, reusing colors if words repeat
    colors = []
    for word in top_10.index:
        if word not in plotted_words_and_colors:
            new_color = color_palette.pop(0)
            plotted_words_and_colors[word] = new_color
        colors.append(plotted_words_and_colors[word])
    
    # Select axes, plot data, set title
    ax = axes[idx]
    ax.bar(top_10.index, top_10.values, color=colors)
    ax.set_title(genre_df.iloc[0].genre.title())
    
fig.tight_layout()
```

## 3) Fit a Count Vectorizer

Now that we have explored the data some, let's prepare it for modeling.

Before we fit a vectorizer to the data, we need to convert the list of tokens for each document back to a string datatype and create a train test split.


```python
# Run this cell without changes
from sklearn.model_selection import train_test_split

# Convert token lists to strings
data["joined_preprocessed_text"] = data["preprocessed_text"].str.join(" ")

# Create train test split
X_train, X_test, y_train, y_test = train_test_split(
    data["joined_preprocessed_text"], data.genre, test_size=0.3, random_state=2021)

X_train
```

**In the cell below, create a CountVectorizer instance ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)) with default arguments, called `vectorizer`, and fit it to the training data.**


```python
# Import the CountVectorizer object from sklearn
# your code here
raise NotImplementedError

# Create a `vectorizer` instance
# your code here
raise NotImplementedError

# Fit the vectorizer to the training data
# your code here
raise NotImplementedError
```


```python
assert vectorizer
# PUT ALL WORK FOR THE ABOVE QUESTION ABOVE THIS CELL
# THIS UNALTERABLE CELL CONTAINS HIDDEN TESTS
```

## 4) Vectorize the Data

In the cell below, vectorize the training and test datasets using the fitted count vectorizer.


```python
# Replace None with appropriate code

X_train_vectorized = None
X_test_vectorized = None
# your code here
raise NotImplementedError
```


```python
from scipy.sparse.csr import csr_matrix
assert type(X_train_vectorized) == csr_matrix
assert type(X_test_vectorized) == csr_matrix
# PUT ALL WORK FOR THE ABOVE QUESTION ABOVE THIS CELL
# THIS UNALTERABLE CELL CONTAINS HIDDEN TESTS
```

## 5) Fit a Decision Tree Model

In the cell below, 

- Create an instance of `sklearn`'s `DecisionTreeClassifier` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)), using default arguments, with the variable name `dt`
- Fit the model to the vectorized training data


```python
# Replace None with appropriate code

# Import DecisionTreeClassifier
None

# Initialize `dt`
dt = None

# Fit the model to the training data
None
# your code here
raise NotImplementedError
```


```python
assert dt
# PUT ALL WORK FOR THE ABOVE QUESTION ABOVE THIS CELL
# THIS UNALTERABLE CELL CONTAINS HIDDEN TESTS
```

The following code will now evaluate our model on the test data:


```python
from sklearn.metrics import plot_confusion_matrix
fig, ax = plt.subplots(figsize=(12,12))
plot_confusion_matrix(dt, X_test_vectorized, y_test, ax=ax, cmap="cividis");
```
