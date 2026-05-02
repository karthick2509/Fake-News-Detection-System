import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def clean_text(text):
    """
    Cleans text by:
    1. Removing non-alphabetic characters
    2. Converting to lowercase
    3. Tokenizing
    4. Removing stopwords
    5. Applying Stemming
    """
    if not isinstance(text, str):
        return ""
    
    # Initialize PorterStemmer
    ps = PorterStemmer()
    
    # Remove non-alphabetic characters and convert to lowercase
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    
    # Tokenize and remove stopwords, then stem
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    
    # Join back into a string
    return ' '.join(review)

if __name__ == "__main__":
    sample = "The quick brown fox jumps over the lazy dog! 123 @#$"
    print(f"Original: {sample}")
    print(f"Cleaned: {clean_text(sample)}")
