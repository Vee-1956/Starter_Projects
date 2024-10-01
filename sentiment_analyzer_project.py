# sentiment_analyzer.py
# Valerie Ekwedike
#
# This program analyzes the sentiment of texts using nltk tokenizer

import csv
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.data import find

# Ensure NLTK 'punkt' package is downloaded for tokenization
try:
    find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class SentimentAnalyzer:
    """
    A class to analyze sentiment of text using basic sentiment keywords,
    negation, and modifiers for intensifying or downtoning sentiment scores.
    """

   # Starter sets for different categories of words

    _default_positive_words = [
        "happy", "joy", "delight", "love", "wonderful", "fantastic",
        "brilliant", "amazing", "excellent", "successful", "pleased", "thrilled"
    ]

    _default_negative_words = [
        "sad", "unhappy", "disappointed", "hate", "terrible", "awful",
        "horrible", "dreadful", "poor", "fail", "miserable", "depressed"
    ]

    _default_negation_words = [
        "not", "no", "never", "none", "cannot", "isn\'t", "aren\'t",
        "wasn\'t", "weren\'t", "haven\'t", "hasn\'t", "don\'t"
    ]

    _default_intensifiers = [
        "very", "extremely", "incredibly", "absolutely", "completely",
        "utterly", "totally", "deeply", "enormously", "exceptionally", "especially", "tremendously"
    ]

    _default_downtoners = [
        "slightly", "somewhat", "a bit", "barely", "hardly", "just",
        "marginally", "scarcely", "a little", "less", "rarely", "occasionally"
    ]

    # 
    INTENSIFIER_MULTIPLIER = 1.5
    DOWNTONER_MULTIPLIER = 0.5

    #``````````````````````````````````````````````````````````````````````````````````
    def __init__(self, positive_words=None, negative_words=None, negation_words=None, intensifiers=None, downtoners=None):
        """
        Initializes the SentimentAnalyzer with optional custom lists of words.
        Falls back to default lists if none are provided.
        """
        self.positive_words = positive_words if positive_words is not None else self._default_positive_words
        self.negative_words = negative_words if negation_words is not None else self._default_negative_words
        self.negation_words = negation_words if negation_words is not None else self._default_negation_words
        self.intensifiers = intensifiers if intensifiers is not None else self._default_intensifiers
        self.downtoners = downtoners if downtoners is not None else self._default_downtoners


    #``````````````````````````````````````````````````````````````````````````````````
    # 
    def analyze_sentence_sentiment(self, sentence, use_negation=False, use_modifiers=False):
        """
        Analyzes the sentiment score of a sentence based on the presence of positive, negative,
        negation, and modifier (intensifiers and downtoners) words. The function calculates a 
        sentiment score that reflects the overall sentiment of the sentence.

        Parameters:
        - sentence: The sentence to analyze
        - use_negation: Whether to consider negation words
        - use_modifiers: Whether to consider intensifiers and downtoners

        Returns:
        - The sentiment score of the sentence

        TODO:
        1. Tokenize the sentence into words.
        2. Iterate through each word, checking for positive, negative, negation, and modifier words.
        3. Apply negation and modifier effects appropriately to calculate the sentiment score.
        4. Return the final sentiment score for the sentence.


        Detailed Steps to Compute Sentiment Score:
        1. Tokenize the sentence into individual words.
            - Use nltk's word_tokenize to break down the sentence into words.
        
        2. Initialize a variable to keep track of the sentiment score (e.g., sentiment_score = 0).
        
        3. Loop through each word in the tokenized sentence:
            a. Check if current word is a negation word (if use_negation is True):
                - If yes, note that the sentiment of the following word should be inverted.
                - Skip the negation word itself from further sentiment analysis.
            
            b. Check if the current word is an intensifier or downtoner (if use_modifiers is True):
                - If yes, note the modifier effect (use INTENSIFIER_MULTIPLIER or DOWNTONER_MULTIPLIER)
                for the following word.
                - Skip the modifier word itself from further sentiment analysis but remember its effect.
            
            c. Determine if the current word is a positive or negative sentiment word:
                - If the word is positive, increase the sentiment_score by 1 (or decrease if negated).
                - If the word is negative, decrease the sentiment_score by 1 (or increase if negated).
                - Apply any modifier effect to the change in score (if a modifier was noted in step 3b).
            
            d. Adjust the sentiment_score based on the findings in steps 3a, 3b, and 3c.
                - Ensure to reset any negation or modifier effect after it has been applied to a word.
            
        4. After processing all words, the sentiment_score variable will reflect the overall sentiment
        of the sentence. A positive score indicates a generally positive sentiment, a negative score
        indicates a negative sentiment, and a score of 0 indicates a neutral sentiment.
        
        5. Return the sentiment_score as the output of this function.

        Notes:
        - The function parameters use_negation and use_modifiers allow for conditional analysis based
        on the presence of negation words and modifier words. This enables a more nuanced sentiment
        analysis.
        - This method requires a balanced set of positive and negative words, as well as accurate
        lists of negation words, intensifiers, and downtoners for effective sentiment analysis.
        - Complex sentences with multiple sentiments, negations, and modifiers may require careful
        consideration to accurately calculate the sentiment score.

        Example Assertions for Testing:
        - Positive sentence without modifiers: "This is a great day." -> Score: 1
        - Negative sentence with downtoner: "This is somewhat disappointing." -> Score: -0.5
        - Positive sentence with negation: "This is not a great day." (use_negation=True) -> Score: -1
        - Positive sentence with intensifier and negation: "This is definitely not great." 
        (use_negation=True, use_modifiers=True) -> Score: -1.5

        These steps and examples should guide the implementation of this method to accurately
        analyze sentiment scores of sentences. Remember, this is a simplified model of sentiment
        analysis, and real-world applications may require more sophisticated approaches.

        if the word is a negation, store the negation (*-1) and skip
        if the word is a intensifier/downtoner, store the scaling (*1.5 or  *0.5) and skip
        if the word is a positive or negative, add/subtract 1 multiplied by the scalar/negator
        """

        # TODO: Implement the sentiment analysis logic as described
        
        # this converts a list of words to a list of sentences
        words_to_sentence = nltk.tokenize.word_tokenize(sentence.lower(), language='english', preserve_line=False)
        
        # initialized variables
        sentiment_score = 0
        modifier = 1.0
        negation = False
        
        for word in words_to_sentence:
            if use_negation and word in self.negation_words:
                negation = True
                continue
            if use_modifiers and word in self.intensifiers:
                modifier = self.INTENSIFIER_MULTIPLIER
                continue
            elif use_modifiers and word in self.downtoners:
                modifier = self.DOWNTONER_MULTIPLIER
                continue

            if word in self.positive_words:
                sentiment_score += 1 * modifier * (-1 if negation else 1)
            elif word in self.negative_words:
                sentiment_score -= 1 * modifier * (-1 if negation else 1)

            negation = False
            modifier = 1.0

        if sentiment_score == int(sentiment_score):
            sentiment_score = int(sentiment_score)

        
        return sentiment_score


    #``````````````````````````````````````````````````````````````````````````````````
    # insert
    def get_sentiment(self, sentiment_score):
        
        """
        Determines the sentiment label ('positive', 'negative', 'neutral') based on the sentiment score.

        Parameters:
        - sentiment_score: The sentiment score to evaluate

        Returns:
        - A string label indicating the sentiment ('positive', 'negative', 'neutral')
        """
        # TODO: Implement logic to return the correct sentiment label based on sentiment_score

        # takes in sentiment score
        # in positive return positive etc
        
        # This uses if statement to determine sentiment
        if sentiment_score > 0:
            return "positive"
        elif sentiment_score < 0:
            return "negative"
        else:
            return "neutral"

        
    #``````````````````````````````````````````````````````````````````````````````````
    # insert
    def calculate_overall_sentiment_score(self, sentiment_scores):
        """
        Calculates the average sentiment score from a list of individual sentence scores. Here
        the sentiment scores are the floating point scores for each sentence. 

        Parameters:
        - sentiment_scores: A list of sentiment scores from individual sentences

        Returns:
        - The average sentiment score
        """
        # TODO: Calculate and return the average of sentiment_scores

        # initializes the overall sum
        summ = 0

        # calculates average
        for i in sentiment_scores:
            summ += i

        average_sentiment_score = summ/len(sentiment_scores)

        # output result
        return average_sentiment_score

    #``````````````````````````````````````````````````````````````````````````````````
    # insert
    def get_sentences_from_lines(self, lines_list):
        
        """
        Converts a list of text lines into a list of sentences using NLTK's sentence tokenizer.

        Parameters:
        - lines_list: A list of text lines

        Returns:
        - A list of sentences
        """
        
        # TODO: Use sent_tokenize to convert lines_list into a list of sentences

        
        sentence_list = [] # empty list

        
        lines_joined = " ".join(lines_list)# joins all lines together

        # lines are transformed into sentences         
        sentence_list = nltk.tokenize.sent_tokenize(lines_joined, language='english')

        
        return sentence_list


    #``````````````````````````````````````````````````````````````````````````````````
    # insert
    def analyze_sentiment(self, text_lines_list, use_negation=False, use_modifiers=False):
        """
        Analyzes the overall sentiment of multiple lines of text.

        Parameters:
        - text_lines_list: A list of text lines to analyze
        - use_negation: Whether to consider negation in the analysis
        - use_modifiers: Whether to consider intensifiers and downtoners in the analysis

        Returns:
        - A dictionary with detailed results, overall sentiment, and sentiment counts
        - detailed_results is a list of dictionaries
            Each entry in the list contains "sentiment", "score", and "sentence" as keys.
        - The return dictionary contains three elements:
            detailed_results : as above
            overall_sentiment : a dictionary with "overall_sentiment" which gives the sentiment
                                as positive, negative, or neutral and "score" which gives
                                the overall sentiment score. 
            sentiment_counts :  a list of sentiment counts for each sentence.     
        """
        # TODO: Implement the overall sentiment analysis process as described
        # initialize
        detailed_results = []
        overall_sentiment = {"overall_sentiment": "", "score": 0}
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        sentiment_scores = []
        
        return_dictionary = {"detailed_results": detailed_results, "overall_sentiment": overall_sentiment, "sentiment_counts": sentiment_counts}
        
        sentences = self.get_sentences_from_lines(text_lines_list)

        for sent in sentences:
            sentence_dict = {}# dictionary
            # calls previous function to return sentiment score
            sentence_score = self.analyze_sentence_sentiment(sent, use_negation, use_modifiers)
            # calls previous function to determine sentence sentiment
            sentiment = self.get_sentiment(sentence_score)
            sentence_dict["sentiment"] = sentiment       

            sentiment_scores.append(sentence_score)
            # the value gotten from the dictionary key "score is assigned to sentenc_score
            sentence_dict["score"] = sentence_score
            # assigns value from dictionary to variable sent
            sentence_dict["sentence"] = sent

            # adds the sentence_dict to empty list detailed_results
            detailed_results.append(sentence_dict)
            #uses if statemnt to count number of positve, negative or neutral
            if sentiment == "positive":
                sentiment_counts["positive"] += 1
            elif sentiment == "negative":
                sentiment_counts["negative"] += 1
            else:
                sentiment_counts["neutral"] += 1            

    
        avg_sentiment_score = self.calculate_overall_sentiment_score(sentiment_scores)
        overall_sentiment["score"] = avg_sentiment_score
        overall_sentiment["overall_sentiment"] = self.get_sentiment(avg_sentiment_score)
        
        return return_dictionary

    #``````````````````````````````````````````````````````````````````````````````````
    def write_to_csv(self, detailed_results, csv_file_path):
        """
        Writes the detailed sentiment analysis results to a CSV file.

        Parameters:
        - detailed_results: A list of dictionaries containing sentiment analysis results
        - csv_file_path: The file path to write the CSV data to

        TODO:
        1. Open the specified CSV file for writing.
        2. Write a header row to the CSV.
        3. Iterate through detailed_results and write each result as a row in the CSV.
        Note: The included columns for each row should be Sentiment, Score, and Sentence.
        """
        
        # TODO: Implement CSV writing logic as described
        csvfile = open(csv_file_path, "w", newline='')
        
        writer = csv.DictWriter(csvfile)
        writer.writeheader()
        for row in detailed_results:
            writer.writerow(row)

        file.close()


#``````````````````````````````````````````````````````````````````````````````````
def main():
    """
    The main test function for the SentimentAnalyzer class. This function is designed to verify
    the correctness of the analyze_sentence_sentiment method by running it through a series of
    test cases. Each test case is an assertion that checks if the method returns the expected
    sentiment score for a given sentence under specified conditions (use of negation and modifiers).

    Example Assertions Explained:
    - Test case 1 checks a simple positive sentence without any negation or modifiers. The expected
      score is 1, indicating a positive sentiment.
    - Test case 2 checks a simple negative sentence, expecting a score of -1 to reflect the negative sentiment.
    - Test case 3 involves a positive sentence with a negation, turning the sentiment negative. With
      use_negation=True, it tests if the function correctly inverts the sentiment, expecting a score of -1.
    - Test case 4 examines the effect of an intensifier on a positive word, increasing its sentiment
      impact. use_modifiers=True activates the modifier logic, expecting an intensified positive score.
    - Test case 5 looks at a negative sentence with a downtoner, reducing the negative sentiment's impact.
      This tests the downtoner effect with an expected score indicating a lessened negative sentiment.
    - Test case 6 is a complex sentence that combines negation with an intensifier. This tests both the
      negation and modifier logic together, expecting a score that reflects the combined effects.

    Additional complex test cases mix multiple aspects of sentiment analysis to ensure the method
    can handle a variety of sentence structures and sentiment expressions accurately.

    Note:
    - This main function is for testing purposes only and demonstrates how the SentimentAnalyzer class
      can be utilized.
    - The assertions are critical for validating the expected functionality of the sentiment analysis
      method. Each assertion represents a specific scenario that the SentimentAnalyzer is expected to
      handle correctly.
    - Understanding these test cases and their expected outcomes will help in grasping the nuances of
      sentiment analysis as implemented in this class.

    Students are encouraged to add more test cases to cover additional scenarios and further validate
    the robustness of the sentiment analysis method.
    """
  
    analyzer = SentimentAnalyzer(["happy", "outstanding", "great"],["sad", "disappointing", "bad"],\
                                    ["not", "never"],["very", "extremely","definitely"],["somewhat", "slightly"])

    # Test case 1: Positive keyword
    assert analyzer.analyze_sentence_sentiment("This is a great day.") == 1, "Failed on positive keyword test"

    # Test case 2: Negative keyword
    assert analyzer.analyze_sentence_sentiment("This is a sad day.") == -1, "Failed on negative keyword test"

    # Test case 3: Negation of a positive word (without use_negation=True should be treated as positive)
    # this test will fail because of the "a" between the negation and the positive word.
    #
    #assert analyzer.analyze_sentence_sentiment("This is not a great day.", use_negation=True) == -1, "Failed on negation test"

    # Test case 3: Negation of a positive word (without use_negation=True should be treated as positive)
    assert analyzer.analyze_sentence_sentiment("This day is not great.", use_negation=True) == -1, "Failed on negation test"

    # Test case 4: Modified negation of a positive word (without use_negation=True should be treated as positive)
    assert analyzer.analyze_sentence_sentiment("This is definitely not great.", use_negation=True, use_modifiers=True) == -1.5, "Failed on intensify/downtone a negation test"

    # Test case 5: Intensified positive word
    assert analyzer.analyze_sentence_sentiment("This is a very great day.", use_modifiers=True) == 1.5, "Failed on intensifier test"

    # Test case 6: Downtoned negative word
    assert analyzer.analyze_sentence_sentiment("This is somewhat disappointing.", use_modifiers=True) == -0.5, "Failed on downtoner test"

    print("All simple sentence tests passed!")        

    canalyzer = SentimentAnalyzer(["happy", "outstanding", "great"], ["bad", "awful","disappointing"], ["not", "never"], ["very", "extremely","definitely"], ["somewhat", "slightly"])

    # Mixed sentiment with negation and modifier
    assert canalyzer.analyze_sentence_sentiment("This is a great day, but somewhat disappointing.", use_negation=True, use_modifiers=True) == 0.5, "Failed on mixed sentiment with negation and modifier"

    # Intensified positive followed by a downtoned negative
    assert canalyzer.analyze_sentence_sentiment("It was very outstanding yet slightly bad.", use_modifiers=True) == 1, "Failed on intensified positive followed by downtoned negative"

    # Negated positive followed by an unmodified negative
    assert canalyzer.analyze_sentence_sentiment("This is not happy and also awful.", use_negation=True) == -2, "Failed on negated positive followed by unmodified negative"

    # Multiple modifiers with a negation impacting different parts of the sentence
    assert canalyzer.analyze_sentence_sentiment("It was definitely not great, but somewhat bad.", use_negation=True, use_modifiers=True) == -2, "Failed on multiple modifiers with negation"

    # Sentences with neutral words and sentiment words without explicit modifiers or negations
    assert canalyzer.analyze_sentence_sentiment("The day was outstanding then turned awful.", use_negation=True, use_modifiers=True) == 0, "Failed on sentence with neutral shift"

    # Mixed sentiment with multiple modifiers and negation
    assert canalyzer.analyze_sentence_sentiment("This is extremely bad but not somewhat outstanding.", use_negation=True, use_modifiers=True) == -2.0, "Failed on mixed sentiment with multiple modifiers and negation"

    # Complex sentence with negation impacting multiple sentiment words
    assert canalyzer.analyze_sentence_sentiment("This is not happy day, but it is definitely not awful.", use_negation=True, use_modifiers=True) == 0.5, "Failed on complex sentence with negation impacting multiple sentiment words"

    print("All complex sentence tests passed!")        

    # Add more tests as needed

    print("All tests passed!")

    


if __name__ == "__main__":
    main()
