"""
Tests for data preparation module
"""

import pytest
import pandas as pd
from src.data_prep import clean_text, create_sentiment_label, get_sentiment_name


class TestCleanText:
    """Tests for the text cleaning function"""

    def test_removes_html_entities(self):
        """HTML codes like &#039; should be replaced with apostrophe"""
        text = "I don&#039;t like this"
        result = clean_text(text)
        assert "&#039;" not in result

    def test_removes_extra_whitespace(self):
        """Multiple spaces should become single space"""
        text = "This   has    extra   spaces"
        result = clean_text(text)
        assert "  " not in result

    def test_handles_empty_input(self):
        """Empty or None input should return empty string"""
        assert clean_text("") == ""
        assert clean_text(None) == ""
        assert clean_text(123) == ""

    def test_strips_quotes(self):
        """Leading/trailing quotes should be removed"""
        text = '"This is a review"'
        result = clean_text(text)
        assert not result.startswith('"')
        assert not result.endswith('"')

    def test_normal_text_unchanged(self):
        """Clean text should pass through without changes"""
        text = "This medication works great"
        result = clean_text(text)
        assert result == "This medication works great"


class TestSentimentLabel:
    """Tests for sentiment label creation"""

    def test_negative_ratings(self):
        """Ratings 1-4 should be labeled as negative (0)"""
        for rating in [1, 2, 3, 4]:
            assert create_sentiment_label(rating) == 0

    def test_neutral_ratings(self):
        """Ratings 5-6 should be labeled as neutral (1)"""
        for rating in [5, 6]:
            assert create_sentiment_label(rating) == 1

    def test_positive_ratings(self):
        """Ratings 7-10 should be labeled as positive (2)"""
        for rating in [7, 8, 9, 10]:
            assert create_sentiment_label(rating) == 2

    def test_sentiment_names(self):
        """Labels should map to correct names"""
        assert get_sentiment_name(0) == "negative"
        assert get_sentiment_name(1) == "neutral"
        assert get_sentiment_name(2) == "positive"

    def test_unknown_label(self):
        """Unknown label should return 'unknown'"""
        assert get_sentiment_name(99) == "unknown"