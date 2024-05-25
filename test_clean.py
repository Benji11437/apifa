import pytest
from pretraitement import cleaned_text 
"""
Cette fonction test tous les cas de figure possible pour que 
la fonction returne un texte bien nettoyé.

""" 

def test_cleaned_text_basic():
    input_text = "Hello, World!"
    expected_output = "hello world"
    assert cleaned_text(input_text) == expected_output

def test_cleaned_text_with_numbers():
    input_text = "Python 3.9 is awesome!"
    expected_output = "python is awesome"
    assert cleaned_text(input_text) == expected_output

def test_cleaned_text_with_special_characters():
    input_text = "Good morning! Have a nice day :)"
    expected_output = "good morning have a nice day"
    assert cleaned_text(input_text) == expected_output

def test_cleaned_text_with_newlines():
    input_text = "Hello\nWorld\n"
    expected_output = "hello world"
    assert cleaned_text(input_text) == expected_output

def test_cleaned_text_multiple_spaces():
    input_text = "This    is a   test"
    expected_output = "this is a test"
    assert cleaned_text(input_text) == expected_output

def test_cleaned_text_empty_string():
    input_text = ""
    expected_output = ""
    assert cleaned_text(input_text) == expected_output

def test_cleaned_text_only_special_characters():
    input_text = "@#$%^&*()!"
    expected_output = ""
    assert cleaned_text(input_text) == expected_output

def test_cleaned_text_already_clean():
    input_text = "already clean"
    expected_output = "already clean"
    assert cleaned_text(input_text) == expected_output
