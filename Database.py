import glob
import os
from nltk.tokenize import RegexpTokenizer


class Database:
    def __init__(self, directory):
        self.dir = directory
        self.words_files = []
        for filename in glob.glob(os.path.join(directory, '*.txt')):
            with open(os.path.join(os.getcwd(), filename), 'r') as file:
                data = file.read().replace('\n', '')
                tokenizer = RegexpTokenizer(r'\w+')
                word_list = tokenizer.tokenize(data)
                self.words_files.append(word_list)

    def get_num_of_documents(self):
        return len(self.words_files)

    def get_num_of_documents_with_word(self, in_word):
        result = 0
        for file in self.words_files:
            for word in file:
                if word == in_word:
                    result += 1
                    break
        return result

    def get_maximum_frequency_of_word(self, in_word):
        result = 0
        for file in self.words_files:
            temp_result = 0
            for word in file:
                if word == in_word:
                    temp_result += 1
            if temp_result > result:
                result = temp_result
        return result
