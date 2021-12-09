import math
from operator import itemgetter

from nltk import RegexpTokenizer
from nltk import tokenize


class Document:
    def __init__(self, database, filepath):
        self.database = database
        self.map_word_counter = {}
        self.map_word_weights = {}
        self.sentences = []
        with open(filepath, 'r') as file:
            data = file.read().replace('\n', '')
            tokenizer = RegexpTokenizer(r'\w+')
            self.sentences = tokenize.sent_tokenize(data)
            for word in tokenizer.tokenize(data):
                if self.map_word_counter.get(word):
                    self.map_word_counter[word] += 1
                else:
                    self.map_word_counter[word] = 1

    def calculates_weights(self):
        for word, num in self.map_word_counter.items():
            max_freq = self.database.get_maximum_frequency_of_word(word)
            if max_freq == 0:
                max_freq = num
            doc_count = self.database.get_num_of_documents_with_word(word)
            in_log = (self.database.get_num_of_documents() + 1) / (doc_count + 1)
            self.map_word_weights[word] = 0.5 * (
                    1 + num / max_freq) * math.log(
                in_log)

    def generate_classic(self):
        tokenizer = RegexpTokenizer(r'\w+')
        sentence_value_list = []
        index = 0
        for sentence in self.sentences:
            sentence_w = 0.0
            for word in tokenizer.tokenize(sentence):
                sentence_w += self.map_word_weights[word]
            sentence_value_list.append([sentence, sentence_w, index])
            index += 1
        result = sorted(sentence_value_list, key=itemgetter(1), reverse=True)
        result = result[:4]
        result = sorted(result, key=itemgetter(2))
        result_str = ''
        for sen, val, value in result:
            result_str += sen + '\n'
        return result_str

    def generate_words_referat(self):
        my_list = []
        for word, w in self.map_word_weights.items():
            my_list.append([word, w])
        result = sorted(my_list, key=itemgetter(1), reverse=True)
        result_str = ""
        for word, w in result[:10]:
            result_str += word + '\n'
        return result_str

