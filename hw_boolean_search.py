#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import codecs
import sys

import numpy as np
import pandas as pd
from transliterate import translit
import re
from collections import Counter

class Index:
    __translation__ = ["`qwertyuiop[]asdfghjkl;'zxcvbnm,.", "ёйцукенгшщзхъфывапролджэячсмитьбю"]
    __translation__ = [i.upper() for i in __translation__]
    __min_size__ = 1
    __stop_list__ = []
    words_occurences = None
    total_occurences = None

    @classmethod
    def get_weight(cls, word):
        return Index.words_occurences[word] / Index.total_occurences
    @classmethod
    def set_stop_list(cls, value):
        cls.__stop_list__ = value
        cls.__stop_list__ = [i.upper() for i in cls.__stop_list__]

    @classmethod
    def set_min_size(cls, value):
        cls.__min_size__ = value

    @classmethod
    def translate(cls, s):
        s = s.upper()
        return s

    @classmethod
    def __is_stop_word__(cls, word):
        if word in cls.__stop_list__ or len(word) < cls.__min_size__:
            return True
        return False

    def __index_column__(self, column, doc_id):
        for word in column:
            word = self.translate(word)
            if self.__is_stop_word__(word):
                continue
            if word not in self.__inverted_index__:
                self.__inverted_index__[word] = np.array([doc_id], dtype=int)
            else:
                self.__inverted_index__[word] = np.append(self.__inverted_index__[word], doc_id)

    def __init__(self, index_file):
        self.__inverted_index__ = dict()
        df = pd.read_csv(index_file, sep='\t', encoding='utf-8', names=['DocumentId', 'Title', 'Text'])
        def words(text):
            return re.findall(r'\w+', text.upper())

        Index.words_occurences = Counter(words(open(index_file).read()))
        Index.total_occurences = sum(Index.words_occurences.values())
        df['Title'] = df['Title'].apply(lambda x: x.split())
        df['Text'] = df['Text'].apply(lambda x: x.split())
        for index, row in df.iterrows():
            doc_id = int(row['DocumentId'][1:])
            self.__index_column__(row['Title'], doc_id)
            self.__index_column__(row['Text'], doc_id)
        for key in self.__inverted_index__.keys():
            self.__inverted_index__[key] = np.unique(self.__inverted_index__[key])
    def __correct_word__(self, word):
        if word.isdigit():
            return None
        word1 = translit(word, 'ru')
        if  word1 in self.__inverted_index__:
            return self.__inverted_index__[word1]
        word1 = translit(word, 'ru', reversed=True)
        if word1 in self.__inverted_index__:
            return self.__inverted_index__[word1]
        word1 = word.translate(str.maketrans(Index.__translation__[0], Index.__translation__[1]))
        if word1 in self.__inverted_index__:
            return self.__inverted_index__[word1]
        word1 = word.translate(str.maketrans(Index.__translation__[1], Index.__translation__[0]))
        if word1 in self.__inverted_index__:
            return self.__inverted_index__[word1]
        words = re.split(r'(\W+)', word)
        if words is None:
            print(word)
        res = self.__and_doc_list__(words)
        return res

    def __and_doc_list__(self, words):
        if len(words) == 0:
            return None
        elif len(words) == 1:
            if words[0] not in self.__inverted_index__:
                return None
            return self.__inverted_index__[words[0]]
        if words[0] not in self.__inverted_index__:
            return None
        res = self.__inverted_index__[words[0]]
        for word2 in words[1:]:
            if word2 not in self.__inverted_index__:
                return None
            res = np.sort(np.intersect1d(res, self.__inverted_index__[word2]), kind='mergesort')
        return res

    def get_doc_list(self, word):
        word = self.translate(word)
        if self.__is_stop_word__(word):
            return None
        if word not in self.__inverted_index__:
            res = self.__correct_word__(word)
            if res is None:
                return np.array([], dtype=int)
            return res
        return self.__inverted_index__[word]


class QueryTree:
    class Expression:
        class Tree:
            __and_op__ = " "
            __or_op__ = "|"

            @classmethod
            def set_or_op(cls, value="|"):
                cls.__or_op__ = value

            @classmethod
            def set_and_op(cls, value=" "):
                cls.__and_op__ = value

            def __init__(self, value, left, right):
                self.__value__ = value
                self.__left__ = left
                self.__right__ = right

            def build(self, index, limit):
                if self.__left__ is None and self.__right__ is None:
                    weight = index.get_weight(self.__value__)
                    res = index.get_doc_list(self.__value__)
                    return weight, res
                elif self.__left__ is None or self.__right__ is None:
                    raise Exception("1")
                elif self.__value__ == self.__and_op__:
                    weight_l, left = self.__left__.build(index, limit)
                    weight_r, right = self.__right__.build(index, limit)
                    if left is None and right is None:
                        return 0, None
                    elif left is None:
                        return weight_r, right
                    elif right is None:
                        return weight_l, left
                    else:
                        if weight_r - weight_l > limit:
                            return weight_r, right
                        elif weight_l - weight_r > limit:
                            return weight_l, left
                        else:
                            res = np.sort(np.intersect1d(left, right), kind='mergesort')
                        return min(weight_l, weight_r), res
                elif self.__value__ == self.__or_op__:
                    weight_l, left = self.__left__.build(index, limit)
                    weight_r, right = self.__right__.build(index, limit)
                    if left is None and right is None:
                        return 0, None
                    elif left is None:
                        return weight_r, right
                    elif right is None:
                        return weight_l, left
                    else:
                        res = np.unique(np.append(left, right))
                        return max(weight_l, weight_r), res
                else:
                    raise Exception("2")

        def __init__(self, query):
            self.__expression__ = query
            self.__get_current_sym__()
            self.__count__ = 0
            self.__sum__ = 0

        def __get_current_sym__(self):
            if len(self.__expression__) > 1:
                self.__curr__ = self.__expression__[0]
                self.__expression__ = self.__expression__[1:]
            elif len(self.__expression__) == 1:
                self.__curr__ = self.__expression__[0]
                self.__expression__ = ""
            else:
                self.__curr__ = None

        def expression(self):
            left_exp = self.__and_expression__()
            if self.__curr__ == self.Tree.__or_op__:
                self.__get_current_sym__()
                right_exp = self.expression()
                left_exp = self.Tree(self.Tree.__or_op__, left_exp, right_exp)
            return left_exp

        def __or_expression__(self):
            left_exp = self.__elem_of_expression__()
            if self.__curr__ == self.Tree.__or_op__:
                self.__get_current_sym__()
                right_exp = self.__or_expression__()
                left_exp = self.Tree(self.Tree.__or_op__, left_exp, right_exp)
            return left_exp

        def __and_expression__(self):
            left_exp = self.__elem_of_expression__()
            if self.__curr__ == self.Tree.__and_op__:
                self.__get_current_sym__()
                right_exp = self.__and_expression__()
                left_exp = self.Tree(self.Tree.__and_op__, left_exp, right_exp)
            return left_exp

        def __elem_of_expression__(self):
            if self.__curr__ == '(':
                self.__get_current_sym__()
                res = self.expression()
                if self.__curr__ == ')':
                    self.__get_current_sym__()
                    return res
                raise Exception("3")
            value = ""
            while (self.__curr__ != self.Tree.__and_op__ and self.__curr__ != self.Tree.__or_op__
                   and self.__curr__ != ')'
                   and self.__curr__ is not None):
                if self.__curr__ == '(':
                    raise Exception("4")
                value += self.__curr__
                self.__get_current_sym__()
            self.__sum__ += Index.get_weight(value)
            self.__count__ += 1
            return self.Tree(value, None, None)

        def get_mean(self):
            return self.__sum__ / self.__count__

    def __init__(self, qid, query):
        self.__id__ = qid
        exp = self.Expression(query)
        self.__tree__ = exp.expression()
        self.__mean__ = exp.get_mean()


    def search(self, index):
        res1 = self.__id__
        res2, res3 = self.__tree__.build(index, self.__mean__ / 2)
        return res1, res3


class SearchResults:
    def __init__(self):
        self.__query__ = dict()

    def add(self, found):
        self.__query__[found[0]] = found[1]

    def __binary_search__(self, key, x):
        i = np.searchsorted(self.__query__[key], [x])
        length = self.__query__[key].shape[0]
        if i != length and self.__query__[key][i] == x:
            return 1
        else:
            return 0

    def print_submission(self, objects_file, submission_file):
        df = pd.read_csv(objects_file, encoding='utf-8', sep=',')
        df['DocumentId'] = df['DocumentId'].apply(lambda x: int(x[1:]))
        res = pd.DataFrame()
        res['ObjectId'] = df['ObjectId']
        res['Relevance'] = df.apply(lambda x: self.__binary_search__(x.QueryId, x.DocumentId), axis=1)
        res.to_csv(submission_file, encoding='utf-8', sep=',', index=False)


def main():
    # Command line arguments.
    parser = argparse.ArgumentParser(description='Homework: Boolean Search')
    parser.add_argument('--queries_file', required=True, help='queries.numerate.txt')
    parser.add_argument('--objects_file', required=True, help='objects.numerate.txt')
    parser.add_argument('--docs_file', required=True, help='docs.tsv')
    parser.add_argument('--submission_file', required=True, help='output file with relevances')
    args = parser.parse_args()

    # Build index.
    index = Index(args.docs_file)

    # Process queries.
    search_results = SearchResults()
    with codecs.open(args.queries_file, mode='r', encoding='utf-8') as queries_fh:
        for line in queries_fh:
            fields = line.rstrip('\n').split('\t')
            qid = int(fields[0])
            query = fields[1]

            # Parse query.
            query_tree = QueryTree(qid, query)

            # Search and save results.
            search_results.add(query_tree.search(index))

    # Generate submission file.
    search_results.print_submission(args.objects_file, args.submission_file)

if __name__ == "__main__":
    main()
