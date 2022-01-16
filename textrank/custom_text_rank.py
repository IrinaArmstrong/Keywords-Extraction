# Basic
import itertools
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Any, Union, Tuple

import pymorphy2
import networkx as nx

import logging_handler
logger = logging_handler.get_logger(__name__)


class Candidate(object):
    """
    The keyphrase candidate data structure.
    """
    def __init__(self, token: str,
                 is_bigramm: bool,
                 pos: pymorphy2.tagset.OpencorporaTag,
                 lexical_forms: Dict[str, str] = None,
                 score: float = 0.0,
                 bigramm_sep: str = "_"):

        self._token = token
        self._is_bigramm = is_bigramm
        self._pos = pos
        self._score = score

        self._lexical_forms = lexical_forms
        self._bigramm_splitted = []

        if self._is_bigramm:
            self._bigramm_splitted = self._token.split(bigramm_sep)

    def __str__(self):
        cand_msg = f"Token: {self._token}"
        if self._is_bigramm:
            cand_msg += " is bigramm,"
        else:
            cand_msg += " is unigramm,"
        cand_msg += f"is {self._pos} POS,"
        cand_msg += f"selected with score = {self._score}."
        return cand_msg


class TextRanker:
    """
    TextRank for keyword extraction.
    This model builds a graph that represents the text. A graph based ranking
    algorithm is then applied to extract the lexical units (here the words) that
    are most important in the text.
    In this implementation,
     - nodes - are words of certain part-of-speech (nouns/adjectives/..)
     - edges - represent co-occurrence relation, controlled by the distance
               between word occurrences - a window of N words).
    """
    rus_lexical_forms = ['nomn', 'gent', 'datv', 'accs', 'ablt', 'loct', 'voct']

    def __init__(self):
        # From pymorphy2 avaliable POS tags
        # ref: http://opencorpora.org/dict.php?act=gram
        self.__pos_list = ['NOUN', 'ADJS', 'COMP', 'VERB', 'INFN',
                           'PRTF', 'PRTS', 'GRND', 'NUMR', 'Abbr']
        # All possible: ['NOUN', 'ADJS', 'ADJF', 'COMP', 'VERB', 'INFN',
        #               'PRTF', 'PRTS', 'GRND', 'NUMR', 'ADVB', 'Abbr']

        # Russian language parser
        self.__morph = pymorphy2.MorphAnalyzer()
        # Words graph
        self.__graph = nx.Graph()
        self.__texts = []  # as List[List[Dict[str, Any]]] -> [[{'words': [], 'bigramm': [], 'POS': []}]]
        # Each inner Dict == single token
        # Each inner List == single text
        # Outer List is composition of texts

        # Keyphrase candidates container
        self.__candidates = defaultdict(Candidate)

    def candidate_weighting(self, texts: List[List[str]],
                            window: int = 2, pos_list: List[str] = None,
                            include_bigramms: bool = True,
                            top_percent: float = None):
        """
        Tailored candidate ranking method for TextRank.
        Keyphrase candidates are either composed from the T-percent (top_percent)
        highest-ranked words or extracted using the `candidate_selection()` method.
        Candidates are ranked using the sum of their words.
        :param window - the window for connecting words in the graph.
        :param pos_list - the set of valid pos for words to be considered as nodes
                    in the graph, defaults to ('NOUN', 'PROPN', 'ADJ').
        :param top_percent - percentage of top vertices to keep for phrase generation.
        """
        if pos_list is not None:
            self.__pos_list = pos_list

        # flatten document as a sequence of (word, bigramm, pos) samples
        self.__texts = self.__tag_words(texts)
        self.__window = window
        self.__include_bigramms = include_bigramms
        self.__build_word_graph()

        # Computes the word scores using the unweighted PageRank formula
        # pagerank_scipy() is a SciPy sparse-matrix implementation of the power-method
        # Returns: pagerank – Dictionary of nodes with PageRank as value
        #         textranked = nx.pagerank_numpy(self.__graph, alpha=0.85, weight=None)
        textranked = nx.pagerank_scipy(self.__graph, alpha=0.85, tol=0.0001, weight=None)

        # Generate the phrases from the T-percent top ranked words
        if top_percent is not None:

            # warn user as this is not the pke way of doing it
            logger.info(f"Candidates are generated using {top_percent}%-top")

            # computing the number of top keywords
            n_nodes = self.__graph.number_of_nodes()
            to_keep = min(int(n_nodes * top_percent), n_nodes)

            # Sorting the nodes by decreasing scores
            top_words = {k: v for k, v in sorted(textranked.items(), key=lambda item: item[1], reverse=True)}
            top_words = {k: v for i, (k, v) in enumerate(top_words.items()) if i <= to_keep}

            # Create candidates
            self.__create_candidates(top_words)
        else:
            # warn user for non-use of reduction of number of keywords
            logger.info(f"Candidates are generated using all tokens")

            # computing the number of all keywords
            n_nodes = self.__graph.number_of_nodes()

            # Sorting the nodes by decreasing scores
            top_words = {k: v for k, v in sorted(textranked.items(), key=lambda item: item[1], reverse=True)}

            # Create candidates
            self.__create_candidates(top_words)

    def __check_validness(self, token_dict: Dict[str, Any]) -> bool:
        """
        Mark token `valid` if it belongs to one of selected POS or it is a bigramm.
        """
        return (token_dict['bigramm']) or (token_dict['pos'] is not None
                                           and any([tag in token_dict['pos'] for tag in self.__pos_list]))

    def __tag_words(self, texts: List[List[str]]):
        """
        Process given texts to selected form:
        [[{'words': [], 'bigramm': [], 'POS': [], 'valid'}]]
        """
        texts = [[{'token': token,
                   'bigramm': True if "_" in token else False,
                   'pos': self.__morph.parse(str(token).lower())[0].tag if "_" not in token else None}
                  for token in tokens]
                 for tokens in tqdm(texts)]
        _ = [[token.update({'valid': self.__check_validness(token)})
              for token in tokens] for tokens in tqdm(texts)]
        return texts

    def __build_word_graph(self):
        """
        Build a graph representation of the document in which nodes/vertices
        are words and edges represent co-occurrence relation. Syntactic filters
        can be applied to select only words of certain Part-of-Speech.
        Co-occurrence relations can be controlled using the distance between
        word occurrences in the document.
        """
        tokens = itertools.chain.from_iterable(self.__texts)
        # add nodes to the graph
        logger.info(f"Adding nodes to graph...")
        self.__graph.add_nodes_from([token['token'] for token in tokens if token['valid']])

        # add edges to the graph
        logger.info(f"Adding edges...")
        for text_i, tokens in enumerate(self.__texts):
            for token_i, token in enumerate(tokens):
                # speed up things
                if not token['valid']:
                    continue
                start_ind = min(token_i, (self.__window - 1) // 2)
                end_ind = min(token_i + self.__window, len(tokens))
                for j in range(start_ind, end_ind):
                    linked_token = tokens[j]
                    if linked_token['valid'] and linked_token['token'] != token['token']:
                        self.__graph.add_edge(token['token'], linked_token['token'])
        logger.info(f"Adding edges ended.")

    def __create_candidates(self, scored_tokens: Dict[str, float]):
        """
        Create keywords dict for all texts.
        """
        for token, score in tqdm(scored_tokens.items()):
            self.__add_candidate(token, score)
        logger.info(f"Keywords candidates created: {len(self.__candidates)}")

    def __add_candidate(self, token: str, score: float = None):
        """
        Add a keyphrase candidate to the candidates container.
        """
        is_bigramm = True if "_" in token else False
        lexical_forms = defaultdict(str)
        parsed_token = self.__morph.parse(str(token).lower())[0]
        pos = parsed_token.tag
        if ~is_bigramm and (('NOUN' in pos) or ('ADJF' in pos)
                            or ('NUMR' in pos) or ('PRTF' in pos)):
            for form in self.rus_lexical_forms:
                infl = parsed_token.inflect({form})
                if infl is not None:
                    lexical_forms[form] = parsed_token.inflect({form}).word
        pos = parsed_token.tag
        self.__candidates[token] = Candidate(token, is_bigramm, pos, lexical_forms, score)

    def get_candidates(self) -> Dict[str, Any]:
        return self.__candidates

    def select_keywords_from_text(self, text: List[str],
                                  return_scores: bool = False) -> Union[List[str], List[Tuple[str, float]]]:
        if return_scores:
            return [(token, candidate._score) for token, candidate in self.__candidates.items() if token in text]
        return [candidate for candidate in self.__candidates if candidate in text]

    def select_keywords_from_texts(self, texts: List[List[str]],
                                   return_scores: bool = False) -> List[Union[List[str], List[Tuple[str, float]]]]:
        return [self.select_keywords_from_text(text, return_scores=return_scores)
                for text in texts]