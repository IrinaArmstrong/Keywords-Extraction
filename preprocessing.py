# Basic
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Optional, List

# W2V
import spacy
import pymorphy2
import multiprocessing

import logging_handler
logger = logging_handler.get_logger(__name__)


def file_opener(filename: str) -> List[str]:
    with open(filename, 'rt', encoding='utf-8-sig') as src:
        file = src.read()
    return [x.strip().lower() for x in file.split('\n') if x]


class DataPreprocessorLemmatizer:
    text_features = ['text']
    stopgrams = [
        'CONJ',  # союз
        'PRCL',  # частица
        'PRED',  # предикатив
        'NPRO',  # местоимение-сущ.
        'INTJ',  # междометие
        'Erro',  # ошибка
        'Dist',  # искажение
        'Ques',  # вопросительное слово
        'Dmns',  # указательное слово
        'Prnt'  # вводное слово
    ]

    def __init__(self, multipocess: bool, num_processors: int = 16, chunksize: int = 100,
                 stopwords_path: str = './lists'):
        # Language parsers
        self.__morph = pymorphy2.MorphAnalyzer()
        self.__nlp = spacy.blank('ru')
        # Multiprocessing params
        self.__multipocess = multipocess
        self.__num_processors = num_processors
        self.__chunksize = chunksize
        # Cleaning utils
        stopwords_path = Path(stopwords_path).resolve()
        if not stopwords_path.exists():
            logger.warning(f"Stopwords folder was not found, no stopwords will be used!")
            self.__stopwords = []
        if (stopwords_path / 'NLTK_stopwords.txt').exists():
            logger.info(f"NLTK stopwords loading..")
            self.__stopwords = file_opener(str((stopwords_path / 'NLTK_stopwords.txt')))
            logger.info(f"{len(self.__stopwords)} loaded.")

    def get_stopwords(self):
        """
        Check intro-words list.
        """
        logger.info(f"Return {len(self.__stopwords)} stopwords")
        return self.__stopwords

    def get_analyzer(self):
        """
        Allow to access to Pymorphy Analyzer instance.
        """
        return self.__morph

    def _process_text(self, text: str):
        """
        Process single text and return list of tokens.
        """
        if pd.isna(text):
            return []
        # Pre-processing part
        text = [str(token).lower()
                for token in self.__nlp.make_doc(text)
                if (token and token.is_alpha and len(str(token.text)) > 2 and ~token.is_stop)]
        # Processing part
        clean_text = []
        for token in text:
            token = self.__morph.parse(str(token).lower())[0]
            if ((token.normal_form not in self.__stopwords)
                    and all([tag not in token.tag for tag in self.stopgrams])):
                clean_text.append(token.normal_form)
        return clean_text

    def process_texts(self, texts: List[str]):
        """
        Process list of texts and return list of lists of tokens.
        """
        if self.__multipocess:
            with multiprocessing.Pool(self.__num_processors) as pool:
                processed_texts = list(tqdm(pool.imap(self._process_text, texts,
                                                      chunksize=self.__chunksize),
                                            total=len(texts)))
            return processed_texts
        else:
            return [self._process_text(text) for text in tqdm(texts)]

    def process(self, data: pd.DataFrame,
                features_cols: Optional[List[str]] = None, copy: bool = True) -> pd.DataFrame:
        """
        Preprocess text for language modelling.
         - clean introduction words, numbers and small prefixes;
         - tokenize and lemmatize texts;
        """
        logger.info("Text processing started.")
        if not features_cols:
            features_cols = self.text_features

        for col_name in features_cols:
            logger.info(f"Processing '{col_name}' column...")
            data_processed = self.process_texts(data[col_name].fillna("").to_list())
            if not copy:
                data[col_name] = data_processed
            else:
                data[col_name + "_proc"] = data_processed

        logger.info("Text preprocessing finished.")
        return data