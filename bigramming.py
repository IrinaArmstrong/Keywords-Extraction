# Basic
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List

# W2V
from gensim.models.phrases import Phrases

import logging_handler
logger = logging_handler.get_logger(__name__)


class Bigrammer:

    def __init__(self, phrase_model=None):
        self.__phrase_model = phrase_model

    def train(self, texts: List[List[str]],
              min_count: int, threshold: float,
              to_save: bool, save_path: str =".", phrases_fn: str ="phrases.pkl"):
        """
        Train gensim Phrases model with NPMI scorer.
        :param texts - The training corpus must be a sequence of sentences,
                        with each sentence a list of tokens.
        :param min_count – Ignore all words and bigrams with total
                            collected count lower than this value.
        :param threshold – Represent a score threshold for forming
                            the phrases (higher means fewer phrases).
                            A phrase of words a followed by b is accepted if the score of
                            the phrase is greater than threshold.
                            For NPMI scorer is in the range -1 to 1.
        """
        logger.info("Training bigrammer started.")
        self.__phrase_model = Phrases(texts, min_count=min_count,
                                      threshold=threshold, scoring='npmi')
        logger.info("Training bigrammer finished.")
        if to_save:
            save_path = Path(save_path).resolve()
            if save_path.exists():
                save_path = save_path / phrases_fn
            else:
                logger.warning(f"Creating saving path at {str(save_path)}")
                save_path.mkdir(exist_ok=True)
                save_path = save_path / phrases_fn
            self.__phrase_model.save(str(save_path))
            if save_path.exists():
                logger.info(f"Bigrammer model successfully saved to: {str(save_path)}")
        return self

    @classmethod
    def load(cls, save_path: str, phrases_fn: str) -> object:
        """
        Load pre-trained model from file and init.
        """
        save_path = Path(save_path).resolve() / phrases_fn
        if not save_path.exists():
            logger.error(f"Bigrammer model was not found by path: {str(save_path)}")
            raise FileNotFoundError(f"Bigrammer model was not found by path: {str(save_path)}")
        else:
            logger.info(f"Bigrammer model loading from: {str(save_path)}")
        phrase_model = Phrases.load(str(save_path))
        logger.info(f"Bigrammer model successfully loaded.")
        return cls(phrase_model=phrase_model)

    def create_bigramms(self, texts: List[List[str]]) -> List[List[str]]:
        """
        Create bi-gramms from given text data, already splitted.
        """
        return [self.__phrase_model[text] if len(text) > 0
                else [] for text in tqdm(texts)]

    def process(self, data: pd.DataFrame,
                text_col: str, copy: bool = True) -> pd.DataFrame:
        """
        Create bi-gramms from given column in dataframe.
        """
        logger.info(f"Bigramms creation for texts in column {text_col} started")
        data_processed = self.create_bigramms(data[text_col].fillna("").to_list())
        if not copy:
            data[text_col] = data_processed
        else:
            data[text_col + "_bigramms"] = data_processed

        logger.info("Bigramms creation finished.")
        return data

    def get_vocab(self) -> Dict[bytes, int]:
        logger.info(f"Bigrammer vocab size: {len(self.__phrase_model.vocab)}")
        return self.__phrase_model.vocab

    def get_phraser(self) -> Phrases:
        return self.__phrase_model