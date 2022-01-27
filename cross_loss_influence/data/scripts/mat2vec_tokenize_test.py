import pandas as pd
import regex
from cross_loss_influence.config import PROJECT_HOME, DATA_DIR
import logging
import string
import unidecode
from chemdataextractor.doc import Paragraph
from gensim.models.phrases import Phraser
import copy
import os

logger = logging.getLogger(__name__)
"""
Modified version of the MaterialsTextProcessor, publicly available in the mat2vec github repo:
https://github.com/materialsintelligence/mat2vec
under the MIT License
Originally authored by Vahe Tshitoyan and credited to John Dagdelen, Leigh Weston, Anubhav Jain
Revisions and updates by Andrew Silva
"""


class MedicalTextProcessor:

    def __init__(self,
                 phraser_path=os.path.join(PROJECT_HOME, 'models', 'checkpoints', 'phraser.pkl'),
                 BERT=False):
        self.punctuation = list(string.punctuation) + ["\"", "“", "”", "≥", "≤", "×"]
        self.phraser = Phraser.load(phraser_path)
        self.BERT = BERT
        if self.BERT:
            self.toker = BertTokenizer(vocab_file=os.path.join(PROJECT_HOME, 'data','objects','itos.txt'))
        self.NR_BASIC = regex.compile(r"^[+-]?\d*\.?\d+\(?\d*\)?+$", regex.DOTALL)
        self.lookup_dict = self.load_replacements()
        self.stop_words = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once',
                           'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do',
                           'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am',
                           'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we',
                           'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself',
                           'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours',
                           'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been',
                           'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over',
                           'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just',
                           'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't',
                           'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further',
                           'was', 'here', 'than', 'î', 'ó', 'ô', 'đ', 'š', 'ƒ', 'ƞ', 'ɤ', 'ώ', 'в', 'и', 'с', '⁶', '₁',
                           '₂', 'ⅴ', 'ⅻ', '↕', '≡', '⋙', '△', '☆', '✕', '－', '＞', '}(n)', '}118', '}56', '~10',
                           '~106', '~116', '~145,000', '~195', '~210', '~28', '~29', '~30', '~300', '~31', '~32', '~49',
                           '~505', '~80', '~90', 'è', 'ê', 'ϕ', '℃', 'ⅸ']

    @staticmethod
    def replace_from_lookup(lowercase_tokens, replacement_dict):
        """
        swap out synonyms (set multiple_sclerosis = ms = multiple sclerosis)
        :param lowercase_tokens: raw altered strings
        :param replacement_dict: dict of form {token to keep: [list of synonyms to replace], token2_to_keep: [list2], etc}
        :return: text input
        """
        for key, val in replacement_dict.items():
            for token in val:
                for _ in range(3):
                    lowercase_tokens = lowercase_tokens.replace(token, key)
        return lowercase_tokens

    @staticmethod
    def load_replacements():
        """
        create a dictionary of {token_to_keep: [list of replacements]} with hard coded lookups
        :return: dictionary
        """
        disease_analogy_gt = pd.read_csv(
            os.path.join(DATA_DIR, 'disease_similarities.csv'))
        molecular_analogy_gt = pd.read_csv(
            os.path.join(DATA_DIR, 'molecular_similarities.csv'))
        master_dict = {}
        replacement_number_df = pd.DataFrame(columns=molecular_analogy_gt.columns)
        for df in [disease_analogy_gt, molecular_analogy_gt, replacement_number_df]:
            for _, row in df.iterrows():
                row = row.dropna().tolist()
                if ' ' in row[0]:
                    row.append(row[0])
                    row[0] = row[0].replace(' ', '_')
                skip_this = False
                for entry in row:
                    if 'X is a number between' in entry:  # If it's one of these, add all the new rows and skip this
                        skip_this = True
                        start = 1
                        end = int(entry.split()[-1])
                        for subscript in range(start, end+1):
                            new_row = copy.deepcopy(row)

                            new_row = new_row[:-1]  # drop the x between nonsense
                            if row[0] == 'CDX' and subscript in [31, 40]:  # weird hard-coded rule to avoid cluster of differentiation 31 and 40
                                continue

                            new_data = {"Molecule": new_row[0].replace('X', str(subscript)), "A1":None, "A2": None, "A3":None,
                                        "A4":None, "A5": None, "A6":None, "A7":None, "A8": None, "A9":None,
                                        "A10":None, "A11": None, "A12":None, "A13":None, "A14": None}
                            for entry_index in range(1, len(new_row)):
                                new_data[f"A{entry_index}"] = new_row[entry_index].replace('X', str(subscript))
                            replacement_number_df.loc[len(replacement_number_df)] = new_data
                if skip_this:
                    continue
                row = [' ' + r.lower() + ' ' for r in row]
                list_of_replacements = row[1:]
                list_of_replacements.sort(key=len)
                list_of_replacements = list_of_replacements[::-1]
                list_of_replacements.append(" " + row[0].strip() + " " + row[0].strip() + " ")
                master_dict[row[0]] = list_of_replacements
        return master_dict

    def token_acceptable(self, text):
        """
        clean tokens of stopwords or urls or whatever else
        :param text: token to clean
        :return: 1 if token is valid or -1 if token is invalid
        """
        stripped = text.strip().lower()
        if not stripped:
            return False
        if 'www.' in stripped:
            return False
        if stripped in self.stop_words:
            return False
        return True

    def tokenize(self, text, keep_sentences=True):
        """Converts a string to a list tokens (words) using a modified chemdataextractor tokenizer.

        Args:
            text: input text as a string

            keep_sentences: if False, will disregard the sentence structure and return tokens as a
                single list of strings. Otherwise returns a list of lists, each sentence separately.

        Returns:
            A list of strings if keep_sentence is False, otherwise a list of list of strings, which each
            list corresponding to a single sentence.
        """
        if not self.BERT:
            cde_p = Paragraph(text)
            tokens = cde_p.tokens
        else:
            tokens = self.toker._tokenize(text)
        toks = []
        for sentence in tokens:
            if keep_sentences:
                toks.append([])
                for tok in sentence:
                    if not self.BERT:
                        tok = tok.text
                    if self.token_acceptable(tok):
                        toks[-1] += [tok]
            else:
                for tok in sentence:
                    if not self.BERT:
                        tok = tok.text
                    if self.token_acceptable(tok):
                        toks += [tok]
        return toks

    def process_for_vocab(self, tokens, exclude_punct=False, convert_num=False, remove_accents=True, make_phrases=False):
        """Processes a pre-tokenized list of strings or a string.

        Selective lower casing, accent removal, phrasing

        Args:
            tokens: A list of strings or a string. If a string is supplied, will use the
                tokenize method first to split it into a list of token strings.
            exclude_punct: Bool flag to exclude all punctuation.
            convert_num: Bool flag to convert numbers (selectively) to <nUm>.
            remove_accents: Bool flag to remove accents, e.g. Néel -> Neel.
            make_phrases: Bool flag to convert single tokens to common materials science phrases.

        Returns:
            A list of strings
        """
        if not isinstance(tokens, str):
            tokens = " ".join(tok for tok in tokens)

        tokens = tokens.lower()
        tokens = self.replace_from_lookup(tokens, self.lookup_dict)

        if not isinstance(tokens, list):  # If it's a string.
            tokens = self.tokenize(tokens, keep_sentences=False)

        processed = []

        for i, tok in enumerate(tokens):
            if exclude_punct and tok in self.punctuation:  # Punctuation.
                continue
            elif convert_num and self.is_number(tok):  # Number.
                # Replace all numbers with <NUM>, except if it is a crystal direction (e.g. "(111)").
                try:
                    if tokens[i - 1] == "(" and tokens[i + 1] == ")" \
                            or tokens[i - 1] == "〈" and tokens[i + 1] == "〉":
                        pass
                    else:
                        tok = "<NUM>"
                except IndexError:
                    tok = "<NUM>"

            if remove_accents:
                tok = self.remove_accent(tok)

            processed.append(tok)

        if make_phrases:
            processed = self.make_phrases(processed, reps=2)
        while '.' in processed:
            eos_ind = processed.index('.')
            processed[eos_ind:eos_ind+1] = ['<EOS>', '<SOS>']
        processed[:0] = ['<SOS>']
        if processed[-1] == '<SOS>':
            processed = processed[:-1]
        return processed

    def process_for_model(self, tokens, exclude_punct=False, convert_num=False, remove_accents=True, make_phrases=False):
        """Processes a pre-tokenized list of strings or a string.

        Selective lower casing, accent removal, phrasing

        Args:
            tokens: A list of strings or a string. If a string is supplied, will use the
                tokenize method first to split it into a list of token strings.
            exclude_punct: Bool flag to exclude all punctuation.
            convert_num: Bool flag to convert numbers (selectively) to <nUm>.
            remove_accents: Bool flag to remove accents, e.g. Néel -> Neel.
            make_phrases: Bool flag to convert single tokens to common materials science phrases.

        Returns:
            A list of strings
        """
        if not isinstance(tokens, str):
            tokens = " ".join(tok for tok in tokens)

        tokens = tokens.lower()
        tokens = self.replace_from_lookup(tokens, self.lookup_dict)

        if not isinstance(tokens, list):  # If it's a string.
            tokens = self.tokenize(tokens, keep_sentences=False)

        processed = []
        punct_checklist = copy.deepcopy(self.punctuation)
        punct_checklist.remove('.')
        for i, tok in enumerate(tokens):
            if exclude_punct and tok in punct_checklist:  # Punctuation.
                continue
            elif convert_num and self.is_number(tok):  # Number.
                # Replace all numbers with <NUM>, except if it is a crystal direction (e.g. "(111)").
                try:
                    if tokens[i - 1] == "(" and tokens[i + 1] == ")" \
                            or tokens[i - 1] == "〈" and tokens[i + 1] == "〉":
                        pass
                    else:
                        tok = "<NUM>"
                except IndexError:
                    tok = "<NUM>"

            if remove_accents:
                tok = self.remove_accent(tok)

            processed.append(tok)

        if make_phrases:
            processed = self.make_phrases(processed, reps=2)
        while '.' in processed:
            eos_ind = processed.index('.')
            processed[eos_ind:eos_ind+1] = ['<EOS>', '<SOS>']
        processed[:0] = ['<SOS>']
        if processed[-1] == '<SOS>':
            processed = processed[:-1]
        return processed

    def make_phrases(self, sentence, reps=2):
        """Generates phrases from a sentence of words.

        Args:
            sentence: A list of tokens (strings).
            reps: How many times to combine the words.

        Returns:
            A list of strings where the strings in the original list are combined
            to form phrases, separated from each other with an underscore "_".
        """
        while reps > 0:
            sentence = self.phraser[sentence]
            reps -= 1
        return sentence

    def is_number(self, s):
        """Determines if the supplied string is number.

        Args:
            s: The input string.

        Returns:
            True if the supplied string is a number (both . and , are acceptable), False otherwise.
        """
        return self.NR_BASIC.match(s.replace(",", "")) is not None

    @staticmethod
    def remove_accent(txt):
        """Removes accents from a string.

        Args:
            txt: The input string.

        Returns:
            The de-accented string.
        """
        # There is a problem with angstrom sometimes, so ignoring length 1 strings.
        return unidecode.unidecode(txt) if len(txt) > 1 else txt


class SciFiTextProcessor:

    def __init__(self,
                 phraser_path=os.path.join(PROJECT_HOME, 'models', 'checkpoints', 'phraser.pkl'),
                 BERT=False):
        self.punctuation = list(string.punctuation) + ["\"", "“", "”", "≥", "≤", "×"]
        self.BERT = BERT
        self.NR_BASIC = regex.compile(r"^[+-]?\d*\.?\d+\(?\d*\)?+$", regex.DOTALL)
        # self.stop_words = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once',
        #                    'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do',
        #                    'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am',
        #                    'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we',
        #                    'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself',
        #                    'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours',
        #                    'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been',
        #                    'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over',
        #                    'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just',
        #                    'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't',
        #                    'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further',
        #                    'was', 'here', 'than', 'î', 'ó', 'ô', 'đ', 'š', 'ƒ', 'ƞ', 'ɤ', 'ώ', 'в', 'и', 'с', '⁶', '₁',
        #                    '₂', 'ⅴ', 'ⅻ', '↕', '≡', '⋙', '△', '☆', '✕', '－', '＞', '}(n)', '}118', '}56', '~10',
        #                    '~106', '~116', '~145,000', '~195', '~210', '~28', '~29', '~30', '~300', '~31', '~32', '~49',
        #                    '~505', '~80', '~90', 'è', 'ê', 'ϕ', '℃', 'ⅸ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
        #                    'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        self.stop_words = []

    def token_acceptable(self, text):
        """
        clean tokens of stopwords or urls or whatever else
        :param text: token to clean
        :return: 1 if token is valid or -1 if token is invalid
        """
        stripped = text.strip().lower()
        if not stripped:
            return False
        if 'www.' in stripped:
            return False
        if stripped in self.stop_words:
            return False
        return True

    def tokenize(self, text, keep_sentences=True):
        """Converts a string to a list tokens (words) using a modified chemdataextractor tokenizer.
        Args:
            text: input text as a string

            keep_sentences: if False, will disregard the sentence structure and return tokens as a
                single list of strings. Otherwise returns a list of lists, each sentence separately.

        Returns:
            A list of strings if keep_sentence is False, otherwise a list of list of strings, which each
            list corresponding to a single sentence.
        """
        tokens = text.split(' ')
        toks = []
        for word in tokens:
            if keep_sentences:
                toks.append([])
                if self.token_acceptable(word):
                    toks[-1] += [word]
            else:
                if self.token_acceptable(word):
                    toks += [word]
        return toks

    def process_for_vocab(self, tokens, exclude_punct=False, convert_num=True, remove_accents=True, make_phrases=False):
        """Processes a pre-tokenized list of strings or a string.

        Selective lower casing, accent removal, phrasing

        Args:
            tokens: A list of strings or a string. If a string is supplied, will use the
                tokenize method first to split it into a list of token strings.
            exclude_punct: Bool flag to exclude all punctuation.
            convert_num: Bool flag to convert numbers (selectively) to <nUm>.
            remove_accents: Bool flag to remove accents, e.g. Néel -> Neel.
            make_phrases: Bool flag to convert single tokens to common materials science phrases.

        Returns:
            A list of strings
        """
        if not isinstance(tokens, str):
            tokens = " ".join(tok for tok in tokens)

        tokens = tokens.lower()

        if not isinstance(tokens, list):  # If it's a string.
            tokens = self.tokenize(tokens, keep_sentences=False)

        processed = []

        for i, tok in enumerate(tokens):
            if exclude_punct and tok in self.punctuation:  # Punctuation.
                continue
            elif convert_num and self.is_number(tok):  # Number.
                # Replace all numbers with <NUM>, except if it is a crystal direction (e.g. "(111)").
                try:
                    if tokens[i - 1] == "(" and tokens[i + 1] == ")" \
                            or tokens[i - 1] == "〈" and tokens[i + 1] == "〉":
                        pass
                    else:
                        tok = "<NUM>"
                except IndexError:
                    tok = "<NUM>"
            if remove_accents:
                tok = self.remove_accent(tok)

            processed.append(tok)
        while '.' in processed:
            eos_ind = processed.index('.')
            processed[eos_ind:eos_ind+1] = ['<EOS>', '<SOS>']
        processed[:0] = ['<SOS>']
        if processed[-1] == '<SOS>':
            processed = processed[:-1]
        return processed

    def process_for_model(self, tokens, exclude_punct=False, convert_num=True, remove_accents=True, make_phrases=False):
        """Processes a pre-tokenized list of strings or a string.

        Selective lower casing, accent removal, phrasing

        Args:
            tokens: A list of strings or a string. If a string is supplied, will use the
                tokenize method first to split it into a list of token strings.
            exclude_punct: Bool flag to exclude all punctuation.
            convert_num: Bool flag to convert numbers (selectively) to <nUm>.
            remove_accents: Bool flag to remove accents, e.g. Néel -> Neel.
            make_phrases: Bool flag to convert single tokens to common materials science phrases.

        Returns:
            A list of strings
        """
        if not isinstance(tokens, str):
            tokens = " ".join(tok for tok in tokens)

        tokens = tokens.lower()

        if not isinstance(tokens, list):  # If it's a string.
            tokens = self.tokenize(tokens, keep_sentences=False)

        processed = []
        punct_checklist = copy.deepcopy(self.punctuation)
        punct_checklist.remove('.')
        for i, tok in enumerate(tokens):
            if exclude_punct and tok in punct_checklist:  # Punctuation.
                continue
            elif convert_num and self.is_number(tok):  # Number.
                # Replace all numbers with <NUM>, except if it is a crystal direction (e.g. "(111)").
                try:
                    if tokens[i - 1] == "(" and tokens[i + 1] == ")" \
                            or tokens[i - 1] == "〈" and tokens[i + 1] == "〉":
                        pass
                    else:
                        tok = "<NUM>"
                except IndexError:
                    tok = "<NUM>"
            if remove_accents:
                tok = self.remove_accent(tok)
            processed.append(tok)

        while '.' in processed:
            eos_ind = processed.index('.')
            processed[eos_ind:eos_ind+1] = ['<EOS>', '<SOS>']
        processed[:0] = ['<SOS>']
        if processed[-1] == '<SOS>':
            processed = processed[:-1]
        return processed

    @staticmethod
    def remove_accent(txt):
        """Removes accents from a string.

        Args:
            txt: The input string.

        Returns:
            The de-accented string.
        """
        # There is a problem with angstrom sometimes, so ignoring length 1 strings.
        return unidecode.unidecode(txt) if len(txt) > 1 else txt

    def is_number(self, s):
        """Determines if the supplied string is number.

        Args:
            s: The input string.

        Returns:
            True if the supplied string is a number (both . and , are acceptable), False otherwise.
        """
        return self.NR_BASIC.match(s.replace(",", "")) is not None



def preprocess_data_for_vocab(corpus_in, processor, lower=True):
    """
    Pass in a set of abstracts from a pickle file, and this will preprocess them to remove numbers, punctuation, casing
    and replace synonyms
    :param corpus_in: loaded pickle file
    :param lower: whether or not to lowercase all tokens
    :return:
    """
    num_errors = 0
    all_tokens = []
    for index, abstract in enumerate(corpus_in):
        try:
            tokens = processor.process_for_vocab(abstract, exclude_punct=True)
        except TypeError as e:
            print(e)
            num_errors += 1
            continue
        except Exception as e:
            print(e)
            print(f"Failed on abstract with index {index} and text: {abstract}")
            print(f"Dying...")
            break
        all_tokens.append(tokens)
    logger.info(f"Completed preprocessing with {num_errors} errors")
    return all_tokens


def preprocess_data_for_model(corpus_in, lower=True):
    """
    Pass in a set of abstracts from a pickle file, and this will preprocess them to remove numbers, punctuation, casing
    and replace synonyms
    :param corpus_in: loaded pickle file
    :param lower: whether or not to lowercase all tokens
    :return:
    """
    num_errors = 0
    text_processor = MedicalTextProcessor(BERT=False)
    all_tokens = []
    for index, abstract in enumerate(corpus_in):
        try:
            tokens = text_processor.process_for_model(abstract, exclude_punct=True)
        except TypeError as e:
            print(e)
            num_errors += 1
            continue
        except Exception as e:
            print(e)
            print(f"Failed on abstract with index {index} and text: {abstract}")
            print(f"Dying...")
            break
        all_tokens.append(tokens)
    logger.info(f"Completed preprocessing with {num_errors} errors")
    return all_tokens


def preprocess_single_abstract_for_model(abstract_in, lower=True):
    """
    Pass in a set of abstracts from a pickle file, and this will preprocess them to remove numbers, punctuation, casing
    and replace synonyms
    :param abstract_in: loaded abstract file
    :param lower: whether or not to lowercase all tokens
    :return:
    """
    text_processor = MedicalTextProcessor(BERT=False)
    try:
        tokens = text_processor.process_for_model(abstract_in, exclude_punct=True)
    except TypeError as e:
        print(e)
        return False
    except Exception as e:
        print(e)
        print(f"Dying...")
        return False
    return tokens
