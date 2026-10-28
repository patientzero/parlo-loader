"""PARLO corpus exploration without implicit audio decoding or ML dependencies."""
from .audio import Audio, load_audio
from .chat import clean_text, normalize_text
from .corpus import Corpus, CorpusView, iter_batches, iter_rows
from .evaluation import SplitManifest, stratified_split, stratified_kfold, leave_one_out

__version__ = '0.1.0'
__all__ = ['Corpus', 'CorpusView', 'Audio', 'load_audio', 'clean_text', 'normalize_text',
           'SplitManifest', 'stratified_split', 'stratified_kfold', 'leave_one_out', 'iter_batches', 'iter_rows']
