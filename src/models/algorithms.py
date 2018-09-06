from gensim.models.fasttext import MAX_WORDS_IN_BATCH
from sklearn.base import BaseEstimator
from gensim.models import FastText
import numpy as np

from sklearn.model_selection import GridSearchCV

_ALGORITHMS = {
}


def available_algorithms():
    """Valid Algorithms for training or prediction

    This function simply returns a dict of known
    algorithms strings and their corresponding estimator function.

    It exists to allow for a description of the mapping for
    each of the valid strings as a docstring

    The valid algorithm names, and the function they map to, are:

    ============     ====================================
    Algorithm        Function
    ============     ====================================

    ============     ====================================
    """
    return _ALGORITHMS


_META_ESTIMATORS = {
    'grid_search': GridSearchCV
}


def available_meta_estimators():
    """Valid Meta-estimators for training or prediction
    This function simply returns the list of known
    meta-estimators

    This function simply returns a dict of known
    algorithms strings and their corresponding estimator function.

    It exists to allow for a description of the mapping for
    each of the valid strings as a docstring

    The valid algorithm names, and the function they map to, are:

    ============     ====================================
    Meta-est         Function
    ============     ====================================
    grid_search      sklearn.model_selection.GridSearchCV
    ============     ====================================
    """
    return _META_ESTIMATORS


class FastTextEstimator(BaseEstimator):
    def __init__(self, sg=0, hs=0, size=100, alpha=0.025, window=5,
                 min_count=5, max_vocab_size=None, word_ngrams=1, sample=1e-3,
                 seed=1, workers=3, min_alpha=0.0001, negative=5,
                 ns_exponent=0.75, cbow_mean=1, hashfxn=hash, iter=5,
                 null_word=0, min_n=3, max_n=6, sorted_vocab=1, bucket=2000000,
                 trim_rule=None, batch_words=MAX_WORDS_IN_BATCH, callbacks=(),
                 restrict_to_corpus=True):
        """

        Parameters
        ----------
        sentences : iterable of list of str, optional
            Can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.
            If you don't supply `sentences`, the model is left uninitialized -- use if you plan to initialize it
            in some other way.
        min_count : int, optional
            The model ignores all words with total frequency lower than this.
        size : int, optional
            Dimensionality of the word vectors.
        window : int, optional
            The maximum distance between the current and predicted word within a sentence.
        workers : int, optional
            Use these many worker threads to train the model (=faster training with multicore machines).
        alpha : float, optional
            The initial learning rate.
        min_alpha : float, optional
            Learning rate will linearly drop to `min_alpha` as training progresses.
        sg : {1, 0}, optional
            Training algorithm: skip-gram if `sg=1`, otherwise CBOW.
        hs : {1,0}, optional
            If 1, hierarchical softmax will be used for model training.
            If set to 0, and `negative` is non-zero, negative sampling will be used.
        seed : int, optional
            Seed for the random number generator. Initial vectors for each word are seeded with a hash of
            the concatenation of word + `str(seed)`. Note that for a fully deterministically-reproducible run,
            you must also limit the model to a single worker thread (`workers=1`), to eliminate ordering jitter
            from OS thread scheduling. (In Python 3, reproducibility between interpreter launches also requires
            use of the `PYTHONHASHSEED` environment variable to control hash randomization).
        max_vocab_size : int, optional
            Limits the RAM during vocabulary building; if there are more unique
            words than this, then prune the infrequent ones. Every 10 million word types need about 1GB of RAM.
            Set to `None` for no limit.
        sample : float, optional
            The threshold for configuring which higher-frequency words are randomly downsampled,
            useful range is (0, 1e-5).
        negative : int, optional
            If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
            should be drawn (usually between 5-20).
            If set to 0, no negative sampling is used.
        ns_exponent : float, optional
            The exponent used to shape the negative sampling distribution. A value of 1.0 samples exactly in proportion
            to the frequencies, 0.0 samples all words equally, while a negative value samples low-frequency words more
            than high-frequency words. The popular default value of 0.75 was chosen by the original Word2Vec paper.
            More recently, in https://arxiv.org/abs/1804.04212, Caselles-Dupré, Lesaint, & Royo-Letelier suggest that
            other values may perform better for recommendation applications.
        cbow_mean : {1,0}, optional
            If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
        hashfxn : function, optional
            Hash function to use to randomly initialize weights, for increased training reproducibility.
        iter : int, optional
            Number of iterations (epochs) over the corpus.
        trim_rule : function, optional
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
            be trimmed away, or handled using the default (discard if word count < min_count).
            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
            or a callable that accepts parameters (word, count, min_count) and returns either
            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
            The rule, if given, is only used to prune vocabulary during
            :meth:`~gensim.models.fasttext.FastText.build_vocab` and is not stored as part of themodel.

            The input parameters are of the following types:
                * `word` (str) - the word we are examining
                * `count` (int) - the word's frequency count in the corpus
                * `min_count` (int) - the minimum count threshold.

        sorted_vocab : {1,0}, optional
            If 1, sort the vocabulary by descending frequency before assigning word indices.
        batch_words : int, optional
            Target size (in words) for batches of examples passed to worker threads (and
            thus cython routines).(Larger batches will be passed if individual
            texts are longer than 10000 words, but the standard cython code truncates to that maximum.)
        min_n : int, optional
            Minimum length of char n-grams to be used for training word representations.
        max_n : int, optional
            Max length of char ngrams to be used for training word representations. Set `max_n` to be
            lesser than `min_n` to avoid char ngrams being used.
        word_ngrams : {1,0}, optional
            If 1, uses enriches word vectors with subword(n-grams) information.
            If 0, this is equivalent to :class:`~gensim.models.word2vec.Word2Vec`.
        bucket : int, optional
            Character ngrams are hashed into a fixed number of buckets, in order to limit the
            memory usage of the model. This option specifies the number of buckets used by the model.
        callbacks : :obj: `list` of :obj: `~gensim.models.callbacks.CallbackAny2Vec`, optional
            List of callbacks that need to be executed/run at specific stages during training.
        restrict_to_corpus : boolean
            Necessarily restrict all lookups to the corpus that already exists within the fit model.
        """

        #self.sentences = sentences
        self.sg = sg
        self.hs = hs
        self.size = size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.word_ngrams = word_ngrams
        self.sample = sample
        self.seed = seed
        self.workers = workers
        self.min_alpha = min_alpha
        self.negative = negative
        self.ns_exponent = ns_exponent
        self.cbow_mean = cbow_mean
        self.hashfxn = hashfxn
        self.iter = iter
        self.null_word = null_word
        self.min_n = min_n
        self.max_n = max_n
        self.sorted_vocab = sorted_vocab
        self.bucket = bucket
        self.trim_rule = trim_rule
        self.batch_words = batch_words
        self.callbacks = callbacks
        self.restrict_to_corpus = restrict_to_corpus

    def fit(self, X):
        """
        Parameters
        ----------
        X : iterable of list of str, optional
            Can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from
            disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`,
                :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in
            :mod:`~gensim.models.word2vec` module for such examples.
        Here X is an iterable of sentences.
        """
        # FastText is currently broken for iterables.
        X = list(X)
        self.model_ = FastText(sentences=X, **self.get_params())

    def transform(self, X, y=None, restrict_to_corpus=None):
        """
        Parameters
        ----------
        X : iterable of list of str, optional
            Can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from
            disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`,
                :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in
            :mod:`~gensim.models.word2vec` module for such examples.
        Here X is an iterable of sentences.
        """
        # XXX: Fix this later to properly use iterables
        wl = [x.split() for x in X]
        if restrict_to_corpus is not None:
            rtc = restrict_to_corpus
        else:
            rtc = self.restrict_to_corpus
        if rtc:
            wordlist = [x for x in wl if x in self.model_.wv.vocab]
        else:
            wordlist = wordlist
        self.last_transformed_wordlist_ = wordlist
        return(np.array([self.model_.wv.word_vec(i) for i in wordlist]))

    def transform_words(self, X, y=None):
        """
        X: iterable over a set of words to lookup within our model
        """
        return(np.array([self.model_.wv.word_vec(i) for i in X]))

#     def partial_fit(self, X, epochs=1, len_X=None):
#         """
#         """

#         # Awww snap!  We have to pass the number of elements in our iterable.
#         if len_X is None:
#             if hasattr(X,'len'):
#                 len_X = len(X)
#             else:
#                 logger.warning('casting iterable to list.  Whahahaha!')
#                 X = list(X)
#                 len_X = len(X)

#         if not hasattr(self,'model_'):
#             self.iter = epochs
#             self.fit(X)
#         else:
#             self.model_.train(sentences= X,epochs=1, total_examples=len_X)
