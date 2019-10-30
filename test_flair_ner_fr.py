from flair.data import Corpus
from flair.datasets import WIKINER_FRENCH
from flair.embeddings import WordEmbeddings, BytePairEmbeddings, StackedEmbeddings

# 1. get the corpus
corpus: Corpus = WIKINER_FRENCH()

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# initialize embeddings
embeddings = StackedEmbeddings(
    [
        # standard FastText word embeddings for French
        WordEmbeddings('fr'),
        # Byte pair embeddings for French
        BytePairEmbeddings('fr'),
    ]
)

# initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type)

# initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train('resources/taggers/fr-ner-0917',
              train_with_dev=True,  
              max_epochs=150, embeddings_storage_mode='gpu')
