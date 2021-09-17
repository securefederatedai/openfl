# Next Word Prediction Tutorial on Keras

### 1. Data
Different envoys could have different texts, there were used 3 books of fairy tales:
- Polish Fairy Tales by A. J. Gliński https://www.gutenberg.org/files/36668/36668-h/36668-h.htm
- English Fairy Tales by Joseph Jacobs https://www.gutenberg.org/cache/epub/7439/pg7439-images.html
- American Fairy Tales by L. FRANK BAUM https://www.gutenberg.org/files/4357/4357-h/4357-h.htm

### 2. Keras Model
At this point OpenFL maintains Sequential API and Functional API. Keras Submodels are not supported.
(Problem at serialization step, more specifically at InputLayer, that is required, but it creates
only if wrap model with functional API https://stackoverflow.com/questions/58153888/how-to-set-the-input-of-a-keras-subclass-model-in-tensorflow)

### 3. Vocabulary storing
As we use secure approach, we couldn't share data between envoys, so there is applied open-source 
word-vectors (code you could find at shard_descriptor.py).
There is you can find prepared vectors at keyed_vectors.pkl