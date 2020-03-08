FILE FORMAT DETAILS

The files train.txt, valid.txt, and test.text contain the training, development, and test set knowledge base triples used in both [1] and [2].
The file text_cvsc.txt contains the textual triples used in [2] and the file text_emnlp.txt contains the textual triples used in [1].

The knowledge base triples contain lines like this:

/m/0grwj	/people/person/profession	/m/05sxg2

The separator is a tab character; the mids are Freebase ids of entities, and the relation is a single or a two-hop relation from Freebase, where an intermediate complex value type entity has been collapsed out.

REFERENCES

[1] Kristina Toutanova, Danqi Chen, Patrick Pantel, Hoifung Poon, Pallavi Choudhury, and Michael Gamon. Representing text for joint embedding of text and knowledge bases.  In Proceedings of EMNLP 2015.

[2] Kristina Toutanova and Danqi Chen. Observed versus latent features for knowledge base and text inference. In Proceedings of the 3rd Workshop on Continuous Vector Space Models and Their Compositionality 2015.

[3] Antoine Bordes, Nicolas Usunier, Alberto Garcia Duran, Jason Weston, and Oksana Yakhnenko.  Translating embeddings for modeling multirelational data. In Advances in Neural Information Processing Systems (NIPS) 2013.

[4] Evgeniy Gabrilovich, Michael Ringgaard, and Amarnag Subramanya. FACC1: Freebase annotation of ClueWeb corpora, Version 1 (release date 2013-06-26, format version 1, correction level 0). http://lemurproject.org/clueweb12/FACC1/

[5] http://lemurproject.org/clueweb12/
