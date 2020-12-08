
import numpy as np

bio_set = ["O", "B", "I"]

domain2entitylist = {
    "politics": ['country', 'politician', 'election', 'person', 'organisation', 'location', 'misc', 'politicalparty', 'event'],
    "science": ['country', 'scientist', 'person', 'organisation', 'location', 'misc', 'university', 'discipline', 'enzyme', 'protein', 'chemicalelement', 'chemicalcompound', 'astronomicalobject', 'academicjournal', 'event', 'theory', 'award'],
    "music": ['musicgenre', 'song', 'band', 'album', 'musicalartist', 'musicalinstrument', 'award', 'event', 'country', 'location', 'organisation', 'person', 'misc'],
    "literature": ['book', 'writer', 'award', 'poem', 'event', 'magazine', 'literarygenre', 'country', 'person', 'location', 'organisation', 'misc'],
    "ai": ['field', 'task', 'product', 'algorithm', 'researcher', 'metrics', 'programlang', 'conference', 'university', 'country', 'person', 'organisation', 'location', 'misc']
}

entity2desp = {"country": "country", "politician": "politician", "election": "election", "person": "person", "organisation": "organisation", "location": "location", "misc": "miscellaneous", "politicalparty": "political party", "event": "event", "scientist": "scientist", "university": "university", "discipline": "discipline", "enzyme": "enzyme", "protein": "protein", "chemicalelement": "chemical element", "chemicalcompound": "chemical compound", "astronomicalobject": "astronomical object", "academicjournal": "academic journal", "theory": "theory", "award": "award", "musicgenre": "music genre", "song": "song", "band": "band", "album": "album", "musicalartist": "musical artist", "musicalinstrument": "musical instrument", "book": "book", "writer": "writer", "award": "award", "poem": "poem", "event": "event", "magazine": "magazine", "literarygenre": "literary genre", "field": "field", "task": "task", "product": "product", "algorithm": "algorithm", "researcher": "researcher", "metrics": "metrics", "programlang": "programming language", "conference": "conference"}

def load_entity_embedding(tgt_dm, emb_dim, emb_file, usechar):
    emb_dim = emb_dim - 100 if usechar else emb_dim
    entity_list = domain2entitylist[tgt_dm]
    embedding = np.random.randn(len(entity_list), emb_dim)
    print("embedding: %d x %d" % (len(entity_list), emb_dim))

    # generate token list
    token_list = []
    for entity in entity_list:
        desp = entity2desp[entity]
        token_list.extend(desp.split())
    token_list = list(set(token_list))

    # load pretrained word embeddings
    assert emb_file is not None
    with open(emb_file, "r") as ef:
        print('Loading embedding file: %s' % emb_file)
        pre_trained = 0
        embedded_words = []
        token2embedding = {}
        for i, line in enumerate(ef):
            line = line.strip()
            sp = line.split()
            try:
                assert len(sp) == emb_dim + 1
            except:
                continue
            if sp[0] in token_list and sp[0] not in embedded_words:
                pre_trained += 1
                token2embedding[sp[0]] = [float(x) for x in sp[1:]]
                embedded_words.append(sp[0])

        print("Pre-train: %d / %d (%.2f)" % (pre_trained, len(token_list), pre_trained / len(token_list)))
    
    # generate entity embeddings
    for idx, entity in enumerate(entity_list):
        desp = entity2desp[entity]
        desp = desp.split()
        for tok in desp:
            emb = token2embedding[tok]
            embedding[idx] += emb

    if usechar:
        print("Loading character embeddings from torchtext.vocab.CharNGram ...")
        import torchtext
        char_ngram_model = torchtext.vocab.CharNGram()
        char_embedding = np.random.randn(len(entity_list), 100)
        for idx, entity in enumerate(entity_list):
            desp = entity2desp[entity]
            charemb = char_ngram_model[desp].squeeze().numpy()
            char_embedding[idx] = charemb
        
        embedding = np.concatenate((embedding, char_embedding), -1)
    
    return embedding
