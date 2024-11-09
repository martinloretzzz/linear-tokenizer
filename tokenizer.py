class TrieNode:
    def __init__(self):
        self.id = None
        self.children = {}

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, token, id):
        node = self.root
        for char in token:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.id = id

    def search(self, text, start_pos):
        match_id = None
        pos = start_pos
        token_len = 0
        node = self.root
        while pos < len(text):
            char = text[pos]
            if char not in node.children:
                break
            node = node.children[char]
            if node.id is not None:
                match_id = node.id
                token_len = (pos - start_pos) + 1
            pos += 1
        return match_id, token_len
    
    def encode(self, text):
        pos = 0
        ids = []
        while pos < len(text):
            id, token_length = self.search(text, pos)
            ids.append(id)
            pos += token_length
        return ids


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


class LinearTokenizer():
    def __init__(self, vocab):
        byte_encoder = bytes_to_unicode()
        byte_decoder = {v:k for k, v in byte_encoder.items()}

        self.vocab_encode = vocab
        self.vocab_decode = {}

        self.trie = Trie()
        for token, token_id in self.vocab_encode.items():
            token_bytes = bytes([byte_decoder[c] for c in token])
            self.vocab_decode[token_id] = token_bytes
            self.trie.insert(token_bytes, token_id)

    def encode(self, text, return_token_tuple=False):
        # TODO special handling of <|endoftext|> token
        ids = self.trie.encode(text.encode('utf-8'))
        return (ids, [self.vocab_decode[id] for id in ids]) if return_token_tuple else ids

    def decode(self, ids):
        out = bytes()
        for id in ids:
            if not id in self.vocab_decode:
                raise Exception(f"Error decoding {id}")
            out += self.vocab_decode[id]
        return out.decode('utf-8', errors="replace")