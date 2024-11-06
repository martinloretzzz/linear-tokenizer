import regex as re


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
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}

        self.vocab_encode = vocab
        self.vocab_decode = {v:k for k,v in vocab.items()}

        self.trie = Trie()
        for token, token_id in self.vocab_encode.items():
            self.trie.insert(token, token_id)

        # https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
        self.pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s""")


    def encode(self, text, return_token_tuple=False):
        # TODO special handling of <|endoftext|> token
        pretokens = self.pattern.findall(text)
        pretokens = [''.join(self.byte_encoder[b] for b in pretoken.encode('utf-8')) for pretoken in pretokens]
        ids = []
        for pretoken in pretokens:
            if pretoken in self.vocab_encode:
                ids.append(self.vocab_encode[pretoken])
            else:
                ids.extend(self.trie.encode(pretoken))

        if return_token_tuple:
            return (ids, [self.vocab_decode[id] for id in ids])
        return ids

    def decode(self, ids):
        out = ""
        for id in ids:
            if not id in self.vocab_decode:
                raise Exception(f"Error decoding {id}")
            out += self.vocab_decode[id]
        return bytearray([self.byte_decoder[c] for c in out]).decode('utf-8', errors="replace")
