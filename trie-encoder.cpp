#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_map>
#include <optional>

namespace py = pybind11;

struct TrieNode {
    std::unordered_map<unsigned char, TrieNode*> children;
    std::optional<int> id;

    TrieNode() = default;
    ~TrieNode() {
        for (auto& child : children) {
            delete child.second;
        }
    }
};

class Trie {
public:
    TrieNode* root;

    Trie() {
        root = new TrieNode();
    }

    ~Trie() {
        delete root;
    }

    void insert(const std::vector<unsigned char>& token, int token_id) {
        TrieNode* node = root;
        for (unsigned char ch : token) {
            if (node->children.find(ch) == node->children.end()) {
                node->children[ch] = new TrieNode();
            }
            node = node->children[ch];
        }
        node->id = token_id;
    }

    std::pair<std::optional<int>, int> search(const std::vector<unsigned char>& text, int start_pos) {
        std::optional<int> match_id;
        int token_len = 0;
        int pos = start_pos;
        TrieNode* node = root;

        while (pos < text.size()) {
            unsigned char ch = text[pos];
            node = node->children[ch];
            if (node == nullptr) {
                break;
            }
            if (node->id) {
                match_id = node->id;
                token_len = (pos - start_pos) + 1;
            }
            pos++;
        }
        return {match_id, token_len};
    }

    std::vector<std::optional<int>> encode(const std::vector<unsigned char>& text) {
        std::vector<std::optional<int>> ids;
        int pos = 0;

        while (pos < text.size()) {
            auto [id, token_length] = search(text, pos);
            ids.push_back(id);
            pos += token_length;
        }
        return ids;
    }
};

PYBIND11_MODULE(trie, m) {
    py::class_<TrieNode>(m, "TrieNode");

    py::class_<Trie>(m, "Trie")
        .def(py::init<>())
        .def("insert", &Trie::insert, "Insert a token with a token_id",
             py::arg("token"), py::arg("token_id"))
        .def("search", &Trie::search, "Search for a token in the text starting from a specific position",
             py::arg("text"), py::arg("start_pos"))
        .def("encode", &Trie::encode, "Encode a text into a list of token IDs",
             py::arg("text"));
}
