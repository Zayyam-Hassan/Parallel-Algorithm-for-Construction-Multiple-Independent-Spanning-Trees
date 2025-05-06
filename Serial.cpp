#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <bitset>
#include <fstream>
#include <unordered_map>

using namespace std;
using namespace std::chrono;

using Permutation = vector<uint8_t>;

// Hash function for Permutation to use in unordered_map
struct PermHash {
    size_t operator()(const Permutation& perm) const {
        size_t h = 0;
        for (uint8_t val : perm) {
            h = h * 31 + val;
        }
        return h;
    }
};

Permutation getIdentity(uint8_t n) {
    Permutation v(n);
    iota(v.begin(), v.end(), 1);
    return v;
}

int findPosition(const Permutation& perm, uint8_t x) {
    return find(perm.begin(), perm.end(), x) - perm.begin();
}

Permutation Swap(const Permutation& perm, uint8_t x) {
    Permutation res = perm;
    int pos = findPosition(res, x);
    if (pos < res.size() - 1) {
        swap(res[pos], res[pos + 1]);
    }
    return res;
}

Permutation FindPosition(const Permutation& v, uint8_t t, uint8_t n) {
    const Permutation identity = getIdentity(n);
    if (t == 2 && Swap(v, t) == identity) {
        return Swap(v, t - 1);
    }
    if (v[n - 2] == t || v[n - 2] == (n - 1)) {
        for (int i = n - 1; i >= 0; --i) {
            if (v[i] != uint8_t(i + 1)) {
                uint8_t j = i + 1;
                return Swap(v, j);
            }
        }
    }
    return Swap(v, t);
}

Permutation getParent(const Permutation& v, uint8_t t, uint8_t n) {
    const Permutation identity = getIdentity(n);
    if (v == identity) return identity;

    if (v.back() == n) {
        if (t != n - 1) return FindPosition(v, t, n);
        else return Swap(v, v[n - 2]);
    } else if (v.back() == (n - 1) && v[n - 2] == n && Swap(v, n) != identity) {
        if (t == 1) return Swap(v, n);
        else return Swap(v, t - 1);
    } else {
        if (v.back() == t) return Swap(v, n);
        else return Swap(v, t);
    }
}

vector<vector<size_t>> generate(
    uint8_t n, 
    vector<Permutation>& perms, 
    vector<vector<vector<uint16_t>>>& children, 
    unordered_map<Permutation, size_t, PermHash>& permIndexMap
) {
    Permutation p = getIdentity(n);
    do {
        permIndexMap[p] = perms.size();
        perms.push_back(p);
    } while (next_permutation(p.begin(), p.end()));

    size_t total = perms.size();
    children.assign(n - 1, vector<vector<uint16_t>>(total));
    vector<vector<size_t>> parentIndices(total, vector<size_t>(n - 1));

    for (size_t i = 0; i < total; ++i) {
        for (uint8_t t = 1; t < n; ++t) {
            Permutation parent = getParent(perms[i], t, n);
            size_t parentIdx = permIndexMap[parent];
            parentIndices[i][t - 1] = parentIdx;
        }
    }

    for (size_t i = 0; i < total; ++i) {
        for (uint8_t t = 0; t < n - 1; ++t) {
            size_t parentIdx = parentIndices[i][t];
            if (parentIdx != i) {
                children[t][parentIdx].push_back(static_cast<uint16_t>(i));
            }
        }
    }

    return parentIndices;
}

void saveTreeStructureToFiles(
    uint8_t n, 
    const vector<Permutation>& perms, 
    const vector<vector<vector<uint16_t>>>& children, 
    const vector<vector<size_t>>& parentIndices) 
{
    for (uint8_t t = 1; t < n; ++t) {
        string filename = "tree_T_" + to_string(t) + ".txt";
        ofstream fout(filename);

        fout << "Tree T_" << static_cast<int>(t) << " Structure:\n";

        for (size_t i = 0; i < perms.size(); ++i) {
            const Permutation& current = perms[i];
            size_t parent_idx = parentIndices[i][t - 1];
            const Permutation& parent = perms[parent_idx];

            fout << "Permutation " << i << " [";
            for (auto num : current) fout << static_cast<int>(num) << " ";
            fout << "] -> Parent [";
            for (auto num : parent) fout << static_cast<int>(num) << " ";
            fout << "] (Index " << parent_idx << "), Children: ";

            const auto& child_indices = children[t - 1][i];
            if (child_indices.empty()) {
                fout << "None";
            } else {
                for (uint16_t child_idx : child_indices) {
                    fout << "[";
                    for (auto num : perms[child_idx]) fout << static_cast<int>(num) << " ";
                    fout << "] ";
                }
            }
            fout << "\n";
        }
        fout.close();
    }
}
void saveTreeStructureAsDOT(
    uint8_t n,
    const vector<Permutation>& perms,
    const vector<vector<vector<uint16_t>>>& children,
    const unordered_map<Permutation, size_t, PermHash>& permIndexMap)
{
    const Permutation identity = getIdentity(n);
    size_t rootIndex = permIndexMap.at(identity);  // Index of the root permutation

    for (uint8_t t = 1; t < n; ++t) {
        string filename = "tree_T_" + to_string(t) + ".dot";
        ofstream fout(filename);

        fout << "digraph Tree_T_" << static_cast<int>(t) << " {\n";
        fout << "  node [shape=box, fontname=\"Courier\"];\n";

        // Define nodes with labels
        for (size_t i = 0; i < perms.size(); ++i) {
            fout << "  n" << i << " [label=\"";
            for (uint8_t val : perms[i]) fout << static_cast<int>(val) << " ";
            fout << "\"];\n";
        }

        // Add edges (skip edges to the root node to ensure it has no parent)
        for (size_t parentIdx = 0; parentIdx < perms.size(); ++parentIdx) {
            for (uint16_t childIdx : children[t - 1][parentIdx]) {
                if (childIdx == rootIndex) continue;  // Skip edges pointing to root
                fout << "  n" << parentIdx << " -> n" << childIdx << ";\n";
            }
        }

        // Optional: force root to appear at the top
        fout << "  { rank=source; n" << rootIndex << " }\n";

        fout << "}\n";
        fout.close();
    }
}

int main() {
    int input_n = 4;
    cout << "Enter n (2-10): ";
    cin >> input_n;
    if (input_n < 2 || input_n > 10) {
        cerr << "Invalid n. Use 2 ≤ n ≤ 10.\n";
        return 1;
    }
    uint8_t n = static_cast<uint8_t>(input_n);

    vector<Permutation> perms;
    vector<vector<vector<uint16_t>>> children;
    unordered_map<Permutation, size_t, PermHash> permIndexMap;

    auto t0 = high_resolution_clock::now();
    vector<vector<size_t>> parentIndices = generate(n, perms, children, permIndexMap);
    auto t1 = high_resolution_clock::now();

    cout << "\nGenerated B_" << input_n << " in "
         << duration_cast<milliseconds>(t1 - t0).count() << " ms\n";
    cout << "Total permutations: " << perms.size() << "\n";

    for (uint8_t t = 1; t < n; ++t) {
        size_t edges = 0;
        for (const auto& v : children[t - 1]) edges += v.size();
        cout << "Tree T_" << static_cast<int>(t) << " edges: " << edges << "\n";
    }
    bool val = true ;
    if (val)
        saveTreeStructureAsDOT(n, perms, children, permIndexMap);
    return 0;
}
