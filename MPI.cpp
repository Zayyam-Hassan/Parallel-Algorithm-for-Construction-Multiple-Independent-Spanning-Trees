#include <sstream>
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <unordered_map>

using namespace std;

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

// Helper functions (unchanged)
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

void saveSingleTree(
    uint8_t t, 
    const vector<Permutation>& perms, 
    const vector<size_t>& parentIndices_t, 
    const vector<vector<size_t>>& children_t,
    int rank
) {
    string filename = "tree_T_" + to_string(t) + "_rank_" + to_string(rank) + ".dot";
    ofstream fout(filename);
    if (!fout.is_open()) {
        cerr << "Rank " << rank << " failed to open file " << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    // Start the DOT graph
    fout << "digraph Tree_T_" << static_cast<int>(t) << " {\n";
    fout << "    rankdir=BT;\n";
    fout << "    node [shape=box];\n\n";
    fout << "    // Nodes\n";

    // Write nodes directly in parallel with critical section
    #pragma omp parallel
    {
        stringstream local_ss;
        #pragma omp for schedule(static)
        for (size_t i = 0; i < perms.size(); ++i) {
            local_ss << "    node" << i << " [label=\"[";
            for (size_t j = 0; j < perms[i].size(); ++j) {
                local_ss << static_cast<int>(perms[i][j]);
                if (j < perms[i].size() - 1) local_ss << " ";
            }
            local_ss << "]\"];\n";
        }
        #pragma omp critical
        fout << local_ss.str();
    }
    fout << "\n";

    // Write edges directly
    fout << "    // Edges\n";
    #pragma omp parallel
    {
        stringstream local_ss;
        #pragma omp for schedule(static)
        for (size_t i = 0; i < perms.size(); ++i) {
            size_t parent_idx = parentIndices_t[i];
            if (i != parent_idx) {
                local_ss << "    node" << i << " -> node" << parent_idx << ";\n";
            }
        }
        #pragma omp critical
        fout << local_ss.str();
    }
    fout << "\n";

    // End the DOT graph
    fout << "}\n";
    fout.close();
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    uint8_t n;
    if (rank == 0) {
        int input_n;
        cout << "Enter n (2-10): ";
        cin >> input_n;
        if (input_n < 2 || input_n > 10) {
            cerr << "Invalid n. Use 2 \u2264 n \u2264 10.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        n = static_cast<uint8_t>(input_n);
    }
    MPI_Bcast(&n, 1, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    vector<Permutation> perms;
    if (rank == 0) {
        Permutation p = getIdentity(n);
        do {
            perms.push_back(p);
        } while (next_permutation(p.begin(), p.end()));
    }

    // Broadcast permutation count first
    size_t total;
    if (rank == 0) total = perms.size();
    MPI_Bcast(&total, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    // Broadcast flattened permutation list
    vector<uint8_t> flat_perms;
    if (rank == 0) {
        flat_perms.resize(total * n);
        // Parallelize copying permutations to flat_perms
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < total; ++i) {
            copy(perms[i].begin(), perms[i].end(), flat_perms.begin() + i * n);
        }
    } else {
        flat_perms.resize(total * n);
    }
    MPI_Bcast(flat_perms.data(), total * n, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        perms.resize(total, Permutation(n));
        // Parallelize copying flat_perms back to perms
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < total; ++i) {
            copy(flat_perms.begin() + i * n, flat_perms.begin() + (i + 1) * n, perms[i].begin());
        }
    }

    unordered_map<Permutation, size_t, PermHash> permIndexMap;
    // Parallelize permIndexMap creation with critical section
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < total; ++i) {
        #pragma omp critical
        {
            permIndexMap[perms[i]] = i;
        }
    }

    double start = MPI_Wtime();

    for (uint8_t t = 1; t < n; ++t) {
        if ((t - 1) % size == rank) {
            vector<size_t> parentIndices(total);
            // Parallelize parent index computation
            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < total; ++i) {
                Permutation parent = getParent(perms[i], t, n);
                size_t parentIdx = permIndexMap[parent];
                parentIndices[i] = parentIdx;
            }

            vector<vector<size_t>> children(total);
            // Parallelize children index computation
            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < total; ++i) {
                size_t parentIdx = parentIndices[i];
                if (parentIdx != i) {
                    #pragma omp critical
                    {
                        children[parentIdx].push_back(i);
                    }
                }
            }

            saveSingleTree(t, perms, parentIndices, children, rank);
        }
    }

    double end = MPI_Wtime();
    double elapsed = end - start;
    double max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Max elapsed time across ranks: " << max_elapsed << " seconds." << endl;
    }

    MPI_Finalize();
    return 0;
}