import java.util.*;

/**
 * Random Projection LSH (SimHash / Charikar)
 *
 * Problem: approximate nearest neighbor search over N embeddings with
 * sub-linear candidate generation, followed by exact rerank.
 *
 * Theory: for unit vectors u, v with angle theta, a random hyperplane has
 * P(sign(r.u) != sign(r.v)) = theta / pi. With b independent hyperplanes per
 * table and L tables, recall improves while candidate set stays small.
 *
 * Complexity: O(N * L * b * d) build; O(L * b * d + |C| * d) per query.
 * Baseline: brute-force cosine scan (exact, O(N * d)).
 */
public class RandomProjectionLSH {

    static final Random RNG = new Random(11);
    static final int DIM = 64;

    public static void main(String[] args) {
        int n = 5000, queries = 200, k = 10;
        int bits = 8, tables = 6;

        double[][] data = clustered(n, DIM, 50, 0.55);
        double[][] queryVecs = clustered(queries, DIM, 50, 0.55);

        LSHIndex lsh = new LSHIndex(bits, tables, DIM);
        for (int i = 0; i < n; i++) lsh.add(i, data[i]);

        long hits = 0, totalTruth = 0, totalCandidates = 0;
        for (double[] q : queryVecs) {
            int[] truth = bruteForce(data, q, k);
            int[] approx = lsh.query(q, k, data);
            Set<Integer> truthSet = new HashSet<>();
            for (int t : truth) truthSet.add(t);
            for (int a : approx) if (truthSet.contains(a)) hits++;
            totalTruth += truth.length;
            totalCandidates += approx.length;
        }

        System.out.println("=== Random Projection LSH ===");
        System.out.printf("Recall@%d              = %.3f%n", k, (double) hits / totalTruth);
        System.out.printf("Avg candidates/query  = %.1f%n", (double) totalCandidates / queries);
        System.out.printf("Brute-force candidates = %d%n", n);
    }

    static int[] bruteForce(double[][] data, double[] q, int k) {
        Integer[] idx = new Integer[data.length];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> Double.compare(cosine(q, data[b]), cosine(q, data[a])));
        int[] out = new int[k];
        for (int i = 0; i < k; i++) out[i] = idx[i];
        return out;
    }

    @SuppressWarnings("unchecked")
    static class LSHIndex {
        int bits, tables, dim;
        double[][][] hyperplanes;
        Map<Long, List<Integer>>[] buckets;

        LSHIndex(int bits, int tables, int dim) {
            this.bits = bits; this.tables = tables; this.dim = dim;
            hyperplanes = new double[tables][bits][dim];
            for (int t = 0; t < tables; t++)
                for (int b = 0; b < bits; b++)
                    for (int d = 0; d < dim; d++)
                        hyperplanes[t][b][d] = RNG.nextGaussian();
            buckets = new HashMap[tables];
            for (int t = 0; t < tables; t++) buckets[t] = new HashMap<>();
        }

        long signature(double[] v, int table) {
            long sig = 0;
            for (int b = 0; b < bits; b++) {
                double dot = 0;
                for (int d = 0; d < dim; d++) dot += hyperplanes[table][b][d] * v[d];
                if (dot >= 0) sig |= (1L << b);
            }
            return sig;
        }

        void add(int id, double[] v) {
            for (int t = 0; t < tables; t++)
                buckets[t].computeIfAbsent(signature(v, t), x -> new ArrayList<>()).add(id);
        }

        int[] query(double[] q, int k, double[][] data) {
            Set<Integer> candidates = new HashSet<>();
            for (int t = 0; t < tables; t++) {
                List<Integer> bucket = buckets[t].get(signature(q, t));
                if (bucket != null) candidates.addAll(bucket);
            }
            List<Integer> list = new ArrayList<>(candidates);
            list.sort((a, b) -> Double.compare(cosine(q, data[b]), cosine(q, data[a])));
            int[] out = new int[Math.min(k, list.size())];
            for (int i = 0; i < out.length; i++) out[i] = list.get(i);
            return out;
        }
    }

    static double cosine(double[] a, double[] b) {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
        return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
    }

    static double[][] clustered(int n, int dim, int clusters, double noise) {
        double[][] centers = new double[clusters][dim];
        for (double[] c : centers) for (int d = 0; d < dim; d++) c[d] = RNG.nextGaussian();
        double[][] out = new double[n][dim];
        for (int i = 0; i < n; i++) {
            double[] c = centers[i % clusters], v = new double[dim];
            double norm = 0;
            for (int d = 0; d < dim; d++) { v[d] = c[d] + noise * RNG.nextGaussian(); norm += v[d]*v[d]; }
            norm = Math.sqrt(norm);
            for (int d = 0; d < dim; d++) v[d] /= norm;
            out[i] = v;
        }
        return out;
    }
}
