import java.util.*;

/**
 * Submodular Context Selector
 *
 * Problem: given a query embedding and N chunks from a vector search first
 * stage, select K chunks that maximize coverage of the query's neighborhood
 * while discouraging redundancy.
 *
 * Objective (facility location, monotone submodular):
 *   f(S) = sum_{v in C} max( sim(v,q), max_{j in S} sim(v,j) )
 *
 * Guarantee: greedy achieves (1 - 1/e) ~= 0.632 of the optimum for
 * monotone submodular maximization under a cardinality constraint.
 *
 * Complexity: O(K * N^2) after precomputing the similarity matrix.
 * Baseline: top-K by query similarity (ignores redundancy).
 */
public class SubmodularContextSelector {

    static final int DIM = 32;
    static final Random RNG = new Random(42);

    public static void main(String[] args) {
        int n = 200, k = 5, topics = 8;
        double[][] chunks = clusteredUnitVectors(n, topics, 0.35);
        double[] query = jitter(chunks[0], 0.35);   // query lives near cluster 0

        double[][] sim = new double[n][n];
        for (int i = 0; i < n; i++) {
            sim[i][i] = 1.0;
            for (int j = i + 1; j < n; j++) {
                double s = cosine(chunks[i], chunks[j]);
                sim[i][j] = sim[j][i] = s;
            }
        }
        double[] simQ = new double[n];
        for (int i = 0; i < n; i++) simQ[i] = cosine(chunks[i], query);

        int[] topK = topK(simQ, k);
        int[] greedy = greedyFacilityLocation(sim, simQ, n, k);

        System.out.println("=== Submodular Context Selector ===");
        System.out.printf("Objective (top-K)   = %.4f%n", objective(topK, sim, simQ, n));
        System.out.printf("Objective (greedy)  = %.4f%n", objective(greedy, sim, simQ, n));
        System.out.printf("Redundancy (top-K)  = %.4f%n", avgPairwiseSim(topK, sim));
        System.out.printf("Redundancy (greedy) = %.4f%n", avgPairwiseSim(greedy, sim));
        System.out.println("Top-K  picks: " + Arrays.toString(topK));
        System.out.println("Greedy picks: " + Arrays.toString(greedy));
    }

    static int[] greedyFacilityLocation(double[][] sim, double[] simQ, int n, int k) {
        double[] best = simQ.clone();          // current coverage per v, query as fixed facility
        boolean[] used = new boolean[n];
        int[] selected = new int[k];
        for (int step = 0; step < k; step++) {
            int bestC = -1; double bestGain = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < n; c++) {
                if (used[c]) continue;
                double gain = 0.0;
                for (int v = 0; v < n; v++)
                    gain += Math.max(best[v], sim[v][c]) - best[v];
                if (gain > bestGain) { bestGain = gain; bestC = c; }
            }
            selected[step] = bestC;
            used[bestC] = true;
            for (int v = 0; v < n; v++) best[v] = Math.max(best[v], sim[v][bestC]);
        }
        return selected;
    }

    static int[] topK(double[] scores, int k) {
        Integer[] idx = new Integer[scores.length];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> Double.compare(scores[b], scores[a]));
        int[] out = new int[k];
        for (int i = 0; i < k; i++) out[i] = idx[i];
        return out;
    }

    static double objective(int[] S, double[][] sim, double[] simQ, int n) {
        double[] best = simQ.clone();
        for (int j : S) for (int v = 0; v < n; v++) best[v] = Math.max(best[v], sim[v][j]);
        double sum = 0; for (double b : best) sum += b; return sum;
    }

    static double avgPairwiseSim(int[] S, double[][] sim) {
        if (S.length < 2) return 0;
        double sum = 0; int cnt = 0;
        for (int i = 0; i < S.length; i++)
            for (int j = i + 1; j < S.length; j++) { sum += sim[S[i]][S[j]]; cnt++; }
        return sum / cnt;
    }

    static double cosine(double[] a, double[] b) {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
        return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
    }

    static double[] clusteredUnitVectors(int n, int topics, double noise) {
        double[][] centers = new double[topics][DIM];
        for (double[] c : centers) for (int d = 0; d < DIM; d++) c[d] = RNG.nextGaussian();
        double[][] out = new double[n][DIM];
        for (int i = 0; i < n; i++) out[i] = jitter(centers[i % topics], noise);
        return out;
    }

    static double[] jitter(double[] base, double noise) {
        double[] v = new double[DIM];
        double norm = 0;
        for (int i = 0; i < DIM; i++) { v[i] = base[i] + noise * RNG.nextGaussian(); norm += v[i]*v[i]; }
        norm = Math.sqrt(norm);
        for (int i = 0; i < DIM; i++) v[i] /= norm;
        return v;
    }
}
