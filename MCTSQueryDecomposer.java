import java.util.*;

/**
 * MCTS Query Decomposer
 *
 * Problem: a multi-hop question must be decomposed into sub-questions. Which
 * decomposition is best is unknown until you retrieve. Search over
 * decomposition trees with a learned (simulated) value function.
 *
 * Reward(S) = coverage(S) - LAMBDA*|S| + SYNERGY_W * synergy(S)
 *   coverage   : fraction of target facts retrieved by the union
 *   synergy    : pairwise bonus for complementary sub-questions
 *                (models dependency between sub-questions)
 *
 * Search: UCT (Monte Carlo Tree Search), branching restricted to top-M
 * candidates by marginal gain, max depth = 4.
 *
 * Baseline: greedy best-marginal, random decomposition.
 */
public class MCTSQueryDecomposer {

    static final Random RNG = new Random(7);
    static int F, Q;
    static BitSet target;
    static BitSet[] retrieval;
    static double[][] coupling;

    static final int MAX_DEPTH = 4;
    static final int M = 8;
    static final double LAMBDA = 0.05;
    static final double SYNERGY_W = 0.3;

    public static void main(String[] args) {
        buildWorld(24, 30);
        report("MCTS",   mctsDecompose(2000));
        report("Greedy", greedyDecompose());
        report("Random", randomDecompose());
    }

    static void buildWorld(int facts, int questions) {
        F = facts; Q = questions;
        target = new BitSet(F);
        for (int f : new int[]{1, 6, 12, 18, 23}) target.set(f);

        retrieval = new BitSet[Q];
        for (int q = 0; q < Q; q++) {
            BitSet b = new BitSet(F);
            int base = RNG.nextInt(F), width = 2 + RNG.nextInt(3);
            for (int j = 0; j < width; j++) b.set((base + j) % F);
            retrieval[q] = b;
        }
        for (int f = target.nextSetBit(0); f >= 0; f = target.nextSetBit(f + 1)) {
            boolean covered = false;
            for (BitSet b : retrieval) if (b.get(f)) { covered = true; break; }
            if (!covered) retrieval[RNG.nextInt(Q)].set(f);
        }

        coupling = new double[Q][Q];
        for (int i = 0; i < Q; i++)
            for (int j = i + 1; j < Q; j++)
                if (RNG.nextDouble() < 0.10) {
                    double c = 0.02 + RNG.nextDouble() * 0.04;
                    coupling[i][j] = coupling[j][i] = c;
                }
    }

    static double reward(BitSet S) {
        BitSet covered = new BitSet(F);
        for (int q = S.nextSetBit(0); q >= 0; q = S.nextSetBit(q + 1)) covered.or(retrieval[q]);
        covered.and(target);
        double coverage = (double) covered.cardinality() / Math.max(1, target.cardinality());
        double syn = 0;
        for (int i = S.nextSetBit(0); i >= 0; i = S.nextSetBit(i + 1))
            for (int j = S.nextSetBit(i + 1); j >= 0; j = S.nextSetBit(j + 1))
                syn += coupling[i][j];
        return coverage - LAMBDA * S.cardinality() + SYNERGY_W * syn;
    }

    static List<Integer> availableActions(BitSet state) {
        List<Integer> idx = new ArrayList<>();
        for (int q = 0; q < Q; q++) if (!state.get(q)) idx.add(q);
        double base = reward(state);
        idx.sort((a, b) -> Double.compare(marginal(state, b, base), marginal(state, a, base)));
        return idx.subList(0, Math.min(M, idx.size()));
    }

    static double marginal(BitSet state, int q, double base) {
        BitSet s2 = (BitSet) state.clone();
        s2.set(q);
        return reward(s2) - base;
    }

    static int[] mctsDecompose(int iterations) {
        Node root = new Node(new BitSet(), null);
        for (int it = 0; it < iterations; it++) {
            Node leaf = select(root);
            backprop(leaf, rollout(leaf));
        }
        Node best = root;
        while (!best.children.isEmpty()) {
            Node next = null; double bestAvg = -1e9;
            for (Node c : best.children.values())
                if (c.visits > 0 && c.totalReward / c.visits > bestAvg) {
                    bestAvg = c.totalReward / c.visits; next = c;
                }
            if (next == null) break;
            best = next;
        }
        return toArray(best.state);
    }

    static Node select(Node node) {
        while (node.visits > 0 && node.state.cardinality() < MAX_DEPTH) {
            if (!node.fullyExpanded()) return expand(node);
            node = uctSelect(node);
        }
        return node;
    }

    static Node expand(Node node) {
        for (int a : availableActions(node.state)) {
            if (!node.children.containsKey(a)) {
                BitSet s = (BitSet) node.state.clone();
                s.set(a);
                Node child = new Node(s, node);
                node.children.put(a, child);
                return child;
            }
        }
        return node;
    }

    static Node uctSelect(Node node) {
        Node best = null; double bestUct = -1e9;
        double logN = Math.log(Math.max(2, node.visits));
        for (Node c : node.children.values()) {
            double uct = c.visits == 0 ? Double.POSITIVE_INFINITY
                    : c.totalReward / c.visits + 1.4 * Math.sqrt(logN / c.visits);
            if (uct > bestUct) { bestUct = uct; best = c; }
        }
        return best;
    }

    static double rollout(Node node) {
        BitSet s = (BitSet) node.state.clone();
        while (s.cardinality() < MAX_DEPTH) {
            List<Integer> actions = availableActions(s);
            if (actions.isEmpty()) break;
            s.set(actions.get(RNG.nextInt(actions.size())));
        }
        return reward(s);
    }

    static void backprop(Node node, double reward) {
        while (node != null) { node.visits++; node.totalReward += reward; node = node.parent; }
    }

    static int[] greedyDecompose() {
        BitSet s = new BitSet();
        while (s.cardinality() < MAX_DEPTH) {
            double base = reward(s);
            int best = -1; double bestGain = 1e-9;
            for (int q = 0; q < Q; q++) {
                if (s.get(q)) continue;
                double g = marginal(s, q, base);
                if (g > bestGain) { bestGain = g; best = q; }
            }
            if (best < 0) break;
            s.set(best);
        }
        return toArray(s);
    }

    static int[] randomDecompose() {
        BitSet s = new BitSet();
        List<Integer> pool = new ArrayList<>();
        for (int i = 0; i < Q; i++) pool.add(i);
        Collections.shuffle(pool, RNG);
        for (int i = 0; i < MAX_DEPTH; i++) s.set(pool.get(i));
        return toArray(s);
    }

    static int[] toArray(BitSet b) {
        int[] out = new int[b.cardinality()];
        int i = 0;
        for (int q = b.nextSetBit(0); q >= 0; q = b.nextSetBit(q + 1)) out[i++] = q;
        return out;
    }

    static void report(String name, int[] decomp) {
        BitSet s = new BitSet();
        for (int q : decomp) s.set(q);
        BitSet covered = new BitSet(F);
        for (int q : decomp) covered.or(retrieval[q]);
        covered.and(target);
        System.out.printf("%-7s reward=%.4f  coverage=%d/%d  subs=%s%n",
                name, reward(s), covered.cardinality(), target.cardinality(),
                Arrays.toString(decomp));
    }

    static class Node {
        BitSet state; Node parent;
        Map<Integer, Node> children = new HashMap<>();
        int visits = 0; double totalReward = 0;
        Node(BitSet state, Node parent) { this.state = state; this.parent = parent; }
        boolean fullyExpanded() { return children.size() >= availableActions(state).size(); }
    }
}
