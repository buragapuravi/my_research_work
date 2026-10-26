import java.util.*;

/**
 * SLO-aware Request Router for LLM serving
 *
 * Problem: route each request to one of R replicas, minimizing latency while
 * preserving KV-cache prefix locality. Each replica holds a small LRU cache of
 * prefixes; cache miss adds a large service-time penalty.
 *
 * Policies compared:
 *   round-robin  : i % R
 *   least-loaded : min replica.queueEnd
 *   cache-aware  : replica with prefix cached (tie: min queue), with a load
 *                  guard that falls back to least-loaded if the cached replica
 *                  is > SLACK_MS behind.
 *
 * Metric: p50 / p99 latency, effective throughput, cache hit rate.
 */
public class SloAwareRouter {

    static final double SLACK_MS = 20.0;

    public static void main(String[] args) {
        int replicas = 10, requests = 50_000, prefixes = 50;
        double meanInterArrivalMs = 5.0, baseServiceMs = 10.0, missPenaltyMs = 20.0;

        System.out.println("=== SLO-aware Request Router ===");
        for (String policy : new String[]{"round-robin", "least-loaded", "cache-aware"}) {
            Result r = simulate(replicas, requests, prefixes,
                    meanInterArrivalMs, baseServiceMs, missPenaltyMs, policy);
            System.out.printf("%-13s p50=%7.2fms  p99=%7.2fms  thr=%7.0f/s  hit=%.2f%n",
                    policy, r.p50, r.p99, r.throughput, r.cacheHitRate);
        }
    }

    static Result simulate(int replicas, int requests, int prefixes,
                           double meanInterArrivalMs, double baseServiceMs,
                           double missPenaltyMs, String policy) {
        Random rng = new Random(42);
        double[] arrival = new double[requests];
        int[] prefix = new int[requests];
        double t = 0;
        for (int i = 0; i < requests; i++) {
            t += -Math.log(1 - rng.nextDouble()) * meanInterArrivalMs;
            arrival[i] = t;
            double u = rng.nextDouble();
            prefix[i] = Math.min(prefixes - 1, (int) Math.floor(prefixes * Math.pow(u, 3)));
        }

        Replica[] reps = new Replica[replicas];
        for (int r = 0; r < replicas; r++) reps[r] = new Replica(prefixes / 2 + 1);

        double[] latency = new double[requests];
        int hits = 0;

        for (int i = 0; i < requests; i++) {
            int chosen;
            if (policy.equals("round-robin")) {
                chosen = i % replicas;
            } else if (policy.equals("least-loaded")) {
                chosen = 0;
                for (int r = 1; r < replicas; r++)
                    if (reps[r].queueEnd < reps[chosen].queueEnd) chosen = r;
            } else {
                int cached = -1, least = 0;
                for (int r = 0; r < replicas; r++) {
                    if (reps[r].cache.containsKey(prefix[i])
                            && (cached < 0 || reps[r].queueEnd < reps[cached].queueEnd)) cached = r;
                    if (reps[r].queueEnd < reps[least].queueEnd) least = r;
                }
                chosen = (cached >= 0 && reps[cached].queueEnd <= reps[least].queueEnd + SLACK_MS)
                        ? cached : least;
            }

            Replica rep = reps[chosen];
            boolean hit = rep.cache.containsKey(prefix[i]);
            if (hit) hits++;
            double svc = baseServiceMs + (hit ? 0 : missPenaltyMs);
            double start = Math.max(arrival[i], rep.queueEnd);
            rep.queueEnd = start + svc;
            rep.cache.put(prefix[i], Boolean.TRUE);
            latency[i] = rep.queueEnd - arrival[i];
        }

        Arrays.sort(latency);
        double maxEnd = 0;
        for (Replica r : reps) maxEnd = Math.max(maxEnd, r.queueEnd);

        Result res = new Result();
        res.p50 = latency[requests / 2];
        res.p99 = latency[(int) (requests * 0.99)];
        res.throughput = requests / ((maxEnd - arrival[0]) / 1000.0);
        res.cacheHitRate = (double) hits / requests;
        return res;
    }

    static class Replica {
        final int capacity;
        final LinkedHashMap<Integer, Boolean> cache;
        double queueEnd = 0;
        Replica(int capacity) {
            this.capacity = capacity;
            this.cache = new LinkedHashMap<Integer, Boolean>(16, 0.75f, true) {
                @Override protected boolean removeEldestEntry(Map.Entry<Integer, Boolean> e) {
                    return size() > Replica.this.capacity;
                }
            };
        }
    }

    static class Result { double p50, p99, throughput, cacheHitRate; }
}
