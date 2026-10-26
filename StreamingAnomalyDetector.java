import java.util.*;

/**
 * Streaming Anomaly Detector (ensemble)
 *
 * Three complementary online detectors, combined by majority vote:
 *
 *   1. EWMA z-score      -- detects slow mean shifts, O(1) memory, adapts fast
 *   2. Robust MAD        -- median/MAD on a sliding window, resistant to outliers
 *   3. Page-Hinkley      -- classic change detection; cumulative deviation test
 *
 * Each is O(1) (or O(W)) memory and single-pass. No batch retraining.
 *
 * Demo: synthetic series with a mean shift at t=1200 (10 -> 14).
 */
public class StreamingAnomalyDetector {

    static final Random RNG = new Random(3);

    public static void main(String[] args) {
        int n = 2000, changeAt = 1200;
        double[] series = new double[n];
        for (int i = 0; i < n; i++) {
            double mean = (i < changeAt) ? 10.0 : 14.0;
            series[i] = mean + RNG.nextGaussian();
        }

        EwmaZ ewma = new EwmaZ(0.05, 3.5);
        RobustMAD mad = new RobustMAD(100, 3.5);
        PageHinkley ph = new PageHinkley(0.005, 4.0);

        int tp = 0, fp = 0, fn = 0, firstDetect = -1;
        for (int i = 0; i < n; i++) {
            double x = series[i];
            int votes = (ewma.update(x) ? 1 : 0) + (mad.update(x) ? 1 : 0) + (ph.update(x) ? 1 : 0);
            boolean alert = votes >= 2;
            boolean truth = i >= changeAt && i < changeAt + 100;
            if (alert && truth) tp++;
            else if (alert) fp++;
            else if (truth && i > changeAt + 5) fn++;
            if (alert && firstDetect < 0 && i >= changeAt) firstDetect = i;
        }

        System.out.println("=== Streaming Anomaly Detector (ensemble) ===");
        System.out.printf("Detected at t=%d  (true change t=%d, latency=%d)%n",
                firstDetect, changeAt, firstDetect - changeAt);
        System.out.printf("TP=%d  FP=%d  FN=%d%n", tp, fp, fn);
        System.out.printf("Precision=%.3f  Recall=%.3f%n",
                (double) tp / Math.max(1, tp + fp), (double) tp / Math.max(1, tp + fn));
    }

    static class EwmaZ {
        final double alpha, k;
        double mean = Double.NaN, var = 0;
        EwmaZ(double alpha, double k) { this.alpha = alpha; this.k = k; }
        boolean update(double x) {
            if (Double.isNaN(mean)) { mean = x; return false; }
            double prev = mean;
            mean = alpha * x + (1 - alpha) * mean;
            var  = alpha * (x - prev) * (x - prev) + (1 - alpha) * var;
            if (var <= 0) return false;
            return Math.abs(x - mean) / Math.sqrt(var) > k;
        }
    }

    static class RobustMAD {
        final int W; final double k;
        final double[] buf; int idx = 0, count = 0;
        RobustMAD(int W, double k) { this.W = W; this.k = k; buf = new double[W]; }
        boolean update(double x) {
            buf[idx] = x; idx = (idx + 1) % W; if (count < W) count++;
            if (count < 20) return false;
            double[] copy = Arrays.copyOf(buf, count);
            Arrays.sort(copy);
            double med = copy[count / 2];
            double[] dev = new double[count];
            for (int i = 0; i < count; i++) dev[i] = Math.abs(copy[i] - med);
            Arrays.sort(dev);
            double mad = dev[count / 2] * 1.4826;
            if (mad <= 0) return false;
            return Math.abs(x - med) > k * mad;
        }
    }

    static class PageHinkley {
        final double delta, lambda;
        double sum = 0, mean = 0, minSum = 0;
        long n = 0;
        PageHinkley(double delta, double lambda) { this.delta = delta; this.lambda = lambda; }
        boolean update(double x) {
            n++;
            mean += (x - mean) / n;
            sum  += x - mean - delta;
            minSum = Math.min(minSum, sum);
            if (sum - minSum > lambda) { sum = 0; minSum = 0; return true; }
            return false;
        }
    }
}
