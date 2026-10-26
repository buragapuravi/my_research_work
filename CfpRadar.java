import java.io.*;
import java.nio.file.*;
import java.time.*;
import java.time.format.DateTimeFormatter;
import java.util.*;

/**
 * CFP Radar — deadline tracker for conference speaking.
 *
 * Reads conferences.csv with header: name,deadline,location,topics,url,status
 * Prints urgent CFPs (<= 14 days) sorted by deadline, plus a "draft" hint.
 */
public class CfpRadar {

    record Conf(String name, LocalDate deadline, String location,
                String topics, String url, String status) {}

    static final int URGENT_DAYS = 14;

    public static void main(String[] args) throws IOException {
        Path file = Path.of(args.length > 0 ? args[0] : "conferences.csv");
        if (!Files.exists(file)) { writeSample(file); System.out.println("Wrote sample " + file + " — edit and rerun."); return; }

        LocalDate today = LocalDate.now();
        List<Conf> confs = new ArrayList<>();
        List<String> lines = Files.readAllLines(file);
        for (int i = 1; i < lines.size(); i++) {
            String[] p = lines.get(i).split(",", -1);
            if (p.length < 6) continue;
            confs.add(new Conf(p[0].trim(),
                    LocalDate.parse(p[1].trim(), DateTimeFormatter.ISO_DATE),
                    p[2].trim(), p[3].trim(), p[4].trim(), p[5].trim()));
        }

        confs.removeIf(c -> c.status.equalsIgnoreCase("submitted"));
        confs.sort(Comparator.comparing(Conf::deadline));

        System.out.println("=== CFP Radar (" + today + ") ===");
        for (Conf c : confs) {
            long days = Duration.between(today.atStartOfDay(), c.deadline.atStartOfDay()).toDays();
            String flag = days <= URGENT_DAYS ? "🔴 URGENT" : days <= 30 ? "🟡 soon  " : "🟢 later ";
            System.out.printf("%s %3d d  %-28s %-15s %s%n", flag, days, c.name, c.location, c.topics);
            if (days <= URGENT_DAYS && days >= 0) {
                System.out.println("         → draft: cfp draft \"" + c.topics.split(";")[0] + "\" --name " + c.name);
            }
        }
    }

    static void writeSample(Path file) throws IOException {
        String csv = """
                name,deadline,location,topics,url,status
                World AI Summit,2026-06-15,Amsterdam,agentic AI;RAG;AI governance,https://example.com,planned
                KubeCon NA,2026-08-10,Salt Lake City,SRE;chaos engineering;platform,https://example.com,planned
                Global Tech Conference,2026-05-01,Santa Clara,GenAI platforms;regulated AI,https://example.com,drafting
                """;
        Files.writeString(file, csv);
    }
}
