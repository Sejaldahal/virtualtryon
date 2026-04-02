import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import fs from "fs";
import { componentTagger } from "lovable-tagger";

function feedbackFilePlugin() {
  return {
    name: "feedback-file-plugin",
    configureServer(server: any) {
      server.middlewares.use("/api/feedback", async (req: any, res: any) => {
        if (req.method !== "POST") {
          res.statusCode = 405;
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ error: "Method not allowed" }));
          return;
        }

        let raw = "";
        req.on("data", (chunk: Buffer) => {
          raw += chunk.toString();
        });

        req.on("end", () => {
          try {
            const payload = JSON.parse(raw || "{}");
            const name = String(payload.name || "Anonymous").trim().slice(0, 80) || "Anonymous";
            const rating = payload.rating == null || payload.rating === "" ? "Not provided" : String(payload.rating);
            const message = String(payload.message || "").trim().slice(0, 2000);

            if (!message) {
              res.statusCode = 400;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: "Feedback message is required" }));
              return;
            }

            const feedbackPath = path.resolve(__dirname, "public", "feedback.txt");
            if (!fs.existsSync(feedbackPath)) {
              fs.writeFileSync(feedbackPath, "# User Feedback\n\n", "utf-8");
            }

            const entry = [
              `## Feedback - ${new Date().toISOString()}`,
              `Name: ${name}`,
              `Rating: ${rating}`,
              "Message:",
              message,
              "",
            ].join("\n");

            fs.appendFileSync(feedbackPath, `${entry}\n`, "utf-8");

            res.statusCode = 200;
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ ok: true }));
          } catch {
            res.statusCode = 500;
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ error: "Could not save feedback" }));
          }
        });
      });
    },
  };
}

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 8080,
  },
  plugins: [react(), feedbackFilePlugin(), mode === "development" && componentTagger()].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
