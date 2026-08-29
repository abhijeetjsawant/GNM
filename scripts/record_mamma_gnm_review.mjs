#!/usr/bin/env node
/** Record a source-synchronised MAMMA + GNM review page as an MP4.
 *
 * The browser viewer is the same Three.js renderer used in review.  Serving
 * only the requested artifact directory plus its pinned vendor module avoids
 * exposing the workspace to the browser process.
 */

import { createServer } from "node:http";
import { mkdtemp, rm } from "node:fs/promises";
import { createReadStream } from "node:fs";
import { spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import { dirname, extname, join, resolve } from "node:path";
import { chromium } from "playwright";

const [artifactArgument, outputArgument] = process.argv.slice(2);
if (!artifactArgument || !outputArgument) {
  throw new Error("usage: record_mamma_gnm_review.mjs ARTIFACT_DIRECTORY OUTPUT.mp4");
}

const artifactRoot = resolve(artifactArgument);
const output = resolve(outputArgument);
const vendorRoot = resolve(
  dirname(new URL(import.meta.url).pathname),
  "..",
  ".cache",
  "autoanim_gnm",
  "viewer",
  "three-0.183.2",
);
const mime = new Map([
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".glb", "model/gltf-binary"],
  [".mp4", "video/mp4"],
  [".json", "application/json"],
]);

function resolveUnder(root, relative) {
  const candidate = resolve(root, relative);
  if (!candidate.startsWith(`${root}/`) && candidate !== root) {
    throw new Error("path traversal rejected");
  }
  return candidate;
}

const server = createServer(async (request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url, "http://localhost").pathname);
    const vendorPrefix = "/api/viewer/vendor/0.183.2/";
    const file = pathname.startsWith(vendorPrefix)
      ? resolveUnder(vendorRoot, pathname.slice(vendorPrefix.length))
      : resolveUnder(artifactRoot, pathname === "/" ? "review.html" : pathname.slice(1));
    response.writeHead(200, {
      "content-type": mime.get(extname(file)) ?? "application/octet-stream",
      "cache-control": "no-store",
    });
    createReadStream(file).on("error", () => response.destroy()).pipe(response);
  } catch {
    response.writeHead(404).end();
  }
});

await new Promise((done) => server.listen(0, "127.0.0.1", done));
const address = server.address();
if (!address || typeof address === "string") throw new Error("review server did not bind");

const recordingDirectory = await mkdtemp(join(tmpdir(), "autoanim-mamma-review-"));
try {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1280, height: 720 } });
  const page = await context.newPage();
  const errors = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  await page.goto(`http://127.0.0.1:${address.port}/review.html`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => document.querySelector("#source")?.readyState >= 2);
  await page.waitForTimeout(1800);
  for (let frame = 0; frame < 150; frame += 1) {
    await page.evaluate(async (seconds) => {
      const source = document.querySelector("#source");
      if (Math.abs(source.currentTime - seconds) > 0.0001) {
        source.currentTime = seconds;
        await new Promise((done) => source.addEventListener("seeked", done, { once: true }));
      }
      await new Promise(requestAnimationFrame);
      await new Promise(requestAnimationFrame);
    }, frame / 30);
    await page.screenshot({
      path: join(recordingDirectory, `frame-${String(frame).padStart(4, "0")}.png`),
    });
  }
  await page.close();
  await context.close();
  await browser.close();
  if (errors.length) throw new Error(`review browser errors: ${errors.join(" | ")}`);
  const transcode = spawnSync(
    "ffmpeg",
    [
      "-y", "-hide_banner", "-loglevel", "error", "-framerate", "30",
      "-start_number", "0", "-i", join(recordingDirectory, "frame-%04d.png"),
      "-frames:v", "150", "-c:v", "libx264", "-crf", "18", "-pix_fmt",
      "yuv420p", output,
    ],
    { stdio: "pipe" },
  );
  if (transcode.status !== 0) throw new Error(transcode.stderr.toString("utf8"));
  console.log(output);
} finally {
  await new Promise((done) => server.close(done));
  await rm(recordingDirectory, { recursive: true, force: true });
}
