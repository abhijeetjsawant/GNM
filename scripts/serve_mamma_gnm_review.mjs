#!/usr/bin/env node
/** Serve exactly one review directory and the pinned Three.js vendor tree. */
import { createServer } from "node:http";
import { createReadStream } from "node:fs";
import { dirname, extname, resolve } from "node:path";

const [reviewArgument, portArgument = "8772"] = process.argv.slice(2);
if (!reviewArgument) throw new Error("usage: serve_mamma_gnm_review.mjs REVIEW_DIRECTORY [PORT]");
const reviewRoot = resolve(reviewArgument);
const vendorRoot = resolve(dirname(new URL(import.meta.url).pathname), "..", ".cache", "autoanim_gnm", "viewer", "three-0.183.2");
const vendorPrefix = "/api/viewer/vendor/0.183.2/";
const contentTypes = new Map([[".html", "text/html; charset=utf-8"], [".js", "text/javascript; charset=utf-8"], [".glb", "model/gltf-binary"], [".mp4", "video/mp4"]]);
const within = (root, relative) => { const candidate = resolve(root, relative); if (!candidate.startsWith(`${root}/`)) throw new Error("path traversal"); return candidate; };
const server = createServer((request, response) => {
  try {
    const path = decodeURIComponent(new URL(request.url, "http://localhost").pathname);
    const file = path.startsWith(vendorPrefix) ? within(vendorRoot, path.slice(vendorPrefix.length)) : within(reviewRoot, path === "/" ? "only-3d-review.html" : path.slice(1));
    response.writeHead(200, { "content-type": contentTypes.get(extname(file)) ?? "application/octet-stream", "cache-control": "no-store" });
    createReadStream(file).on("error", () => response.writeHead(404).end()).pipe(response);
  } catch { response.writeHead(404).end(); }
});
server.listen(Number(portArgument), "127.0.0.1", () => console.log(`http://127.0.0.1:${portArgument}/`));
