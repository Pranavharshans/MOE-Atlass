import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const frontendRoot = resolve(here, "..");
const sourceRoot = resolve(frontendRoot, "dist");
const packageRoot = resolve(frontendRoot, "../src/moeatlas/server/static");
const sourceAssets = resolve(sourceRoot, "assets");
const targetAssets = resolve(packageRoot, "assets");

await mkdir(packageRoot, { recursive: true });
await rm(targetAssets, { recursive: true, force: true });
await cp(sourceAssets, targetAssets, { recursive: true });
const publishedIndex = resolve(packageRoot, "index.html");
await cp(resolve(sourceRoot, "index.html"), publishedIndex);
const indexMarkup = await readFile(publishedIndex, "utf8");
await writeFile(publishedIndex, indexMarkup.replace(/^<!doctype html>/, "<!DOCTYPE html>"));
await cp(resolve(frontendRoot, "public/favicon.svg"), resolve(packageRoot, "favicon.svg"));

console.log(`Published React assets to ${packageRoot}`);
