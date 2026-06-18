#!/usr/bin/env node
"use strict";

const fs = require("fs");

function main() {
  const inPath = process.argv[2];
  const outPath = process.argv[3];
  if (!inPath || !outPath) {
    console.error("Usage: node ua-tour-analyze.js <input.json> <output.json>");
    process.exit(1);
  }

  const raw = fs.readFileSync(inPath, "utf8");
  const data = JSON.parse(raw);
  const nodes = data.nodes || [];
  const edges = data.edges || [];
  const layers = data.layers || [];

  const nodeById = new Map();
  for (const n of nodes) nodeById.set(n.id, n);

  // Fan in / out
  const fanIn = new Map();
  const fanOut = new Map();
  for (const n of nodes) { fanIn.set(n.id, 0); fanOut.set(n.id, 0); }
  for (const e of edges) {
    if (fanOut.has(e.source)) fanOut.set(e.source, fanOut.get(e.source) + 1);
    if (fanIn.has(e.target)) fanIn.set(e.target, fanIn.get(e.target) + 1);
  }

  const nameOf = (id) => (nodeById.get(id) ? nodeById.get(id).name : id);
  const summaryOf = (id) => (nodeById.get(id) ? nodeById.get(id).summary || "" : "");
  const typeOf = (id) => (nodeById.get(id) ? nodeById.get(id).type : "");

  const fanInRanking = nodes
    .map((n) => ({ id: n.id, fanIn: fanIn.get(n.id), name: n.name }))
    .sort((a, b) => b.fanIn - a.fanIn)
    .slice(0, 20);

  const fanOutRanking = nodes
    .map((n) => ({ id: n.id, fanOut: fanOut.get(n.id), name: n.name }))
    .sort((a, b) => b.fanOut - a.fanOut)
    .slice(0, 20);

  // Entry point candidates
  const entryNames = new Set([
    "index.ts","index.js","main.ts","main.js","app.ts","app.js","server.ts","server.js",
    "mod.rs","main.go","main.py","main.rs","manage.py","app.py","wsgi.py","asgi.py","run.py",
    "__main__.py","Application.java","Main.java","Program.cs","config.ru","index.php","App.swift",
    "Application.kt","main.cpp","main.c"
  ]);

  const fanOutVals = nodes.map((n) => fanOut.get(n.id)).sort((a, b) => b - a);
  const fanInVals = nodes.map((n) => fanIn.get(n.id)).sort((a, b) => a - b);
  const top10pctIdx = Math.max(0, Math.floor(fanOutVals.length * 0.1) - 1);
  const fanOutThreshold = fanOutVals[top10pctIdx] !== undefined ? fanOutVals[top10pctIdx] : Infinity;
  const bottom25Idx = Math.max(0, Math.floor(fanInVals.length * 0.25) - 1);
  const fanInThreshold = fanInVals[bottom25Idx] !== undefined ? fanInVals[bottom25Idx] : 0;

  const epScores = [];
  for (const n of nodes) {
    let score = 0;
    const fp = n.filePath || "";
    const depth = fp.split("/").length;
    if (n.type === "document") {
      if (n.name === "README.md" && depth === 1) score += 5;
      else if (/\.md$/.test(n.name) && depth === 1) score += 2;
    } else {
      if (entryNames.has(n.name)) score += 3;
      if (depth <= 2) score += 1;
      if (fanOut.get(n.id) >= fanOutThreshold && fanOutThreshold !== Infinity) score += 1;
      if (fanIn.get(n.id) <= fanInThreshold) score += 1;
    }
    if (score > 0) epScores.push({ id: n.id, score, name: n.name, summary: n.summary || "" });
  }
  epScores.sort((a, b) => b.score - a.score);
  const entryPointCandidates = epScores.slice(0, 5);

  // BFS from top code entry point
  const codeEntry = entryPointCandidates.find((c) => typeOf(c.id) !== "document");
  const startNode = codeEntry ? codeEntry.id : (nodes[0] ? nodes[0].id : null);

  const adj = new Map();
  for (const n of nodes) adj.set(n.id, []);
  for (const e of edges) {
    if ((e.type === "imports" || e.type === "calls" || e.type === "related") && adj.has(e.source)) {
      adj.get(e.source).push(e.target);
    }
  }

  const order = [];
  const depthMap = {};
  if (startNode) {
    const queue = [[startNode, 0]];
    const seen = new Set([startNode]);
    while (queue.length) {
      const [cur, d] = queue.shift();
      order.push(cur);
      depthMap[cur] = d;
      for (const nx of (adj.get(cur) || [])) {
        if (!seen.has(nx)) { seen.add(nx); queue.push([nx, d + 1]); }
      }
    }
  }
  const byDepth = {};
  for (const id of order) {
    const d = String(depthMap[id]);
    if (!byDepth[d]) byDepth[d] = [];
    byDepth[d].push(id);
  }

  // Non-code inventory
  const nonCodeFiles = { documentation: [], infrastructure: [], data: [], config: [] };
  for (const n of nodes) {
    const entry = { id: n.id, name: n.name, type: n.type, summary: n.summary || "" };
    if (n.type === "document") nonCodeFiles.documentation.push(entry);
    else if (["service","pipeline","resource"].includes(n.type)) nonCodeFiles.infrastructure.push(entry);
    else if (["table","schema","endpoint"].includes(n.type)) nonCodeFiles.data.push(entry);
    else if (n.type === "config") nonCodeFiles.config.push(entry);
  }

  // Clusters: bidirectional pairs then expand
  const edgeSet = new Set(edges.map((e) => e.source + "|" + e.target));
  const pairs = [];
  for (const e of edges) {
    if (edgeSet.has(e.target + "|" + e.source) && e.source < e.target) {
      pairs.push([e.source, e.target]);
    }
  }
  const neighbors = new Map();
  for (const n of nodes) neighbors.set(n.id, new Set());
  for (const e of edges) {
    if (neighbors.has(e.source)) neighbors.get(e.source).add(e.target);
    if (neighbors.has(e.target)) neighbors.get(e.target).add(e.source);
  }
  const clusters = [];
  for (const [a, b] of pairs) {
    const cluster = new Set([a, b]);
    for (const n of nodes) {
      if (cluster.has(n.id)) continue;
      let cnt = 0;
      for (const m of cluster) if (neighbors.get(n.id) && neighbors.get(n.id).has(m)) cnt++;
      if (cnt >= 2 && cluster.size < 5) cluster.add(n.id);
    }
    let edgeCount = 0;
    const arr = [...cluster];
    for (const e of edges) {
      if (cluster.has(e.source) && cluster.has(e.target)) edgeCount++;
    }
    clusters.push({ nodes: arr, edgeCount });
  }
  clusters.sort((a, b) => b.edgeCount - a.edgeCount);
  const topClusters = clusters.slice(0, 10);

  // node summary index
  const nodeSummaryIndex = {};
  for (const n of nodes) {
    nodeSummaryIndex[n.id] = { name: n.name, type: n.type, summary: n.summary || "" };
  }

  const result = {
    scriptCompleted: true,
    entryPointCandidates,
    fanInRanking,
    fanOutRanking,
    bfsTraversal: { startNode, order, depthMap, byDepth },
    nonCodeFiles,
    clusters: topClusters,
    layers: { count: layers.length, list: layers },
    nodeSummaryIndex,
    totalNodes: nodes.length,
    totalEdges: edges.length
  };

  fs.writeFileSync(outPath, JSON.stringify(result, null, 2));
  process.exit(0);
}

try { main(); } catch (err) {
  console.error("Fatal: " + (err && err.stack ? err.stack : err));
  process.exit(1);
}
