#!/usr/bin/env node
'use strict';

const fs = require('fs');

function main() {
  const inPath = process.argv[2];
  const outPath = process.argv[3];
  if (!inPath || !outPath) {
    console.error('Usage: node ua-arch-analyze.js <input.json> <output.json>');
    process.exit(1);
  }
  const data = JSON.parse(fs.readFileSync(inPath, 'utf8'));
  const fileNodes = data.fileNodes || [];
  const importEdges = data.importEdges || [];
  const allEdges = data.allEdges || [];

  const idToNode = {};
  fileNodes.forEach(n => { idToNode[n.id] = n; });

  // Common prefix of filePaths
  const paths = fileNodes.map(n => n.filePath || '');
  function topDir(fp) {
    const parts = fp.split('/');
    return parts.length > 1 ? parts[0] : 'root';
  }

  // A. Directory grouping
  const directoryGroups = {};
  fileNodes.forEach(n => {
    const g = topDir(n.filePath || n.name);
    (directoryGroups[g] = directoryGroups[g] || []).push(n.id);
  });

  // B. Node type grouping
  const nodeTypeGroups = {};
  fileNodes.forEach(n => {
    const t = n.type || 'file';
    (nodeTypeGroups[t] = nodeTypeGroups[t] || []).push(n.id);
  });

  // C. Fan in / out
  const fanIn = {}, fanOut = {};
  fileNodes.forEach(n => { fanIn[n.id] = 0; fanOut[n.id] = 0; });
  importEdges.forEach(e => {
    if (fanOut[e.source] !== undefined) fanOut[e.source]++;
    if (fanIn[e.target] !== undefined) fanIn[e.target]++;
  });

  // map id -> group
  const idToGroup = {};
  Object.keys(directoryGroups).forEach(g => directoryGroups[g].forEach(id => idToGroup[id] = g));

  // D. Cross-category edges
  const crossMap = {};
  allEdges.forEach(e => {
    const s = idToNode[e.source], t = idToNode[e.target];
    if (!s || !t) return;
    const key = (s.type||'file') + '->' + (t.type||'file') + ':' + e.type;
    crossMap[key] = (crossMap[key]||0)+1;
  });
  const crossCategoryEdges = Object.keys(crossMap).map(k => {
    const [pair, edgeType] = k.split(':');
    const [fromType, toType] = pair.split('->');
    return { fromType, toType, edgeType, count: crossMap[k] };
  });

  // E. Inter-group imports
  const interMap = {};
  importEdges.forEach(e => {
    const gs = idToGroup[e.source], gt = idToGroup[e.target];
    if (gs === undefined || gt === undefined || gs === gt) return;
    const key = gs + '->' + gt;
    interMap[key] = (interMap[key]||0)+1;
  });
  const interGroupImports = Object.keys(interMap).map(k => {
    const [from, to] = k.split('->');
    return { from, to, count: interMap[k] };
  });

  // F. Intra-group density
  const intraGroupDensity = {};
  Object.keys(directoryGroups).forEach(g => {
    let internal = 0, total = 0;
    importEdges.forEach(e => {
      const gs = idToGroup[e.source], gt = idToGroup[e.target];
      if (gs === g || gt === g) total++;
      if (gs === g && gt === g) internal++;
    });
    intraGroupDensity[g] = { internalEdges: internal, totalEdges: total, density: total ? +(internal/total).toFixed(3) : 0 };
  });

  // G. Pattern matching
  const dirPatterns = [
    [/^(routes|api|controllers|endpoints|handlers)$/, 'api'],
    [/^(services|core|lib|domain|logic)$/, 'service'],
    [/^(models|db|data|persistence|repository|entities)$/, 'data'],
    [/^(utils|helpers|common|shared|tools)$/, 'utility'],
    [/^(config|constants|env|settings)$/, 'config'],
    [/^(tests?|spec|specs|__tests__)$/, 'test'],
    [/^(docs|documentation|wiki)$/, 'documentation'],
  ];
  function fileLevelPattern(n) {
    const fp = n.filePath || n.name;
    const base = n.name || fp.split('/').pop();
    if (/\.(test|spec)\./.test(base) || /_test\.|test_/.test(base)) return 'test';
    if (/\.d\.ts$/.test(base)) return 'types';
    if (base === '__init__.py') return 'entry';
    if (base === 'main.py') return 'entry';
    if (/\.(md|rst)$/.test(base)) return 'documentation';
    if (/ignore$/.test(base) || /^\..*ignore/.test(base)) return 'config';
    if ((n.tags||[]).includes('configuration') || (n.tags||[]).includes('tooling')) return 'config';
    return null;
  }
  const patternMatches = {};
  Object.keys(directoryGroups).forEach(g => {
    for (const [re, label] of dirPatterns) { if (re.test(g)) { patternMatches[g] = label; break; } }
  });
  const filePatternMatches = {};
  fileNodes.forEach(n => { const p = fileLevelPattern(n); if (p) filePatternMatches[n.id] = p; });

  // H. Deployment topology
  const infraFiles = [];
  let hasDockerfile=false, hasCompose=false, hasK8s=false, hasTerraform=false, hasCI=false;
  fileNodes.forEach(n => {
    const fp = n.filePath || n.name;
    if (/Dockerfile/.test(fp)) { hasDockerfile=true; infraFiles.push(fp); }
    if (/docker-compose/.test(fp)) { hasCompose=true; infraFiles.push(fp); }
    if (/\.tf$|\.tfvars$/.test(fp)) { hasTerraform=true; infraFiles.push(fp); }
    if (/workflows\/|gitlab-ci|Jenkinsfile/.test(fp)) { hasCI=true; infraFiles.push(fp); }
  });

  // I. Data pipeline
  const dataPipeline = { schemaFiles: [], migrationFiles: [], dataModelFiles: [], apiHandlerFiles: [] };

  // J. Doc coverage
  const groupsWithDocs = [];
  Object.keys(directoryGroups).forEach(g => {
    const has = directoryGroups[g].some(id => /\.(md|rst)$/.test(idToNode[id].filePath || ''));
    if (has) groupsWithDocs.push(g);
  });
  const totalGroups = Object.keys(directoryGroups).length;
  const docCoverage = {
    groupsWithDocs: groupsWithDocs.length,
    totalGroups,
    coverageRatio: totalGroups ? +(groupsWithDocs.length/totalGroups).toFixed(2) : 0,
    undocumentedGroups: Object.keys(directoryGroups).filter(g => !groupsWithDocs.includes(g))
  };

  // K. Dependency direction
  const pairDir = {};
  interGroupImports.forEach(({from,to,count}) => { pairDir[from+'|'+to] = count; });
  const seen = new Set();
  const dependencyDirection = [];
  interGroupImports.forEach(({from,to}) => {
    const k = [from,to].sort().join('::');
    if (seen.has(k)) return; seen.add(k);
    const ab = pairDir[from+'|'+to]||0;
    const ba = pairDir[to+'|'+from]||0;
    if (ab >= ba) dependencyDirection.push({dependent: from, dependsOn: to});
    else dependencyDirection.push({dependent: to, dependsOn: from});
  });

  const filesPerGroup = {};
  Object.keys(directoryGroups).forEach(g => filesPerGroup[g] = directoryGroups[g].length);
  const nodeTypeCounts = {};
  Object.keys(nodeTypeGroups).forEach(t => nodeTypeCounts[t] = nodeTypeGroups[t].length);

  const result = {
    scriptCompleted: true,
    directoryGroups,
    nodeTypeGroups,
    crossCategoryEdges,
    interGroupImports,
    intraGroupDensity,
    patternMatches,
    filePatternMatches,
    deploymentTopology: { hasDockerfile, hasCompose, hasK8s, hasTerraform, hasCI, infraFiles },
    dataPipeline,
    docCoverage,
    dependencyDirection,
    fileStats: { totalFileNodes: fileNodes.length, filesPerGroup, nodeTypeCounts },
    fileFanIn: fanIn,
    fileFanOut: fanOut
  };
  fs.writeFileSync(outPath, JSON.stringify(result, null, 2));
  process.exit(0);
}

try { main(); } catch (e) { console.error(e && e.stack || e); process.exit(1); }
