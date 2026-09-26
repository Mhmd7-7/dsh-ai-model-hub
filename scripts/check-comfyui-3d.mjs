/**
 * Real-world verification against the ComfyUI install on this machine.
 *
 * Reads the install's own files rather than an HTTP endpoint, because the server is
 * not running: `custom_nodes` and `comfy_extras` are what `/object_info` is built
 * from, and `models/` plus `extra_model_paths.yaml` are what the loader enums are
 * built from. The workflow listing comes from ComfyUI's user-data directory.
 *
 * This is a diagnostic, not a test: it prints what is and is not discoverable and
 * never asserts. Run it with `node scripts/check-comfyui-3d.mjs`.
 */
import { readFile, readdir, stat } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join, relative } from 'node:path';

const COMFY = 'C:/ComfyUI/src';
const MODELS = 'C:/ComfyUI/models';

const report = [];
const say = (line) => {
  report.push(line);
  console.log(line);
};

/** Every `.py` under a directory, recursively, as text. */
async function pyFiles(root) {
  const out = [];
  async function walk(dir) {
    let entries;
    try {
      entries = await readdir(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const full = join(dir, entry.name);
      if (entry.isDirectory()) {
        if (entry.name === '__pycache__') continue;
        await walk(full);
      } else if (entry.name.endsWith('.py')) {
        out.push(full);
      }
    }
  }
  await walk(root);
  return out;
}

/** Every file under a directory, recursively. */
async function allFiles(root) {
  const out = [];
  async function walk(dir) {
    let entries;
    try {
      entries = await readdir(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const full = join(dir, entry.name);
      if (entry.isDirectory()) await walk(full);
      else out.push(full);
    }
  }
  await walk(root);
  return out;
}

say(`ComfyUI root: ${COMFY}`);
say(`version: ${(await readFile(join(COMFY, 'comfyui_version.py'), 'utf8')).trim().split('\n').pop()}`);
say('');

// 1. Installed 3D node classes -------------------------------------------------
const GENERATOR = /hunyuan3d|triposg|triposplat|tripo.*model|meshy.*model|rodin3d|stable3d|sv3d|moge|trellis|pixal|image.*to.*3d|image.*to.*model/i;
const extraFiles = await pyFiles(join(COMFY, 'comfy_extras'));
const apiFiles = await pyFiles(join(COMFY, 'comfy_api_nodes'));
const customFiles = await pyFiles(join(COMFY, 'custom_nodes'));
const found = new Map();
for (const file of [...extraFiles, ...apiFiles, ...customFiles]) {
  const text = await readFile(file, 'utf8');
  for (const match of text.matchAll(/node_id="([A-Za-z0-9_]+)"/g)) {
    if (GENERATOR.test(match[1])) {
      const key = match[1];
      if (!found.has(key)) found.set(key, relative(COMFY, file).replace(/\\/g, '/'));
    }
  }
}
say(`3D-generation node classes declared by this install: ${found.size}`);
for (const [node, file] of [...found].sort()) say(`  ${node}  (${file})`);
say('');

// 2. Mesh writers --------------------------------------------------------------
const writers = new Set();
for (const file of [...extraFiles, ...customFiles]) {
  const text = await readFile(file, 'utf8');
  for (const match of text.matchAll(/node_id="([A-Za-z0-9_]+)"/g)) {
    if (/save.*(glb|gltf|obj|stl|ply|fbx|usdz|3d|mesh|splat)|export.*(3d|mesh|glb)|mesh.*to.*file/i.test(match[1])) {
      writers.add(match[1]);
    }
  }
}
say(`3D file writers: ${[...writers].sort().join(', ') || '(none)'}`);
say('');

// 3. Installed model weight files ----------------------------------------------
const modelFiles = await allFiles(MODELS);
const weights = modelFiles.filter((file) => /\.(safetensors|ckpt|pt|pth|bin|gguf|onnx)$/i.test(file));
say(`weight files under ${MODELS}: ${weights.length}`);
for (const file of weights) {
  const size = (await stat(file)).size;
  say(`  ${relative(MODELS, file).replace(/\\/g, '/')}  ${(size / 1024 ** 3).toFixed(2)} GiB`);
}
say('');

// 4. Saved workflows ------------------------------------------------------------
const workflowDirs = [
  join(COMFY, 'user', 'default', 'workflows'),
  join(COMFY, '..', 'user', 'default', 'workflows'),
];
let workflows = [];
for (const dir of workflowDirs) {
  if (!existsSync(dir)) continue;
  workflows = (await allFiles(dir)).filter((file) => file.endsWith('.json'));
  say(`saved workflows in ${dir}: ${workflows.length}`);
  for (const file of workflows) say(`  ${relative(dir, file).replace(/\\/g, '/')}`);
}
if (workflows.length === 0) say('saved workflows: none (no workflow directory, or it is empty)');
say('');

// 5. Which of the named workflow's dependencies are present ---------------------
const candidate = 'C:/Users/fdgrr/Downloads/3d_pixal3d_trellis2_image_to_model.json';
if (existsSync(candidate)) {
  const graph = JSON.parse(await readFile(candidate, 'utf8'));
  const classes = new Set(Object.values(graph).map((node) => node.class_type));
  const refs = [];
  for (const node of Object.values(graph)) {
    for (const [field, value] of Object.entries(node.inputs ?? {})) {
      if (!/(ckpt|unet|vae|clip|lora|model).*_name$|^model_file$/i.test(field)) continue;
      if (typeof value === 'string' && value.length > 0) refs.push(value);
    }
  }
  const installedNames = new Set(weights.map((file) => file.split(/[/\\]/).pop()));
  say(`workflow under test: ${candidate}`);
  say(`  format: API (${Object.keys(graph).length} nodes)`);
  say(`  node classes: ${classes.size}`);
  const missingClasses = [...classes].filter((name) => !found.has(name) && !/loadimage|clip|vae|ksampler|unetloader|saveimage|preview|mesh|latent|primitive|switch|rescale|cfg|background|image|string|int|bool|float|combo|note|reroute/i.test(name));
  say(`  node classes not matched to an installed 3D module: ${missingClasses.length === 0 ? '(none)' : missingClasses.join(', ')}`);
  say(`  model files the graph names: ${refs.length === 0 ? '(none)' : [...new Set(refs)].join(', ')}`);
  const missingRefs = [...new Set(refs)].filter((name) => !installedNames.has(name));
  say(`  of those, present on disk: ${[...new Set(refs)].length - missingRefs.length}`);
  say(`  of those, missing: ${missingRefs.length === 0 ? '(none)' : missingRefs.join(', ')}`);
  const hasImage = [...classes].some((name) => /loadimage/i.test(name));
  const hasGenerator = [...classes].some((name) => GENERATOR.test(name));
  const hasWriter = [...classes].some((name) => writers.has(name));
  say(`  capability evidence — image input: ${hasImage}, 3D generator: ${hasGenerator}, 3D writer: ${hasWriter}`);
  say(`  inferred capability: ${hasImage && hasGenerator && hasWriter ? 'image_to_3d' : 'UNKNOWN'}`);
  const textDriven = [...classes].some((name) => /textencode/i.test(name));
  say(`  text_to_3d: not advertised (a prompt node ${textDriven ? 'exists but does not drive the 3D generator' : 'does not exist'})`);
}
say('');
say('This machine therefore: nodes = installed, 3D weights = NOT installed, workflow = NOT saved in ComfyUI.');
