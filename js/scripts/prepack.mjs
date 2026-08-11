// SPDX-FileCopyrightText: ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Runs before `npm pack` / `npm publish`.
 *
 *   node scripts/prepack.mjs               check the version, the WebAssembly
 *                                          artifacts and refresh LICENSE
 *   node scripts/prepack.mjs --check-only  only check that package.json and
 *                                          VERSION_NUMBER agree
 *   node scripts/prepack.mjs --write       write VERSION_NUMBER into
 *                                          package.json instead of failing
 */

import { copyFileSync, existsSync, readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const packageJsonPath = fileURLToPath(new URL('../package.json', import.meta.url));
const versionNumberPath = fileURLToPath(new URL('../../VERSION_NUMBER', import.meta.url));
const licensePath = fileURLToPath(new URL('../../LICENSE', import.meta.url));
const packageLicensePath = fileURLToPath(new URL('../LICENSE', import.meta.url));

const artifacts = ['../dist/onnxoptimizer.mjs', '../dist/onnxoptimizer.wasm'].map((path) =>
  fileURLToPath(new URL(path, import.meta.url)),
);

const checkOnly = process.argv.includes('--check-only');
const write = process.argv.includes('--write');

function fail(message) {
  console.error(`prepack: ${message}`);
  process.exit(1);
}

const version = readFileSync(versionNumberPath, 'utf8').trim();
const packageJson = readFileSync(packageJsonPath, 'utf8');
const parsed = JSON.parse(packageJson);

if (parsed.version !== version) {
  if (!write) {
    fail(
      `package.json says ${parsed.version} but VERSION_NUMBER says ${version}. ` +
        'Run `node scripts/prepack.mjs --write` to sync them.',
    );
  }
  // Rewrite the one line rather than re-serializing, to keep the formatting.
  const updated = packageJson.replace(
    /^(\s*"version":\s*")[^"]*(")/m,
    `$1${version}$2`,
  );
  if (updated === packageJson) {
    fail('cannot find the version field in package.json');
  }
  writeFileSync(packageJsonPath, updated);
  console.log(`prepack: set the package version to ${version}`);
}

if (checkOnly) {
  process.exit(0);
}

const missing = artifacts.filter((path) => !existsSync(path));
if (missing.length > 0) {
  fail(
    `the WebAssembly module is missing (${missing.join(', ')}). ` +
      'Build it with scripts/build_wasm.sh before publishing.',
  );
}

// The package is published from js/, which cannot reach the repository's
// LICENSE through the `files` field.
copyFileSync(licensePath, packageLicensePath);

console.log(`prepack: onnxoptimizer ${version} is ready to pack`);
