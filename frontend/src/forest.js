import * as ort from "onnxruntime-web";
import { fromArrayBuffer } from "geotiff";
import stats from "./band_stats.json";

// the wasm runtime files are loaded from a CDN, matching the installed version
ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/`;

const TILE = 512;
const PIXEL_HA = 0.01; // 10 m Sentinel-2 pixels
const { scale: SCALE, mean: MEAN, std: STD } = stats;

let sessionPromise = null;
function getSession() {
    if (!sessionPromise) {
        sessionPromise = ort.InferenceSession.create("/forest_unet_int8.onnx", {
            executionProviders: ["wasm"],
        }).catch((e) => {
            sessionPromise = null;
            throw e;
        });
    }
    return sessionPromise;
}

export async function readGeoTiff(file) {
    const tiff = await fromArrayBuffer(await file.arrayBuffer());
    const image = await tiff.getImage();
    const raster = await image.readRasters(); // one typed array per band
    let max = 0;
    for (const band of raster) for (let i = 0; i < band.length; i++) if (band[i] > max) max = band[i];
    return { raster, width: image.getWidth(), height: image.getHeight(), bands: raster.length, max };
}

// same steps as src/dataset.py: /10000, clip to [0,1], standardize; edge-pad to full tiles
export async function runModel({ raster, width, height }, onProgress = () => { }) {
    const session = await getSession();
    const prob = new Float32Array(width * height);
    const tilesX = Math.ceil(width / TILE);
    const tilesY = Math.ceil(height / TILE);
    const plane = TILE * TILE;
    const input = new Float32Array(4 * plane);
    let done = 0;

    for (let ty = 0; ty < tilesY; ty++) {
        for (let tx = 0; tx < tilesX; tx++) {
            const x0 = tx * TILE;
            const y0 = ty * TILE;
            for (let b = 0; b < 4; b++) {
                const band = raster[b];
                for (let y = 0; y < TILE; y++) {
                    const row = Math.min(y0 + y, height - 1) * width;
                    for (let x = 0; x < TILE; x++) {
                        const v = Math.min(Math.max(band[row + Math.min(x0 + x, width - 1)] / SCALE, 0), 1);
                        input[b * plane + y * TILE + x] = (v - MEAN[b]) / STD[b];
                    }
                }
            }
            const out = await session.run({ input: new ort.Tensor("float32", input, [1, 4, TILE, TILE]) });
            const logits = out.logits.data;
            const h = Math.min(TILE, height - y0);
            const w = Math.min(TILE, width - x0);
            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    prob[(y0 + y) * width + x0 + x] = 1 / (1 + Math.exp(-logits[y * TILE + x]));
                }
            }
            done += 1;
            onProgress(`${done}/${tilesX * tilesY} tiles`);
        }
    }
    return { prob, raster, width, height };
}

function toDataUrl(width, height, fill) {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(width, height);
    fill(img.data);
    ctx.putImageData(img, 0, 0);
    return canvas.toDataURL("image/png");
}

// cheap: re-thresholds the stored probabilities, so the slider needs no model re-run
export function render({ prob, raster, width, height }, threshold) {
    const n = width * height;
    let forest = 0;
    for (let i = 0; i < n; i++) if (prob[i] > threshold) forest++;

    const image_png = toDataUrl(width, height, (d) => {
        for (let i = 0; i < n; i++) {
            for (let c = 0; c < 3; c++) d[i * 4 + c] = Math.min(Math.max(raster[c][i] / 1500, 0), 1) * 255;
            d[i * 4 + 3] = 255;
        }
    });
    const mask_png = toDataUrl(width, height, (d) => {
        for (let i = 0; i < n; i++) {
            const v = prob[i] > threshold ? 255 : 0;
            d[i * 4] = d[i * 4 + 1] = d[i * 4 + 2] = v;
            d[i * 4 + 3] = 255;
        }
    });
    const overlay_png = toDataUrl(width, height, (d) => {
        for (let i = 0; i < n; i++) {
            if (prob[i] > threshold) {
                d[i * 4] = 34; d[i * 4 + 1] = 197; d[i * 4 + 2] = 94; d[i * 4 + 3] = 255;
            }
        }
    });

    return {
        forest_percent: Math.round((10000 * forest) / n) / 100,
        forest_ha: Math.round(forest * PIXEL_HA * 10) / 10,
        width, height, image_png, mask_png, overlay_png,
    };
}