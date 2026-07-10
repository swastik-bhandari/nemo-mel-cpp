// Verify the CTC decode + label-parsing logic matches our proven Python decode,
// using the same logits our dummy model would produce, without a browser.
import fs from 'fs';

// Simulate label parsing exactly as index.html does
const labTxt = " \na\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\nn\no\np\nq\nr\ns\nt\nu\nv\nw\nx\ny\nz\n'";
let labels = labTxt.split('\n').map(l => l.replace(/\r$/,''));
if (labels.length && labels[labels.length-1] === '') labels.pop();
console.log('labels parsed:', labels.length, '| first="'+labels[0]+'"(space) last="'+labels[labels.length-1]+'"');

// CTC decode fn copied from index.html
function ctcDecode(dims, data, labels) {
  const [ , T, V] = dims;
  const blank = labels.length;
  let out = '', prev = -1;
  for (let t = 0; t < T; t++) {
    let best = 0, bestv = data[t*V];
    for (let v = 1; v < V; v++) { const val = data[t*V+v]; if (val > bestv) { bestv = val; best = v; } }
    if (best !== prev && best !== blank) out += labels[best];
    prev = best;
  }
  return out;
}

// Build a fake logit sequence that should decode to "hi" with blanks/repeats:
// indices: h=8, i=9, blank=28
// frames: [blank, h, h, blank, i, i, blank]  -> "hi"
const V = 29;
const seq = [28, 8, 8, 28, 9, 9, 28];
const data = new Float32Array(seq.length * V);
seq.forEach((idx, t) => { data[t*V + idx] = 10.0; });  // one-hot-ish
const text = ctcDecode([1, seq.length, V], data, labels);
console.log('decoded:', JSON.stringify(text), '(expected "hi")');
console.log(text === 'hi' ? 'PASS' : 'FAIL');
