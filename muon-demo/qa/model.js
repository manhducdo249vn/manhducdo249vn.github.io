// The small GPT from muontail.py, run in plain JavaScript (fp32 maths on fp16 weights).
// Mirrors GPT.forward exactly: token + position embeddings, pre-norm blocks with RMSNorm,
// attention with RMSNorm on Q and K per head, GELU MLP, final RMSNorm, untied output head.
// Greedy decoding with a key/value cache, like ask.py's answer().
(function (root) {
  "use strict";

  const F16 = (() => {                       // fp16 bit pattern -> float32 value
    const t = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
      const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
      t[h] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
    }
    return t;
  })();

  function decodeWeights(pack, atob) {
    const bin = atob(pack.b64), n = bin.length, bytes = new Uint8Array(n);
    for (let i = 0; i < n; i++) bytes[i] = bin.charCodeAt(i);
    const u16 = new Uint16Array(bytes.buffer), W = {};
    for (const [name, [off, shape]] of Object.entries(pack.index)) {
      const size = shape.reduce((a, b) => a * b, 1), out = new Float32Array(size);
      for (let i = 0; i < size; i++) out[i] = F16[u16[off + i]];
      W[name] = out;
    }
    return W;
  }

  function linear(W, x, out, nOut, nIn) {    // out = W x, W is [nOut, nIn] row-major (nn.Linear)
    for (let o = 0; o < nOut; o++) {
      let s = 0;
      const r = o * nIn;
      for (let i = 0; i < nIn; i++) s += W[r + i] * x[i];
      out[o] = s;
    }
    return out;
  }
  function rmsnorm(x, w, out, off = 0, n = x.length, eps = 1e-6) {
    let ss = 0;
    for (let i = 0; i < n; i++) ss += x[off + i] * x[off + i];
    const r = 1 / Math.sqrt(ss / n + eps);
    for (let i = 0; i < n; i++) out[off + i] = x[off + i] * r * w[i];
    return out;
  }
  function erf(x) {                          // Abramowitz-Stegun 7.1.26, |error| < 1.5e-7
    const s = x < 0 ? -1 : 1; x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }
  const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));   // exact GELU, as torch F.gelu

  class GPT {
    constructor(W, cfg) {
      this.W = W; this.d = cfg.d; this.L = cfg.n_layer; this.h = cfg.n_head;
      this.hd = cfg.d / cfg.n_head; this.V = cfg.vocab; this.T = cfg.seq_len;
    }
    // One token at position pos; appends its keys/values to cache; returns the final hidden state.
    step(tok, pos, cache) {
      const { W, d, h, hd } = this;
      const x = new Float32Array(d), a = new Float32Array(d), q = new Float32Array(d), k = new Float32Array(d),
        v = new Float32Array(d), y = new Float32Array(d), o = new Float32Array(d),
        u = new Float32Array(4 * d), z = new Float32Array(d);
      const te = W["tok_emb.weight"], pe = W["pos_emb.weight"];
      for (let i = 0; i < d; i++) x[i] = te[tok * d + i] + pe[pos * d + i];
      for (let l = 0; l < this.L; l++) {
        const p = `blocks.${l}.`;
        rmsnorm(x, W[p + "ln1.weight"], a);
        linear(W[p + "attn.q.weight"], a, q, d, d);
        linear(W[p + "attn.k.weight"], a, k, d, d);
        linear(W[p + "attn.v.weight"], a, v, d, d);
        const qn = W[p + "attn.q_norm.weight"], kn = W[p + "attn.k_norm.weight"];
        for (let j = 0; j < h; j++) { rmsnorm(q, qn, q, j * hd, hd); rmsnorm(k, kn, k, j * hd, hd); }
        cache[l].k.push(k.slice()); cache[l].v.push(v.slice());
        const K = cache[l].k, Vv = cache[l].v, T = K.length, sc = new Float32Array(T), inv = 1 / Math.sqrt(hd);
        for (let j = 0; j < h; j++) {
          const b = j * hd;
          let mx = -Infinity;
          for (let t = 0; t < T; t++) {
            let s = 0;
            for (let i = 0; i < hd; i++) s += q[b + i] * K[t][b + i];
            sc[t] = s * inv; if (sc[t] > mx) mx = sc[t];
          }
          let sum = 0;
          for (let t = 0; t < T; t++) { sc[t] = Math.exp(sc[t] - mx); sum += sc[t]; }
          for (let i = 0; i < hd; i++) {
            let s = 0;
            for (let t = 0; t < T; t++) s += sc[t] * Vv[t][b + i];
            y[b + i] = s / sum;
          }
        }
        linear(W[p + "attn.o.weight"], y, o, d, d);
        for (let i = 0; i < d; i++) x[i] += o[i];
        rmsnorm(x, W[p + "ln2.weight"], a);
        linear(W[p + "mlp.fc_in.weight"], a, u, 4 * d, d);
        for (let i = 0; i < 4 * d; i++) u[i] = gelu(u[i]);
        linear(W[p + "mlp.fc_out.weight"], u, z, d, 4 * d);
        for (let i = 0; i < d; i++) x[i] += z[i];
      }
      return rmsnorm(x, W["ln_f.weight"], new Float32Array(d));
    }
    argmaxNext(hidden) {
      const H = this.W["lm_head.weight"], d = this.d;
      let best = -Infinity, arg = 0;
      for (let o = 0; o < this.V; o++) {
        let s = 0;
        const r = o * d;
        for (let i = 0; i < d; i++) s += H[r + i] * hidden[i];
        if (s > best) { best = s; arg = o; }
      }
      return arg;
    }
    // Greedy answer for a prompt (token ids); stops at eos or after maxNew tokens.
    generate(prompt, eos, maxNew = 8) {
      const cache = Array.from({ length: this.L }, () => ({ k: [], v: [] }));
      let hid = null, pos = 0;
      for (const t of prompt) hid = this.step(t, pos++, cache);
      const out = [];
      for (let n = 0; n < maxNew && pos < this.T; n++) {
        const nxt = this.argmaxNext(hid);
        if (nxt === eos) break;
        out.push(nxt);
        hid = this.step(nxt, pos++, cache);
      }
      return out;
    }
  }

  root.MuonQA = { decodeWeights, GPT };
})(typeof window !== "undefined" ? window : this);
