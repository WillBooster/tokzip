declare module '*.wasm' {
  /** A compiled module, as Cloudflare Workers bundlers provide for a `.wasm` import. */
  const module: WebAssembly.Module;
  export default module;
}
