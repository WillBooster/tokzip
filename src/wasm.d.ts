declare module '*.wasm' {
  /** A compiled module under bundlers/runtimes with wasm support, or the file path (Bun). */
  const module: WebAssembly.Module | string;
  export default module;
}
