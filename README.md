# QuartTET-ChatNoir core

4次元四面体メッシュの Rust コア。ブラウザに依存しない te4 読込・検査、6平面の回転、CPU 参照断面抽出、WebAssembly API と WebGPU compute shader を提供します。

## 役割

- `Mesh4`: 4D 頂点と四面体胞。これは4次元物体の **3次元境界** を四面体で表すモデルです。
- `Mesh4::section`: 四面体と超平面の交差を三角形化する CPU 参照実装。
- `shaders/section.wgsl`: 実際の viewer で使う GPU 断面抽出。1 invocation / 四面体、atomic による出力圧縮、間接描画用カウンタを生成。
- `ViewerModel`: wasm-bindgen API。読込時だけ CPU で検査と共有面の所有胞を決定し、GPU 用データを渡します。

viewer: <https://github.com/QuartTET-ChatNoir/viewer>

## 最小 TE4 仕様

この実装で受け付ける最小仕様です。全て little-endian。過去の全 exporter との互換性を保証するものではありません。

|順序|内容|型|
|---|---|---|
|1|file type = 1|u32|
|2|頂点数 n|u32|
|3|四面体胞数 m|u32|
|4|n 個の頂点 `(x,y,z,w)`|f32 × 4|
|5|m 個の胞 `(a,b,c,d)`|u32 × 4|

長さは厳密に `12 + 16n + 16m` bytes。通常は頂点番号が0始まり。1始まりは引数で明示できます。保存は0始まりです。type 2–4、法線、テクスチャ、周期性の拡張は未対応です。

切断軸 X/Y/Z/W を除いた3座標をそのまま3次元座標とします。回転順は XY → XZ → XW → YZ → YW → ZW。回転と切断は元モデルの AABB 中心を原点として行い、表示時だけ外接半径で規格化します。

切断超平面に四面体全体が一致する場合、交差は3次元体積になるため三角形として出力せず胞数を報告します。共有三角形に切断面が一致する場合は所有胞だけが出力します。GPU epsilon は外接半径 × 1e-6、f32。CPU は f64 と半径 × 1e-9 を使います。

## GPU バッファ契約

WGSL を独立したファイルとして読み込んでください。`ViewerModel::positions()` と `gpu_cells()` をロード時に転送します。

|binding|用途|レイアウト|
|---|---|---|
|0|入力頂点|`vec4<f32>[]`|
|1|入力胞|32 bytes / 胞: 頂点番号4個 + 共有面所有ビットマスク + padding 3個|
|2|出力頂点|32 bytes / 頂点: position vec4 + flat normal vec4。最大6頂点 / 胞|
|3|カウンタ|u32 5個: vertexCount, instanceCount=1, firstVertex=0, firstInstance=0, coplanarCount|
|4|uniform|112 bytes: rotation mat4, center vec4, (offset, axis, epsilon, radius), (cellCount, 0, 0, 0)|

dispatch は `ceil(cellCount / 64)`。毎回 binding 3 を `[0,1,0,0,0]` に初期化してから計算し、先頭16 bytesを `drawIndirect` に渡します。出力三角形の順序は不定です。

## 開発・検証

Rust は `stable` の最新版を使います。この初版の検証環境は Rust 1.99.0、wasm-bindgen 0.2.129。

```sh
rustup update stable
cargo fmt --check
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
cargo build --locked --release --target wasm32-unknown-unknown
```

`Cargo.lock` を追跡して再現性を保持します。更新するときは `cargo update` して検証してください。wasm-pack 内蔵の古い Binaryen の自動取得は無効化し、現行 Rust/LLVM の release 最適化を使っています。

添付論文で扱われた四面体メッシュと超平面の交差を基本にした実装です。論文ファイルそのものはリポジトリに含めません。
