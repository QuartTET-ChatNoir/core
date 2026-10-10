struct Params { rotation: mat4x4<f32>, center: vec4<f32>, plane: vec4<f32>, counts: vec4<u32> }
struct Node { low: vec4<f32>, high: vec4<f32>, info: vec4<u32> }
@group(0) @binding(0) var<storage,read> nodes: array<Node>;
@group(0) @binding(1) var<storage,read_write> candidates: array<u32>;
@group(0) @binding(2) var<storage,read_write> counters: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage,read_write> dispatch_args: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if id.x >= params.counts.w { return; }
  let axis=u32(params.plane.y);
  // Transform the plane back to model coordinates; bounds never need rotation.
  let n=vec4<f32>(params.rotation[0][axis],params.rotation[1][axis],params.rotation[2][axis],params.rotation[3][axis]);
  var at=nodes[id.x].info.w;
  let end=nodes[at].info.x;
  loop {
    if at >= end { break; }
    let node=nodes[at];
    let center=node.low*0.5+node.high*0.5;
    let extent=node.high*0.5-node.low*0.5;
    let distance=dot(n,center-params.center)-params.plane.x;
    let radius=dot(abs(n),extent);
    if abs(distance) > radius+params.plane.z*4.0 { at=node.info.x; continue; }
    if node.info.z > 0u {
      let start=atomicAdd(&counters[5],node.info.z);
      for(var i=0u;i<node.info.z;i++) { candidates[start+i]=node.info.y+i; }
      at=node.info.x;
    } else { at++; }
  }
}
@compute @workgroup_size(1)
fn prepare() {
  dispatch_args[0]=(atomicLoad(&counters[5])+63u)/64u;
  dispatch_args[1]=1u; dispatch_args[2]=1u;
}
