struct Params {
  rotation: mat4x4<f32>,
  center: vec4<f32>,
  plane: vec4<f32>, // offset, axis, relative epsilon, radius
  counts: vec4<u32>,
}
struct Cell { indices: vec4<u32>, ownership: vec4<u32> }
struct Vertex { position: vec4<f32>, normal: vec4<f32> }
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> cells: array<Cell>;
@group(0) @binding(2) var<storage, read_write> output: array<Vertex>;
@group(0) @binding(3) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: Params;
fn project(p: vec4<f32>, axis: u32) -> vec3<f32> {
  switch axis {
    case 0u: { return p.yzw; }
    case 1u: { return p.xzw; }
    case 2u: { return p.xyw; }
    default: { return p.xyz; }
  }
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if id.x >= params.counts.x { return; }
  let cell = cells[id.x].indices;
  let axis = u32(params.plane.y);
  let eps = params.plane.z;
  var p: array<vec4<f32>,4>;
  var d: array<f32,4>;
  var on = 0u;
  var positive = 0u;
  var negative = 0u;
  for (var i=0u; i<4u; i++) {
    p[i] = params.rotation * (positions[cell[i]] - params.center);
    d[i] = p[i][axis] - params.plane.x;
    if abs(d[i]) <= eps { on++; }
    else if d[i] > 0.0 { positive++; }
    else { negative++; }
  }
  if on == 4u { atomicAdd(&counters[4], 1u); return; }
  if positive == 4u || negative == 4u { return; }
  if on == 3u {
    for (var i=0u; i<4u; i++) {
      if abs(d[i]) > eps && (cells[id.x].ownership.x & (1u << i)) == 0u { return; }
    }
  }
  var polygon: array<vec3<f32>,4>;
  var n = 0u;
  for (var i=0u; i<4u; i++) {
    if abs(d[i]) <= eps { polygon[n] = project(p[i],axis); n++; }
  }
  for (var a=0u; a<4u; a++) {
    for (var b=a+1u; b<4u; b++) {
      if (d[a]>eps && d[b]< -eps) || (d[b]>eps && d[a]< -eps) {
        let t = d[a] / (d[a]-d[b]);
        polygon[n] = project(mix(p[a],p[b],t),axis); n++;
      }
    }
  }
  if n < 3u { return; }
  var center = vec3<f32>(0.0);
  for (var i=0u; i<n; i++) { center += polygon[i]; }
  center /= f32(n);
  let u = polygon[0] - center;
  var normal = vec3<f32>(0.0);
  for (var i=1u; i<n; i++) {
    normal = cross(u,polygon[i]-center);
    if length(normal) > eps*eps { break; }
  }
  if length(normal) <= eps*eps { return; }
  normal = normalize(normal);
  let v = cross(normal,u);
  var angles: array<f32,4>;
  for (var i=0u; i<n; i++) {
    let q=polygon[i]-center;
    angles[i]=atan2(dot(q,v),dot(q,u));
  }
  for (var i=0u; i<n; i++) {
    for (var j=i+1u; j<n; j++) {
      if angles[j]<angles[i] {
        let angle=angles[i]; angles[i]=angles[j]; angles[j]=angle;
        let point=polygon[i]; polygon[i]=polygon[j]; polygon[j]=point;
      }
    }
  }
  // Maximum 2 triangles per tetrahedron. Compact on the GPU, then indirect draw.
  for (var i=1u; i+1u<n; i++) {
    let a=polygon[0]; let b=polygon[i]; let c=polygon[i+1u];
    let cross_product=cross(b-a,c-a);
    if length(cross_product)<=eps*eps { continue; }
    let start=atomicAdd(&counters[0],3u);
    let face_normal=vec4<f32>(normalize(cross_product),0.0);
    output[start] = Vertex(vec4<f32>(a/params.plane.w,1.0),face_normal);
    output[start+1u] = Vertex(vec4<f32>(b/params.plane.w,1.0),face_normal);
    output[start+2u] = Vertex(vec4<f32>(c/params.plane.w,1.0),face_normal);
  }
}
