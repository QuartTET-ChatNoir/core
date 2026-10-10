struct Params {
  rotation: mat4x4<f32>,
  center: vec4<f32>,
  plane: vec4<f32>, // offset, axis, relative epsilon, radius
  counts: vec4<u32>, // cell count, normal flags (vertex=1, cell=2), reserved
}
struct Cell { indices: vec4<u32>, ownership: vec4<u32> }
struct Vertex { position: vec4<f32>, normal: vec4<f32> }
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> cells: array<Cell>;
@group(0) @binding(2) var<storage, read_write> output: array<Vertex>;
@group(0) @binding(3) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read> vertex_normals: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> cell_normals: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> original_ids: array<u32>;
@group(0) @binding(8) var<storage, read> candidates: array<u32>;
fn key_less(a: vec2<u32>, b: vec2<u32>) -> bool {
  return a.x < b.x || (a.x == b.x && a.y < b.y);
}
fn diagonal_key(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {
  if key_less(b,a) { return vec4<u32>(b,a); }
  return vec4<u32>(a,b);
}
fn diagonal_less(a: vec4<u32>, b: vec4<u32>) -> bool {
  for (var i=0u; i<4u; i++) {
    if a[i] != b[i] { return a[i] < b[i]; }
  }
  return false;
}
fn interior_angle(a: vec3<f32>, center: vec3<f32>, b: vec3<f32>) -> f32 {
  let u=a-center; let v=b-center;
  return atan2(length(cross(u,v)),dot(u,v));
}
fn safe_normal(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
  if dot(value,value) > 1e-20 { return normalize(value); }
  return fallback;
}
fn project(p: vec4<f32>, axis: u32) -> vec3<f32> {
  switch axis {
    case 0u: { return p.yzw; }
    case 1u: { return p.xzw; }
    case 2u: { return p.xyw; }
    default: { return p.xyz; }
  }
}
// Project the unique triangular faces of the tetrahedral mesh, up to 12 vertices/cell.
fn project_cell(cell_id: u32, axis: u32) {
  let cell=cells[cell_id];
  var points: array<vec3<f32>,4>;
  for(var i=0u;i<4u;i++) {
    let p=params.rotation*(positions[cell.indices[i]]-params.center);
    var q=project(p,axis)/params.plane.w;
    if params.counts.z == 2u { q*=params.plane.x/(params.plane.x-p[axis]); }
    points[i]=q;
  }
  for(var omitted=0u;omitted<4u;omitted++) {
    if (cell.ownership.x & (1u<<omitted)) == 0u { continue; }
    var face: array<u32,3>; var at=0u;
    for(var i=0u;i<4u;i++) { if i!=omitted { face[at]=i; at++; } }
    let a=points[face[0]];let b=points[face[1]];let c=points[face[2]];
    let cross_product=cross(b-a,c-a);
    if length(cross_product)<=1e-12 { continue; }
    let normal=vec4<f32>(normalize(cross_product),0.0);
    let start=atomicAdd(&counters[0],3u);
    output[start]=Vertex(vec4<f32>(a,1.0),normal);
    output[start+1u]=Vertex(vec4<f32>(b,1.0),normal);
    output[start+2u]=Vertex(vec4<f32>(c,1.0),normal);
  }
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  var cell_id=id.x;
  let use_bvh=(params.counts.y & 4u)!=0u && params.counts.z==0u;
  if use_bvh {
    if id.x >= atomicLoad(&counters[5]) { return; }
    cell_id=candidates[id.x];
  } else if id.x >= params.counts.x { return; }
  if params.counts.z != 0u { project_cell(cell_id,u32(params.plane.y)); return; }
  let cell = cells[cell_id].indices;
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
      if abs(d[i]) > eps && (cells[cell_id].ownership.x & (1u << i)) == 0u { return; }
    }
  }
  var polygon: array<vec3<f32>,4>;
  var shading: array<vec3<f32>,4>;
  var keys: array<vec2<u32>,4>;
  var projected_normals: array<vec3<f32>,4>;
  if (params.counts.y & 1u) != 0u {
    for (var i=0u; i<4u; i++) {
      projected_normals[i] = project(params.rotation * vertex_normals[cell[i]],axis);
    }
  }
  var n = 0u;
  for (var i=0u; i<4u; i++) {
    if abs(d[i]) <= eps { polygon[n] = project(p[i],axis); shading[n] = projected_normals[i];
      keys[n] = vec2<u32>(original_ids[cell[i]]); n++; }
  }
  for (var a=0u; a<4u; a++) {
    for (var b=a+1u; b<4u; b++) {
      if (d[a]>eps && d[b]< -eps) || (d[b]>eps && d[a]< -eps) {
        let t = d[a] / (d[a]-d[b]);
        polygon[n] = project(mix(p[a],p[b],t),axis);
        shading[n] = mix(projected_normals[a],projected_normals[b],t);
        let ia=original_ids[cell[a]]; let ib=original_ids[cell[b]];
        keys[n]=vec2<u32>(min(ia,ib),max(ia,ib)); n++;
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
  if (params.counts.y & 2u) != 0u {
    normal = safe_normal(project(params.rotation * cell_normals[cell_id],axis),normal);
  }
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
        let shade=shading[i]; shading[i]=shading[j]; shading[j]=shade;
        let key=keys[i]; keys[i]=keys[j]; keys[j]=key;
      }
    }
  }
  if n == 4u {
    let sum=interior_angle(polygon[0],polygon[1],polygon[2])
      + interior_angle(polygon[2],polygon[3],polygon[0]);
    var flip=sum > 3.141592653589793;
    if abs(sum-3.141592653589793) <= 1e-6 {
      flip=diagonal_less(diagonal_key(keys[1],keys[3]),diagonal_key(keys[0],keys[2]));
    }
    if flip {
      let first=polygon[0]; let first_shade=shading[0];
      for (var i=0u; i<3u; i++) { polygon[i]=polygon[i+1u]; shading[i]=shading[i+1u]; }
      polygon[3]=first; shading[3]=first_shade;
    }
  }
  // Maximum 2 triangles per tetrahedron. Compact on the GPU, then indirect draw.
  for (var i=1u; i+1u<n; i++) {
    let a=polygon[0]; let b=polygon[i]; let c=polygon[i+1u];
    let cross_product=cross(b-a,c-a);
    if length(cross_product)<=eps*eps { continue; }
    let start=atomicAdd(&counters[0],3u);
    let face_normal=normal;
    var na=face_normal; var nb=face_normal; var nc=face_normal;
    if (params.counts.y & 1u) != 0u {
      na=safe_normal(shading[0],face_normal);
      nb=safe_normal(shading[i],face_normal);
      nc=safe_normal(shading[i+1u],face_normal);
    }
    output[start] = Vertex(vec4<f32>(a/params.plane.w,1.0),vec4<f32>(na,0.0));
    output[start+1u] = Vertex(vec4<f32>(b/params.plane.w,1.0),vec4<f32>(nb,0.0));
    output[start+2u] = Vertex(vec4<f32>(c/params.plane.w,1.0),vec4<f32>(nc,0.0));
  }
}
