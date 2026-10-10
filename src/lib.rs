//! Browser-independent 4D tetrahedral mesh operations.
use std::collections::{HashMap, HashSet};
use wasm_bindgen::prelude::*;

pub type Point4 = [f64; 4];
#[derive(Clone, Debug)]
pub struct Mesh4 {
    pub vertices: Vec<Point4>,
    pub cells: Vec<[u32; 4]>,
    pub vertex_normals: Vec<[f32; 4]>,
    pub cell_normals: Vec<[f32; 4]>,
}
#[derive(Debug, Default)]
pub struct Section {
    pub triangles: Vec<[[f64; 3]; 3]>,
    pub coplanar_cells: usize,
}

impl Mesh4 {
    /// TE4 types 1–4: header, positions, cells, optional vertex then cell normals.
    /// Index base is explicit because historical exporters may use one-based indices.
    pub fn from_te4(bytes: &[u8], one_based: bool) -> Result<Self, String> {
        if bytes.len() < 12 {
            return Err("TE4 header is truncated".into());
        }
        let word = |i| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        let kind = word(0);
        if !(1..=4).contains(&kind) {
            return Err(format!("Unknown TE4 type {kind}; expected 1–4"));
        }
        let n = word(4) as usize;
        let m = word(8) as usize;
        let vertex_normals = matches!(kind, 2 | 4);
        let cell_normals = matches!(kind, 3 | 4);
        let expected = n
            .checked_add(m)
            .and_then(|x| x.checked_add(if vertex_normals { n } else { 0 }))
            .and_then(|x| x.checked_add(if cell_normals { m } else { 0 }))
            .and_then(|x| x.checked_mul(16))
            .and_then(|x| x.checked_add(12))
            .ok_or("TE4 size overflow")?;
        if expected != bytes.len() {
            return Err(format!(
                "TE4 length mismatch: expected {expected}, got {}",
                bytes.len()
            ));
        }
        if n == 0 || m == 0 {
            return Err("TE4 must contain vertices and cells".into());
        }
        let mut vertices = Vec::new();
        vertices
            .try_reserve_exact(n)
            .map_err(|_| "Insufficient memory for TE4 vertices")?;
        for i in 0..n {
            let p = std::array::from_fn(|j| f32::from_bits(word(12 + i * 16 + j * 4)) as f64);
            if !p.iter().all(|x| x.is_finite()) {
                return Err(format!("Vertex {i} contains a non-finite coordinate"));
            }
            vertices.push(p);
        }
        let mut cells = Vec::new();
        cells
            .try_reserve_exact(m)
            .map_err(|_| "Insufficient memory for TE4 cells")?;
        for i in 0..m {
            let mut cell = [0; 4];
            for (j, index) in cell.iter_mut().enumerate() {
                *index = word(12 + n * 16 + i * 16 + j * 4)
                    .checked_sub(u32::from(one_based))
                    .ok_or("Zero index in one-based TE4")?;
                if *index as usize >= n {
                    return Err(format!("Cell {i} references a missing vertex"));
                }
            }
            for a in 0..4 {
                for b in a + 1..4 {
                    if cell[a] == cell[b] {
                        return Err(format!("Cell {i} repeats a vertex"));
                    }
                }
            }
            cells.push(cell);
        }
        let mut at = 12 + (n + m) * 16;
        let mut read_normals = |count: usize| -> Result<Vec<[f32; 4]>, String> {
            let mut normals = Vec::new();
            normals
                .try_reserve_exact(count)
                .map_err(|_| "Insufficient memory for TE4 normals")?;
            for _ in 0..count {
                let values: [f64; 4] =
                    std::array::from_fn(|j| f32::from_bits(word(at + j * 4)) as f64);
                at += 16;
                if !values.iter().all(|v| v.is_finite()) {
                    return Err("TE4 contains a non-finite normal".into());
                }
                let length = values.iter().map(|v| v * v).sum::<f64>().sqrt();
                // Normalize in f64 to handle even very large or tiny f32 normals.
                normals.push(values.map(|v| {
                    if length > 0.0 {
                        (v / length) as f32
                    } else {
                        0.0
                    }
                }));
            }
            Ok(normals)
        };
        let vertex_normals = read_normals(if vertex_normals { n } else { 0 })?;
        let cell_normals = read_normals(if cell_normals { m } else { 0 })?;
        let mut mesh = Self {
            vertices,
            cells,
            vertex_normals,
            cell_normals,
        };
        mesh.generate_missing_normals()?;
        Ok(mesh)
    }

    /// Cache missing 4D normals once. Authored normals take precedence.
    /// Cell index order supplies orientation; adjacent cells should be consistently wound.
    pub fn generate_missing_normals(&mut self) -> Result<(), String> {
        let missing_cells = self.cell_normals.is_empty();
        let missing_vertices = self.vertex_normals.is_empty();
        if !missing_cells && !missing_vertices {
            return Ok(());
        }
        if missing_cells {
            self.cell_normals
                .try_reserve_exact(self.cells.len())
                .map_err(|_| "Insufficient memory for generated cell normals")?;
        }
        let mut sums: Vec<[f64; 4]> = Vec::new();
        if missing_vertices {
            sums.try_reserve_exact(self.vertices.len())
                .map_err(|_| "Insufficient memory for vertex normal accumulation")?;
            sums.resize(self.vertices.len(), [0.; 4]);
        }
        for (id, cell) in self.cells.iter().enumerate() {
            let mut geometric = tetra_normal(cell.map(|v| self.vertices[v as usize]));
            if missing_cells && !missing_vertices {
                let direction: Point4 = std::array::from_fn(|j| {
                    cell.iter()
                        .map(|v| self.vertex_normals[*v as usize][j] as f64)
                        .sum()
                });
                if (0..4).map(|j| geometric[j] * direction[j]).sum::<f64>() < 0. {
                    geometric = geometric.map(|v| -v);
                }
            }
            // Magnitude is six times the tetrahedron's intrinsic 3D volume.
            let weight = geometric.iter().map(|v| v * v).sum::<f64>().sqrt();
            if missing_cells {
                self.cell_normals
                    .push(geometric.map(|v| if weight > 0. { (v / weight) as f32 } else { 0. }));
            }
            if missing_vertices {
                for vertex in cell {
                    for (axis, value) in self.cell_normals[id].iter().enumerate() {
                        sums[*vertex as usize][axis] += *value as f64 * weight;
                    }
                }
            }
        }
        if missing_vertices {
            self.vertex_normals
                .try_reserve_exact(self.vertices.len())
                .map_err(|_| "Insufficient memory for generated vertex normals")?;
            for sum in sums {
                let length = sum.iter().map(|v| v * v).sum::<f64>().sqrt();
                self.vertex_normals
                    .push(sum.map(|v| if length > 0. { (v / length) as f32 } else { 0. }));
            }
        }
        Ok(())
    }

    pub fn to_te4(&self) -> Vec<u8> {
        let mut b = Vec::new();
        for v in [1, self.vertices.len() as u32, self.cells.len() as u32] {
            b.extend(v.to_le_bytes());
        }
        for p in &self.vertices {
            for v in p {
                b.extend((*v as f32).to_le_bytes());
            }
        }
        for c in &self.cells {
            for v in c {
                b.extend((v + 1).to_le_bytes());
            }
        }
        b
    }
    pub fn center_radius(&self) -> (Point4, f64) {
        let mut low = [f64::INFINITY; 4];
        let mut high = [f64::NEG_INFINITY; 4];
        for p in &self.vertices {
            for j in 0..4 {
                low[j] = low[j].min(p[j]);
                high[j] = high[j].max(p[j]);
            }
        }
        let center = std::array::from_fn(|j| (low[j] + high[j]) * 0.5);
        let radius = self
            .vertices
            .iter()
            .map(|p| {
                (0..4)
                    .map(|j| (p[j] - center[j]).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .fold(0.0, f64::max);
        (center, radius)
    }
    /// Rotations in XY, XZ, XW, YZ, YW, ZW order, about the bounding-box center.
    pub fn transformed(&self, angles: [f64; 6]) -> Vec<Point4> {
        let (center, _) = self.center_radius();
        self.vertices
            .iter()
            .map(|p| {
                let mut q: Point4 = std::array::from_fn(|j| p[j] - center[j]);
                for ((a, b), theta) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
                    .into_iter()
                    .zip(angles)
                {
                    let (s, c) = theta.sin_cos();
                    let x = q[a];
                    let y = q[b];
                    q[a] = c * x - s * y;
                    q[b] = s * x + c * y;
                }
                q
            })
            .collect()
    }
    /// Intersect tetrahedral boundary cells with an axis-aligned 3D hyperplane.
    /// Coplanar tetrahedra are volumes, not triangles; they are counted and skipped.
    pub fn section(&self, axis: usize, offset: f64, angles: [f64; 6]) -> Result<Section, String> {
        if axis > 3 || !offset.is_finite() || !angles.iter().all(|x| x.is_finite()) {
            return Err("Invalid section parameters".into());
        }
        let points = self.transformed(angles);
        let (_, radius) = self.center_radius();
        let eps = (radius * 1e-9).max(f64::MIN_POSITIVE);
        let axes: Vec<_> = (0..4).filter(|j| *j != axis).collect();
        let mut result = Section::default();
        let mut seen = HashSet::new();
        for cell in &self.cells {
            let d: [f64; 4] = std::array::from_fn(|j| points[cell[j] as usize][axis] - offset);
            if d.iter().all(|x| x.abs() <= eps) {
                result.coplanar_cells += 1;
                continue;
            }
            if d.iter().all(|x| *x > eps) || d.iter().all(|x| *x < -eps) {
                continue;
            }
            // Keys preserve exact vertex/edge identity across neighboring cells.
            let mut polygon: Vec<([u32; 2], [f64; 3])> = Vec::with_capacity(4);
            for j in 0..4 {
                if d[j].abs() <= eps {
                    let p = points[cell[j] as usize];
                    polygon.push(([cell[j], cell[j]], std::array::from_fn(|k| p[axes[k]])));
                }
            }
            for a in 0..4 {
                for b in a + 1..4 {
                    if (d[a] > eps && d[b] < -eps) || (d[b] > eps && d[a] < -eps) {
                        let t = d[a] / (d[a] - d[b]);
                        let p = points[cell[a] as usize];
                        let q = points[cell[b] as usize];
                        polygon.push((
                            [cell[a].min(cell[b]), cell[a].max(cell[b])],
                            std::array::from_fn(|k| p[axes[k]] + t * (q[axes[k]] - p[axes[k]])),
                        ));
                    }
                }
            }
            if polygon.len() < 3 {
                continue;
            }
            let centroid: [f64; 3] = std::array::from_fn(|k| {
                polygon.iter().map(|x| x.1[k]).sum::<f64>() / polygon.len() as f64
            });
            let u = sub(polygon[0].1, centroid);
            let mut normal = [0.0; 3];
            for p in polygon.iter().skip(1) {
                normal = cross(u, sub(p.1, centroid));
                if norm(normal) > eps * eps {
                    break;
                }
            }
            let length = norm(normal);
            if length <= eps * eps {
                continue;
            }
            normal = normal.map(|x| x / length);
            let v = cross(normal, u);
            polygon.sort_by(|a, b| {
                let a = sub(a.1, centroid);
                let b = sub(b.1, centroid);
                dot(a, v)
                    .atan2(dot(a, u))
                    .total_cmp(&dot(b, v).atan2(dot(b, u)))
            });
            if polygon.len() == 4 {
                let angle = |at: usize| {
                    let a = sub(polygon[(at + 3) % 4].1, polygon[at].1);
                    let b = sub(polygon[(at + 1) % 4].1, polygon[at].1);
                    norm(cross(a, b)).atan2(dot(a, b))
                };
                let sum = angle(1) + angle(3);
                let diagonal = |a: usize, b: usize| {
                    let mut key = [polygon[a].0, polygon[b].0];
                    key.sort();
                    key
                };
                // Same angular tolerance as f32 GPU, and an index-based rule
                // for cocircular quads independent of polygon starting corner.
                let flip = if (sum - std::f64::consts::PI).abs() <= 1e-6 {
                    diagonal(1, 3) < diagonal(0, 2)
                } else {
                    sum > std::f64::consts::PI
                };
                if flip {
                    polygon.rotate_left(1);
                }
            }
            for j in 1..polygon.len() - 1 {
                let tri = [polygon[0], polygon[j], polygon[j + 1]];
                if norm(cross(sub(tri[1].1, tri[0].1), sub(tri[2].1, tri[0].1))) <= eps * eps {
                    continue;
                }
                let mut key = tri.map(|p| p.0);
                key.sort();
                if seen.insert(key) {
                    result.triangles.push(tri.map(|p| p.1));
                }
            }
        }
        Ok(result)
    }
    /// Pack cells for the GPU with a bitmask of canonical shared-face owners.
    /// The lowest cell index owns each triangular face, so sections through
    /// three shared vertices are emitted once even on an exact slice plane.
    pub fn gpu_cells(&self) -> Vec<u32> {
        let mut owners = HashMap::new();
        let mut packed = Vec::with_capacity(self.cells.len() * 8);
        for (id, cell) in self.cells.iter().enumerate() {
            let mut mask = 0u32;
            for omitted in 0..4 {
                let mut face = [0u32; 3];
                let mut at = 0;
                for (j, v) in cell.iter().enumerate() {
                    if j != omitted {
                        face[at] = *v;
                        at += 1;
                    }
                }
                face.sort();
                if *owners.entry(face).or_insert(id) == id {
                    mask |= 1 << omitted;
                }
            }
            packed.extend(cell);
            packed.extend([mask, 0, 0, 0]);
        }
        packed
    }

    fn convex_sample(vertices: Vec<Point4>, mut cells: Vec<[u32; 4]>) -> Self {
        for cell in &mut cells {
            let points = cell.map(|v| vertices[v as usize]);
            let normal = tetra_normal(points);
            let center: Point4 =
                std::array::from_fn(|j| points.iter().map(|p| p[j]).sum::<f64>() / 4.);
            if (0..4).map(|j| normal[j] * center[j]).sum::<f64>() < 0. {
                cell.swap(0, 1);
            }
        }
        let mut mesh = Self {
            vertices,
            cells,
            vertex_normals: Vec::new(),
            cell_normals: Vec::new(),
        };
        mesh.generate_missing_normals()
            .expect("Small built-in sample normals");
        mesh
    }
    pub fn sample(kind: u32) -> Self {
        if kind == 1 {
            // Regular 16-cell boundary.
            let vertices = (0..4)
                .flat_map(|a| {
                    [-1.0, 1.0].map(move |s| std::array::from_fn(|j| if j == a { s } else { 0.0 }))
                })
                .collect();
            let cells = (0..16)
                .map(|mask| std::array::from_fn(|a| 2 * a as u32 + ((mask >> a) & 1)))
                .collect();
            return Self::convex_sample(vertices, cells);
        }
        if kind == 2 {
            let vertices = vec![
                [1., 1., 1., -1.0 / 5f64.sqrt()],
                [1., -1., -1., -1.0 / 5f64.sqrt()],
                [-1., 1., -1., -1.0 / 5f64.sqrt()],
                [-1., -1., 1., -1.0 / 5f64.sqrt()],
                [0., 0., 0., 4.0 / 5f64.sqrt()],
            ];
            let cells = (0..5)
                .map(|omit| {
                    let c: Vec<_> = (0..5).filter(|i| *i != omit).collect();
                    [c[0], c[1], c[2], c[3]]
                })
                .collect();
            return Self::convex_sample(vertices, cells);
        }
        let vertices = (0..16)
            .map(|mask| std::array::from_fn(|j| if mask & (1 << j) == 0 { -1. } else { 1. }))
            .collect();
        let mut cells = Vec::new();
        for axis in 0..4 {
            for side in 0..2 {
                let free: Vec<_> = (0..4).filter(|j| *j != axis).collect();
                for order in [
                    [0, 1, 2],
                    [0, 2, 1],
                    [1, 0, 2],
                    [1, 2, 0],
                    [2, 0, 1],
                    [2, 1, 0],
                ] {
                    let mut mask = side << axis;
                    let mut c = [mask as u32; 4];
                    for j in 0..3 {
                        mask |= 1 << free[order[j]];
                        c[j + 1] = mask as u32;
                    }
                    cells.push(c);
                }
            }
        }
        Self::convex_sample(vertices, cells)
    }
}
fn tetra_normal(points: [Point4; 4]) -> Point4 {
    let edges: [Point4; 3] =
        std::array::from_fn(|i| std::array::from_fn(|j| points[i + 1][j] - points[0][j]));
    std::array::from_fn(|omitted| {
        let mut axes = [0; 3];
        let mut at = 0;
        for j in 0..4 {
            if j != omitted {
                axes[at] = j;
                at += 1;
            }
        }
        let a = edges[0];
        let b = edges[1];
        let c = edges[2];
        let [x, y, z] = [axes[0], axes[1], axes[2]];
        let det = a[x] * (b[y] * c[z] - b[z] * c[y]) - a[y] * (b[x] * c[z] - b[z] * c[x])
            + a[z] * (b[x] * c[y] - b[y] * c[x]);
        if omitted % 2 == 0 { det } else { -det }
    })
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|j| a[j] - b[j])
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|j| a[j] * b[j]).sum()
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

#[wasm_bindgen]
pub struct ViewerModel {
    mesh: Mesh4,
    coplanar: usize,
}
#[wasm_bindgen]
impl ViewerModel {
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8], one_based: bool) -> Result<ViewerModel, JsValue> {
        Ok(Self {
            mesh: Mesh4::from_te4(bytes, one_based).map_err(|e| JsValue::from_str(&e))?,
            coplanar: 0,
        })
    }
    pub fn sample(kind: u32) -> ViewerModel {
        Self {
            mesh: Mesh4::sample(kind),
            coplanar: 0,
        }
    }
    pub fn vertex_count(&self) -> usize {
        self.mesh.vertices.len()
    }
    pub fn cell_count(&self) -> usize {
        self.mesh.cells.len()
    }
    pub fn coplanar_count(&self) -> usize {
        self.coplanar
    }
    pub fn radius(&self) -> f64 {
        self.mesh.center_radius().1
    }
    pub fn positions(&self) -> Vec<f32> {
        self.mesh
            .vertices
            .iter()
            .flatten()
            .map(|x| *x as f32)
            .collect()
    }
    pub fn indices(&self) -> Vec<u32> {
        self.mesh.cells.iter().flatten().copied().collect()
    }
    pub fn gpu_cells(&self) -> Vec<u32> {
        self.mesh.gpu_cells()
    }
    pub fn vertex_normals(&self) -> Vec<f32> {
        self.mesh.vertex_normals.iter().flatten().copied().collect()
    }
    pub fn cell_normals(&self) -> Vec<f32> {
        self.mesh.cell_normals.iter().flatten().copied().collect()
    }
    pub fn center(&self) -> Vec<f64> {
        self.mesh.center_radius().0.to_vec()
    }
    pub fn export_te4(&self) -> Vec<u8> {
        self.mesh.to_te4()
    }
    pub fn bounds(&self, angles: &[f64]) -> Result<Vec<f64>, JsValue> {
        let angles: [f64; 6] = angles
            .try_into()
            .map_err(|_| JsValue::from_str("Expected six angles"))?;
        if !angles.iter().all(|x: &f64| x.is_finite()) {
            return Err(JsValue::from_str("Invalid angles"));
        }
        let points = self.mesh.transformed(angles);
        let mut bounds = vec![f64::INFINITY; 8];
        for j in 0..4 {
            bounds[2 * j + 1] = f64::NEG_INFINITY;
        }
        for p in points {
            for j in 0..4 {
                bounds[2 * j] = bounds[2 * j].min(p[j]);
                bounds[2 * j + 1] = bounds[2 * j + 1].max(p[j]);
            }
        }
        Ok(bounds)
    }
    /// Interleaved XYZ + flat normal, in radius-normalized coordinates for display.
    pub fn slice(&mut self, axis: usize, offset: f64, angles: &[f64]) -> Result<Vec<f32>, JsValue> {
        let angles: [f64; 6] = angles
            .try_into()
            .map_err(|_| JsValue::from_str("Expected six angles"))?;
        let section = self
            .mesh
            .section(axis, offset, angles)
            .map_err(|e| JsValue::from_str(&e))?;
        self.coplanar = section.coplanar_cells;
        let radius = self.radius().max(f64::MIN_POSITIVE);
        let mut data = Vec::with_capacity(section.triangles.len() * 18);
        for t in section.triangles {
            let n = cross(sub(t[1], t[0]), sub(t[2], t[0]));
            let length = norm(n);
            let n = n.map(|x| (x / length) as f32);
            for p in t {
                data.extend(p.map(|x| (x / radius) as f32));
                data.extend(n);
            }
        }
        Ok(data)
    }
}
