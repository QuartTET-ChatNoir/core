use quarttet_core::Mesh4;
use std::collections::HashMap;

#[test]
fn tesseract_boundary_is_closed_without_added_vertices() {
    let mesh = Mesh4::sample(0);
    assert_eq!(mesh.vertices.len(), 16);
    assert_eq!(mesh.cells.len(), 48);
    let mut faces = HashMap::new();
    let mut volume = [0.0f64; 8];
    for (cell, normal) in mesh.cells.iter().zip(&mesh.cell_normals) {
        let axis = normal.iter().position(|x| x.abs() > 0.5).unwrap();
        let side = usize::from(normal[axis] > 0.0);
        assert!(
            cell.iter()
                .all(|v| mesh.vertices[*v as usize][axis] == if side == 1 { 1.0 } else { -1.0 })
        );
        let axes: Vec<_> = (0..4).filter(|j| *j != axis).collect();
        let points = cell.map(|v| mesh.vertices[v as usize]);
        let edges: [[f64; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|j| points[i + 1][axes[j]] - points[0][axes[j]])
        });
        let [a, b, c] = edges;
        let determinant = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]);
        volume[axis * 2 + side] += determinant.abs() / 6.0;
        for omitted in 0..4 {
            let mut face: Vec<_> = cell
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != omitted)
                .map(|(_, v)| *v)
                .collect();
            face.sort();
            *faces.entry(face).or_insert(0) += 1;
        }
    }
    assert!(
        faces.values().all(|count| *count == 2),
        "every triangular face must belong to two tetrahedra"
    );
    assert!(volume.iter().all(|value| (value - 8.0).abs() < 1e-10));
}

#[test]
fn axis_sections_have_cube_boundary_and_closed_triangulation() {
    let mesh = Mesh4::sample(0);
    for axis in 0..4 {
        for offset in [-0.731, 0.0, 0.243, 0.91] {
            let section = mesh.section(axis, offset, [0.0; 6]).unwrap();
            let mut edges = HashMap::new();
            let mut vertices = HashMap::new();
            for triangle in &section.triangles {
                assert!(
                    (0..3).any(|j| triangle.iter().all(|p| (p[j] - 1.0).abs() < 1e-9)
                        || triangle.iter().all(|p| (p[j] + 1.0).abs() < 1e-9)),
                    "all triangles must lie on a cube face"
                );
                let keys = triangle.map(|p| p.map(|x| (x * 1e9).round() as i64));
                for p in keys {
                    vertices.insert(p, ());
                }
                for j in 0..3 {
                    let mut edge = [keys[j], keys[(j + 1) % 3]];
                    edge.sort();
                    *edges.entry(edge).or_insert(0) += 1;
                }
            }
            assert!(
                edges.values().all(|count| *count == 2),
                "section must have no holes or T-junctions"
            );
            for mask in 0..8 {
                let corner = std::array::from_fn(|j| {
                    if mask & (1 << j) == 0 {
                        -1_000_000_000
                    } else {
                        1_000_000_000
                    }
                });
                assert!(vertices.contains_key(&corner));
            }
        }
    }
}
