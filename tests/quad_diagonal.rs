use quarttet_core::Mesh4;
use std::collections::HashMap;
fn diagonal(shear: f64, cell: [u32; 4]) -> [[i64; 3]; 2] {
    let mesh = Mesh4 {
        vertices: vec![
            [-2., 0., 0., -1.],
            [2., 0., 0., -1.],
            [shear, -2., 0., 1.],
            [-shear, 2., 0., 1.],
        ],
        cells: vec![cell],
        vertex_normals: Vec::new(),
        cell_normals: Vec::new(),
    };
    let section = mesh.section(3, 0., [0.; 6]).unwrap();
    assert_eq!(section.triangles.len(), 2);
    let mut edges = HashMap::new();
    for tri in section.triangles {
        let keys = tri.map(|p| p.map(|x| (x * 1e6).round() as i64));
        for i in 0..3 {
            let mut edge = [keys[i], keys[(i + 1) % 3]];
            edge.sort();
            *edges.entry(edge).or_insert(0) += 1;
        }
    }
    *edges.iter().find(|(_, count)| **count == 2).unwrap().0
}
#[test]
fn delaunay_quad_and_square_tie_are_independent_of_cell_order() {
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                for d in 0..4 {
                    let cell = [a, b, c, d];
                    if (0..4).any(|i| (i + 1..4).any(|j| cell[i] == cell[j])) {
                        continue;
                    }
                    assert_eq!(
                        diagonal(1., cell),
                        [[-500000, -1000000, 0], [500000, 1000000, 0]]
                    );
                    assert_eq!(
                        diagonal(0., cell),
                        [[-1000000, -1000000, 0], [1000000, 1000000, 0]]
                    );
                }
            }
        }
    }
}
