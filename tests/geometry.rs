use quarttet_core::Mesh4;
fn area(mesh: &Mesh4, axis: usize, offset: f64) -> f64 {
    mesh.section(axis, offset, [0.; 6])
        .unwrap()
        .triangles
        .iter()
        .map(|t| {
            let a: [f64; 3] = std::array::from_fn(|j| t[1][j] - t[0][j]);
            let b: [f64; 3] = std::array::from_fn(|j| t[2][j] - t[0][j]);
            let c = [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ];
            c.iter().map(|v| v * v).sum::<f64>().sqrt() * 0.5
        })
        .sum()
}
#[test]
fn tesseract_sections_are_unit_cubes() {
    let m = Mesh4::sample(0);
    assert_eq!(m.cells.len(), 48);
    for axis in 0..4 {
        for offset in [-0.731, 0., 0.243, 0.91] {
            assert!((area(&m, axis, offset) - 24.).abs() < 1e-10);
        }
    }
}
#[test]
fn outside_is_empty() {
    assert!(
        Mesh4::sample(0)
            .section(3, 2., [0.; 6])
            .unwrap()
            .triangles
            .is_empty()
    );
}
#[test]
fn octahedron_area() {
    let m = Mesh4::sample(1);
    assert!((area(&m, 3, 0.) - 4. * 3f64.sqrt()).abs() < 1e-10);
}
#[test]
fn rotation_preserves_radius() {
    let m = Mesh4::sample(0);
    for p in m.transformed([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) {
        assert!((p.iter().map(|x| x * x).sum::<f64>() - 4.).abs() < 1e-10);
    }
}
#[test]
fn te4_round_trip() {
    for kind in 0..3 {
        let m = Mesh4::sample(kind);
        let parsed = Mesh4::from_te4(&m.to_te4(), true).unwrap();
        assert_eq!(parsed.cells, m.cells);
        assert_eq!(parsed.vertices.len(), m.vertices.len());
    }
}
#[test]
fn rejects_corruption() {
    let m = Mesh4::sample(0);
    let mut b = m.to_te4();
    assert!(Mesh4::from_te4(&b[..b.len() - 1], false).is_err());
    b[12..16].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(Mesh4::from_te4(&b, false).is_err());
    let mut b = m.to_te4();
    let start = 12 + 16 * m.vertices.len();
    b[start..start + 4].copy_from_slice(&100u32.to_le_bytes());
    assert!(Mesh4::from_te4(&b, false).is_err());
    let mut b = m.to_te4();
    b[0..4].copy_from_slice(&5u32.to_le_bytes());
    assert!(Mesh4::from_te4(&b, false).is_err());
}
#[test]
fn te4_is_one_based_and_zero_based_is_compatibility_only() {
    let m = Mesh4::sample(1);
    let mut b = m.to_te4();
    assert_eq!(Mesh4::from_te4(&b, true).unwrap().cells, m.cells);
    assert!(Mesh4::from_te4(&b, false).is_err());
    for i in (12 + 16 * m.vertices.len()..b.len()).step_by(4) {
        let value = u32::from_le_bytes(b[i..i + 4].try_into().unwrap());
        b[i..i + 4].copy_from_slice(&(value - 1).to_le_bytes());
    }
    assert_eq!(Mesh4::from_te4(&b, false).unwrap().cells, m.cells);
    assert!(Mesh4::from_te4(&b, true).is_err());
}

#[test]
fn coplanar_cells_are_reported() {
    let s = Mesh4::sample(0).section(3, 1., [0.; 6]).unwrap();
    assert_eq!(s.coplanar_cells, 6);
}
#[test]
fn invalid_parameters_fail() {
    let m = Mesh4::sample(0);
    assert!(m.section(4, 0., [0.; 6]).is_err());
    assert!(m.section(3, f64::NAN, [0.; 6]).is_err());
}

#[test]
fn accepts_more_than_two_million_vertices_and_64_mib() {
    let n = 4_194_305u32;
    let mut bytes = vec![0u8; 12 + n as usize * 16 + 16];
    bytes[..4].copy_from_slice(&1u32.to_le_bytes());
    bytes[4..8].copy_from_slice(&n.to_le_bytes());
    bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
    let start = 12 + n as usize * 16;
    for j in 0..4usize {
        bytes[start + j * 4..start + j * 4 + 4].copy_from_slice(&(j as u32 + 1).to_le_bytes());
    }
    let mesh = Mesh4::from_te4(&bytes, true).unwrap();
    assert_eq!(mesh.vertices.len(), n as usize);
    assert_eq!(mesh.cells, vec![[0, 1, 2, 3]]);
}

#[test]
fn reads_all_te4_normal_variants() {
    let mesh = Mesh4::sample(1);
    for kind in 1..=4u32 {
        let mut bytes = mesh.to_te4();
        bytes[..4].copy_from_slice(&kind.to_le_bytes());
        let normal_count = if matches!(kind, 2 | 4) {
            mesh.vertices.len()
        } else {
            0
        } + if matches!(kind, 3 | 4) {
            mesh.cells.len()
        } else {
            0
        };
        for _ in 0..normal_count {
            for value in [1.0f32, 0.0, 0.0, 0.0] {
                bytes.extend(value.to_le_bytes());
            }
        }
        let parsed = Mesh4::from_te4(&bytes, true).unwrap();
        assert_eq!(parsed.vertices, mesh.vertices);
        assert_eq!(parsed.vertex_normals.len(), mesh.vertices.len());
        assert_eq!(parsed.cell_normals.len(), mesh.cells.len());
        if matches!(kind, 2 | 4) {
            assert!(parsed.vertex_normals.iter().all(|n| *n == [1., 0., 0., 0.]));
        }
        if matches!(kind, 3 | 4) {
            assert!(parsed.cell_normals.iter().all(|n| *n == [1., 0., 0., 0.]));
        }
        assert_eq!(parsed.cells, mesh.cells);
        assert_eq!(
            parsed.section(3, 0.0, [0.; 6]).unwrap().triangles.len(),
            mesh.section(3, 0.0, [0.; 6]).unwrap().triangles.len()
        );
        assert!(Mesh4::from_te4(&bytes[..bytes.len() - 1], false).is_err());
        if normal_count > 0 {
            let end = bytes.len();
            bytes[end - 4..].copy_from_slice(&f32::NAN.to_le_bytes());
            assert!(Mesh4::from_te4(&bytes, true).is_err());
        }
    }
}

#[test]
fn generates_volume_weighted_4d_vertex_normals() {
    let mut mesh = Mesh4 {
        vertices: vec![
            [0., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [1., 0., 0., 0.],
            [0., 0., 0., 2.],
        ],
        cells: vec![[0, 1, 2, 3], [0, 4, 2, 5]],
        vertex_normals: Vec::new(),
        cell_normals: Vec::new(),
    };
    mesh.generate_missing_normals().unwrap();
    assert_eq!(mesh.cell_normals, vec![[1., 0., 0., 0.], [0., -1., 0., 0.]]);
    let expected = [1. / 5f32.sqrt(), -2. / 5f32.sqrt(), 0., 0.];
    for (actual, expected) in mesh.vertex_normals[0].iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-6);
    }
    let original = mesh.vertex_normals.clone();
    mesh.generate_missing_normals().unwrap();
    assert_eq!(mesh.vertex_normals, original);
}
