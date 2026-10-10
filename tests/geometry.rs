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
        let parsed = Mesh4::from_te4(&m.to_te4(), false).unwrap();
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
    b[0..4].copy_from_slice(&2u32.to_le_bytes());
    assert!(Mesh4::from_te4(&b, false).is_err());
}
#[test]
fn one_based_is_explicit() {
    let m = Mesh4::sample(1);
    let mut b = m.to_te4();
    for i in (12 + 16 * m.vertices.len()..b.len()).step_by(4) {
        let value = u32::from_le_bytes(b[i..i + 4].try_into().unwrap());
        b[i..i + 4].copy_from_slice(&(value + 1).to_le_bytes());
    }
    assert_eq!(Mesh4::from_te4(&b, true).unwrap().cells, m.cells);
    assert!(Mesh4::from_te4(&b, false).is_err());
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
