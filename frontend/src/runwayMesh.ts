import * as THREE from "three";
import type { RunwayInfo } from "./protocol";

export function createEnvironment(runway: RunwayInfo): THREE.Group {
  const root = new THREE.Group();
  root.name = "environment";

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(18000, 18000),
    new THREE.MeshStandardMaterial({ color: 0x8aa16d, roughness: 0.95, metalness: 0 }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  root.add(ground);

  const grid = new THREE.GridHelper(12000, 80, 0x6f8758, 0x7b9264);
  grid.position.y = 0.02;
  root.add(grid);

  addRunway(root, runway);
  addThresholdChevrons(root, runway);
  addHills(root);
  return root;
}

function addRunway(root: THREE.Group, runway: RunwayInfo): void {
  const heading = THREE.MathUtils.degToRad(runway.heading_deg);
  const group = new THREE.Group();
  // Runway local +Z is along heading in a Y-up frame after we rotate around Y.
  // We build the strip along +X (east) then rotate so +X aligns with heading.
  group.rotation.y = -heading + Math.PI / 2;

  const pavement = new THREE.Mesh(
    new THREE.BoxGeometry(runway.length_m, 0.16, runway.width_m),
    new THREE.MeshStandardMaterial({ color: 0x4a4f57, roughness: 0.85 }),
  );
  pavement.position.set(runway.length_m / 2, 0.08, 0);
  pavement.receiveShadow = true;
  group.add(pavement);

  const shoulder = new THREE.Mesh(
    new THREE.BoxGeometry(runway.length_m + 40, 0.08, runway.width_m + 18),
    new THREE.MeshStandardMaterial({ color: 0x6b6e66, roughness: 0.95 }),
  );
  shoulder.position.set(runway.length_m / 2, 0.03, 0);
  group.add(shoulder);

  const markMat = new THREE.MeshStandardMaterial({ color: 0xf5f5f0, roughness: 0.4 });
  const dashCount = Math.floor(runway.length_m / 40);
  for (let i = 0; i < dashCount; i += 1) {
    const dash = new THREE.Mesh(new THREE.BoxGeometry(18, 0.05, 0.45), markMat);
    dash.position.set(30 + i * 40, 0.18, 0);
    group.add(dash);
  }

  for (let i = -4; i <= 4; i += 1) {
    const bar = new THREE.Mesh(new THREE.BoxGeometry(18, 0.05, 0.7), markMat);
    bar.position.set(22, 0.18, i * 2.6);
    group.add(bar);
  }

  const edgeMat = new THREE.MeshStandardMaterial({ color: 0xfff3c4, emissive: 0x665522, roughness: 0.4 });
  for (let i = 0; i < 24; i += 1) {
    const x = 20 + i * (runway.length_m / 24);
    for (const z of [-runway.width_m / 2 - 0.6, runway.width_m / 2 + 0.6]) {
      const light = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.35, 0.5), edgeMat);
      light.position.set(x, 0.3, z);
      group.add(light);
    }
  }

  root.add(group);
}

function addThresholdChevrons(root: THREE.Group, runway: RunwayInfo): void {
  const mat = new THREE.MeshStandardMaterial({ color: 0xd9c27a, roughness: 0.7 });
  const heading = THREE.MathUtils.degToRad(runway.heading_deg);
  for (let i = 1; i <= 6; i += 1) {
    const chevron = new THREE.Mesh(new THREE.ConeGeometry(6, 14, 3), mat);
    chevron.rotation.x = Math.PI / 2;
    const dist = -40 - i * 28;
    chevron.position.set(Math.sin(heading) * dist, 0.2, -Math.cos(heading) * dist);
    chevron.rotation.z = -heading;
    root.add(chevron);
  }
}

function addHills(root: THREE.Group): void {
  const mat = new THREE.MeshStandardMaterial({ color: 0x6f8a5c, roughness: 1 });
  const spots = [
    [1800, -2200, 90, 280],
    [-2400, 1600, 70, 220],
    [3200, 2600, 110, 340],
    [-3600, -1400, 80, 200],
    [900, 3800, 60, 160],
  ] as const;
  for (const [x, z, r, h] of spots) {
    const hill = new THREE.Mesh(new THREE.SphereGeometry(r, 12, 8), mat);
    hill.position.set(x, h * 0.15 - r * 0.35, z);
    hill.scale.y = h / r;
    hill.receiveShadow = true;
    root.add(hill);
  }
}
