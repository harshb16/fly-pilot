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
  group.rotation.y = -heading + Math.PI / 2;

  const approachLength = 4200;
  const corridor = new THREE.Mesh(
    new THREE.BoxGeometry(approachLength + runway.length_m, 0.06, 70),
    new THREE.MeshStandardMaterial({ color: 0xc4b48a, roughness: 1 }),
  );
  corridor.position.set((runway.length_m - approachLength) / 2, 0.03, 0);
  corridor.receiveShadow = true;
  group.add(corridor);

  const pavement = new THREE.Mesh(
    new THREE.BoxGeometry(runway.length_m, 0.22, runway.width_m),
    new THREE.MeshStandardMaterial({ color: 0x3f4450, roughness: 0.82 }),
  );
  pavement.position.set(runway.length_m / 2, 0.12, 0);
  pavement.receiveShadow = true;
  group.add(pavement);

  const shoulder = new THREE.Mesh(
    new THREE.BoxGeometry(runway.length_m + 40, 0.1, runway.width_m + 22),
    new THREE.MeshStandardMaterial({ color: 0x5c5f58, roughness: 0.95 }),
  );
  shoulder.position.set(runway.length_m / 2, 0.05, 0);
  group.add(shoulder);

  const markMat = new THREE.MeshStandardMaterial({ color: 0xf5f5f0, roughness: 0.4 });
  const dashCount = Math.floor(runway.length_m / 40);
  for (let i = 0; i < dashCount; i += 1) {
    const dash = new THREE.Mesh(new THREE.BoxGeometry(18, 0.08, 0.7), markMat);
    dash.position.set(30 + i * 40, 0.24, 0);
    group.add(dash);
  }

  for (let i = -4; i <= 4; i += 1) {
    const bar = new THREE.Mesh(new THREE.BoxGeometry(22, 0.08, 0.9), markMat);
    bar.position.set(24, 0.24, i * 2.6);
    group.add(bar);
  }

  const numbers = new THREE.Mesh(new THREE.BoxGeometry(28, 0.08, 12), markMat);
  numbers.position.set(70, 0.24, 0);
  group.add(numbers);

  const edgeMat = new THREE.MeshStandardMaterial({
    color: 0xfff3c4,
    emissive: 0x887733,
    emissiveIntensity: 0.6,
    roughness: 0.4,
  });
  for (let i = 0; i < 24; i += 1) {
    const x = 20 + i * (runway.length_m / 24);
    for (const z of [-runway.width_m / 2 - 0.6, runway.width_m / 2 + 0.6]) {
      const light = new THREE.Mesh(new THREE.BoxGeometry(0.7, 0.5, 0.7), edgeMat);
      light.position.set(x, 0.4, z);
      group.add(light);
    }
  }

  root.add(group);
}

function addThresholdChevrons(root: THREE.Group, runway: RunwayInfo): void {
  const mat = new THREE.MeshStandardMaterial({ color: 0xd9c27a, roughness: 0.7 });
  const heading = THREE.MathUtils.degToRad(runway.heading_deg);
  for (let i = 1; i <= 18; i += 1) {
    const chevron = new THREE.Mesh(new THREE.ConeGeometry(8, 22, 3), mat);
    chevron.rotation.x = Math.PI / 2;
    const dist = -50 - i * 70;
    chevron.position.set(Math.sin(heading) * dist, 0.4, -Math.cos(heading) * dist);
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
