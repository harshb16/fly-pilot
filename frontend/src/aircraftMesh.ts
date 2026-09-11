import * as THREE from "three";

/** High-wing Cessna-like mesh. Nose points along local -Z. */
export function createCessna(): THREE.Group {
  const aircraft = new THREE.Group();
  aircraft.name = "cessna172";

  const white = new THREE.MeshStandardMaterial({ color: 0xf4f1ea, roughness: 0.42, metalness: 0.08 });
  const stripe = new THREE.MeshStandardMaterial({ color: 0x1f4f8a, roughness: 0.45, metalness: 0.1 });
  const glass = new THREE.MeshStandardMaterial({
    color: 0x87b8d4,
    roughness: 0.12,
    metalness: 0.35,
    transparent: true,
    opacity: 0.55,
  });
  const dark = new THREE.MeshStandardMaterial({ color: 0x22262c, roughness: 0.5, metalness: 0.2 });
  const tire = new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 0.9 });

  const fuselage = new THREE.Mesh(new THREE.CylinderGeometry(0.55, 0.42, 7.2, 16, 1, false), white);
  fuselage.rotation.x = Math.PI / 2;
  fuselage.castShadow = true;
  aircraft.add(fuselage);

  const nose = new THREE.Mesh(new THREE.SphereGeometry(0.42, 16, 12), dark);
  nose.position.z = -3.55;
  aircraft.add(nose);

  const cabin = new THREE.Mesh(new THREE.SphereGeometry(0.72, 16, 12, 0, Math.PI * 2, 0, Math.PI / 1.7), glass);
  cabin.position.set(0, 0.22, -0.4);
  cabin.scale.set(1.05, 0.85, 1.35);
  aircraft.add(cabin);

  const band = new THREE.Mesh(new THREE.CylinderGeometry(0.56, 0.44, 7.0, 16, 1, true), stripe);
  band.rotation.x = Math.PI / 2;
  band.scale.set(1, 1, 0.18);
  band.position.y = -0.12;
  aircraft.add(band);

  const wing = new THREE.Mesh(new THREE.BoxGeometry(11.2, 0.12, 1.7), white);
  wing.position.set(0, 0.95, -0.2);
  wing.castShadow = true;
  aircraft.add(wing);

  const wingStripe = new THREE.Mesh(new THREE.BoxGeometry(11.25, 0.04, 0.22), stripe);
  wingStripe.position.set(0, 1.02, -0.2);
  aircraft.add(wingStripe);

  const strutL = new THREE.Mesh(new THREE.BoxGeometry(0.06, 1.1, 0.06), dark);
  strutL.position.set(-1.6, 0.35, -0.1);
  strutL.rotation.z = 0.45;
  aircraft.add(strutL);
  const strutR = strutL.clone();
  strutR.position.x = 1.6;
  strutR.rotation.z = -0.45;
  aircraft.add(strutR);

  const hstab = new THREE.Mesh(new THREE.BoxGeometry(3.4, 0.08, 0.85), white);
  hstab.position.set(0, 0.15, 3.15);
  aircraft.add(hstab);

  const vstab = new THREE.Mesh(new THREE.BoxGeometry(0.1, 1.35, 1.1), white);
  vstab.position.set(0, 0.85, 3.15);
  aircraft.add(vstab);
  const rudderStripe = new THREE.Mesh(new THREE.BoxGeometry(0.12, 1.1, 0.18), stripe);
  rudderStripe.position.set(0, 0.85, 3.5);
  aircraft.add(rudderStripe);

  const spinner = new THREE.Mesh(new THREE.ConeGeometry(0.16, 0.28, 12), dark);
  spinner.rotation.x = -Math.PI / 2;
  spinner.position.z = -3.85;
  aircraft.add(spinner);

  const prop = new THREE.Mesh(new THREE.BoxGeometry(0.08, 1.9, 0.12), dark);
  prop.position.z = -3.72;
  prop.name = "prop";
  aircraft.add(prop);

  addWheel(aircraft, 0, -0.85, -2.15, tire, dark);
  addWheel(aircraft, -1.05, -0.95, 0.35, tire, dark);
  addWheel(aircraft, 1.05, -0.95, 0.35, tire, dark);

  aircraft.traverse((obj) => {
    if ((obj as THREE.Mesh).isMesh) {
      obj.castShadow = true;
      obj.receiveShadow = true;
    }
  });
  return aircraft;
}

function addWheel(
  parent: THREE.Group,
  x: number,
  y: number,
  z: number,
  tire: THREE.Material,
  strut: THREE.Material,
): void {
  const wheel = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 0.12, 12), tire);
  wheel.rotation.z = Math.PI / 2;
  wheel.position.set(x, y, z);
  parent.add(wheel);
  const leg = new THREE.Mesh(new THREE.BoxGeometry(0.05, Math.abs(y) + 0.2, 0.05), strut);
  leg.position.set(x, y / 2, z);
  parent.add(leg);
}

export function applyJsbsimPose(
  object: THREE.Object3D,
  eastM: number,
  northM: number,
  upM: number,
  headingDeg: number,
  pitchDeg: number,
  rollDeg: number,
): void {
  // Visual world: X east, Y up, Z south so heading 0 looks along -Z (north).
  object.position.set(eastM, upM, -northM);
  object.rotation.order = "YXZ";
  object.rotation.y = THREE.MathUtils.degToRad(-headingDeg);
  object.rotation.x = THREE.MathUtils.degToRad(pitchDeg);
  object.rotation.z = THREE.MathUtils.degToRad(-rollDeg);
}
