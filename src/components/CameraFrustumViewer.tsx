import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { Empty } from 'antd';
import type { InputImageCameraInfo } from '@/types/api';

interface Props {
  images?: InputImageCameraInfo[];
  height?: number;
}

export function CameraFrustumViewer({ images, height = 400 }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

  const imagesWithExtrinsics = useMemo(
    () => (images?.filter((img) => img.cameraExtrinsics) ?? []).sort((a, b) => a.index - b.index),
    [images],
  );

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !imagesWithExtrinsics.length) return;

    const w = container.clientWidth;
    const h = height;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);

    const camera = new THREE.PerspectiveCamera(45, w / h, 0.05, 50);
    camera.position.set(4, 3, 4);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;

    scene.add(new THREE.GridHelper(6, 12, 0x333344, 0x222233));
    scene.add(new THREE.AxesHelper(2));

    const ctxColor = 0x1677ff;
    const normColor = 0x888888;
    const trajectoryPoints: THREE.Vector3[] = [];

    imagesWithExtrinsics.forEach((img) => {
      const { translation } = img.cameraExtrinsics!;
      if (!translation?.length) return;
      const color = img.isContextView ? ctxColor : normColor;
      const position = new THREE.Vector3(translation[0] ?? 0, translation[1] ?? 0, translation[2] ?? 0);
      trajectoryPoints.push(position.clone());

      const dot = new THREE.Mesh(
        new THREE.SphereGeometry(0.05, 8, 8),
        new THREE.MeshBasicMaterial({ color }),
      );
      dot.position.copy(position);
      scene.add(dot);
    });

    if (trajectoryPoints.length >= 2) {
      const trajectory = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(trajectoryPoints),
        new THREE.LineBasicMaterial({ color: 0xffd666 }),
      );
      scene.add(trajectory);
    }

    if (trajectoryPoints.length) {
      const start = new THREE.Mesh(
        new THREE.SphereGeometry(0.075, 12, 12),
        new THREE.MeshBasicMaterial({ color: 0x52c41a }),
      );
      start.position.copy(trajectoryPoints[0]);
      scene.add(start);

      const end = new THREE.Mesh(
        new THREE.SphereGeometry(0.075, 12, 12),
        new THREE.MeshBasicMaterial({ color: 0xff4d4f }),
      );
      end.position.copy(trajectoryPoints[trajectoryPoints.length - 1]);
      scene.add(end);
    }

    let id = 0;
    const animate = () => {
      id = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const onResize = () => {
      const nw = container.clientWidth;
      const nh = height;
      camera.aspect = nw / nh;
      camera.updateProjectionMatrix();
      renderer.setSize(nw, nh);
    };
    window.addEventListener('resize', onResize);

    return () => {
      cancelAnimationFrame(id);
      window.removeEventListener('resize', onResize);
      controls.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [imagesWithExtrinsics]);

  if (!imagesWithExtrinsics.length) {
    return <Empty description={'\u6682\u65e0\u76f8\u673a\u53c2\u6570\u6570\u636e\uff0c\u65e0\u6cd5\u663e\u793a\u76f8\u673a\u8f68\u8ff9'} />;
  }

  return <div ref={containerRef} style={{ width: '100%', height, borderRadius: 8, overflow: 'hidden' }} />;
}
