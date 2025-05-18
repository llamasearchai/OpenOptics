export class ARController {
  constructor(videoElement: HTMLVideoElement, options: any) {}
  addElement(element: ARElement, transformation: any) {}
  addMarker(marker: ARMarker) {}
  start() {}
  stop() {}
}

export class ARMarker {
  constructor(patternUrl: string, size: number) {}
  public onDetect: ((transformation: any) => void) | undefined;
}

export interface ARElement {
  id: string;
  type: 'model3d' | 'text' | 'overlay' | 'light' | 'status_indicator';
  data?: any; // e.g., path to 3D model, text content, color, intensity
  position?: { x: number; y: number; z: number };
  rotation?: { x: number; y: number; z: number };
  scale?: { x: number; y: number; z: number };
  interactive?: boolean;
  parentId?: string | null; // For hierarchical elements
  metadata?: Record<string, any>; // For additional info
} 