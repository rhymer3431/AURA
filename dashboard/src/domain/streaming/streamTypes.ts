// src/domain/streaming/streamTypes.ts
// Backend Transport Layer Types — Domain 독립적

export type StreamState =
  | "idle"
  | "connecting"
  | "playing"
  | "paused"
  | "error";

/**
 * Entity detected in a frame (backend → frontend)
 */
export type StreamEntity = {
  entityId: number;
  trackId: number;
  cls: string;
  box: number[];
  score: number;
};

/**
 * LTM Entity record (backend → frontend)
 */
export type StreamEntityRecord = {
  entityId: number;
  baseCls: string;
  lastBox: number[];
  lastSeenFrame: number;
  seenCount: number;
  trackHistory: number[];
  meta?: Record<string, unknown>;
};

/**
 * Backend relation format (serialize_sg_diff output)
 */
export type BackendRelation = {
  subject: number;
  predicate: string;
  object: number;
  confidence?: number;
  type?: "static" | "temporal";
};

/**
 * Backend removed-edge key format
 */
export type BackendEdgeKey = {
  subject: number;
  predicate: string;
  object: number;
};

/**
 * Raw SG Diff returned from backend (exact JSON shape)
 */
export type RawSceneGraphDiff = {
  nodesAdded?: StreamEntity[];
  nodesRemoved?: Array<number | string>;

  edgesAdded?: BackendRelation[];
  edgesRemoved?: BackendEdgeKey[];
};

/**
 * Transport-layer metadata message (backend → frontend)
 * Domain 변환은 별도 단계에서 수행함.
 */
export type StreamMetadata = {
  frameIdx: number;
  caption?: string;
  focusTargets?: string[];
  entities?: StreamEntity[];
  entityRecords?: StreamEntityRecord[];

  // Backend relation snapshot
  relations?: BackendRelation[];

  // Raw diff from backend (Transport only)
  sceneGraphSnapshot?: unknown;
  sceneGraphDiff?: RawSceneGraphDiff | null;
  streamSource?: StreamSource;
  slamPoseTopic?: string;
  slamPose?: StreamSlamPose;
  semanticMap?: StreamSemanticMap;
};

/**
 * Initialization message
 */
export type InitMessage = {
  type: "init";
  runtimeReady: boolean;
  llmModel: string;
  device: string;
};

export type MetadataMessage = StreamMetadata & { type?: "metadata" };

export type StreamError = {
  message: string;
};

export type StreamSource = {
  type?: string;
  imageTopic?: string;
  videoPath?: string;
};

export type StreamSlamPose = {
  position: {
    x: number;
    y: number;
    z: number;
  };
  orientation: {
    x: number;
    y: number;
    z: number;
    w: number;
  };
  stamp?: {
    sec: number;
    nanosec: number;
  };
  frameId?: string;
};

export type StreamSemanticMap = {
  projectedMapTopic?: string;
  octomapCloudTopic?: string;
  projectedMap?: StreamProjectedMap;
  octomapCloud?: StreamOctomapCloud;
};

export type StreamProjectedMap = {
  width: number;
  height: number;
  resolution: number;
  occupiedCells: number;
  freeCells: number;
  unknownCells: number;
  knownRatio: number;
  frameId?: string;
  stamp?: {
    sec: number;
    nanosec: number;
  };
  previewRevision?: number;
  preview?: StreamProjectedMapPreview;
};

export type StreamProjectedMapPreview = {
  encoding: string;
  width: number;
  height: number;
  rows: string[];
  revision?: number;
};

export type StreamOctomapCloud = {
  pointCount: number;
  frameId?: string;
  stamp?: {
    sec: number;
    nanosec: number;
  };
};
