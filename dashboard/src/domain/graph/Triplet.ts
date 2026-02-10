export type Triplet = {
    subject: string;
    subject_id: string | number;
    predicate: string;
    object: string;
    object_id: string | number;
    confidence: number;
    type: "static" | "temporal";
};
