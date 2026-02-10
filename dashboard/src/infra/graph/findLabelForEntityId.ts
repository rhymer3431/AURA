
export function findLabelForEntityId(meta: any, entityId: number | string): string {
    const idNum =
        typeof entityId === "string" ? Number(entityId) : entityId;

    if (Number.isFinite(idNum)) {
        const records = meta.entityRecords ?? [];
        const entities = meta.entities ?? [];

        const rec = records.find((r: any) => r.entityId === idNum);
        if (rec?.baseCls) return rec.baseCls;

        const ent = entities.find((e: any) => e.entityId === idNum);
        if (ent?.cls) return ent.cls;
    }

    return "entity";
}