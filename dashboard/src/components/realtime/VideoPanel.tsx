import { MutableRefObject } from "react";
import { BoundingBoxOverlay } from "../../domain/visualization/boundingBoxes";

type VideoPanelProps = {
    hasVideo: boolean;
    videoRef: MutableRefObject<HTMLVideoElement | null>;
    overlays: BoundingBoxOverlay[];
    onVideoElementReady?: (el: HTMLVideoElement | null) => void;
};

export function VideoPanel({
    hasVideo,
    videoRef,
    overlays,
    onVideoElementReady,
}: VideoPanelProps) {
    const handleVideoRef = (el: HTMLVideoElement | null) => {
        videoRef.current = el;
        onVideoElementReady?.(el);
    };

    return (
        <div className="relative h-full w-full overflow-hidden rounded-xl border border-gray-800 bg-gradient-to-br from-gray-800 to-gray-900 shadow-[0_8px_20px_rgba(0,0,0,0.4)]">
            {/* 실제 비디오 */}
            <video
                ref={handleVideoRef}
                className={`absolute inset-0 h-full w-full object-cover transition-opacity duration-500 ${hasVideo ? "opacity-100" : "opacity-0"
                    }`}
                autoPlay
                playsInline
                muted
            />

            {/* 비디오가 없을 때: 그리드 + 텍스트 플레이스홀더 */}
            {!hasVideo && (
                <div className="absolute inset-0 flex items-center justify-center">

                    {/* 중앙 정보 텍스트 */}
                    <div className="relative text-center text-gray-300">
                        <div className="mt-2 text-[16px] uppercase tracking-[0.18em] text-gray-500">
                            Connecting...
                        </div>
                    </div>
                </div>
            )}

            {/* 좌상단 활성 엔티티 태그들 */}
            {overlays.length > 0 && (
                <div className="absolute left-4 top-4 z-30 flex flex-wrap gap-2">
                    {overlays.slice(0, 4).map((overlay) => (
                        <div
                            key={`chip-${overlay.id}`}
                            className="rounded-md px-2 py-1 text-[11px] text-white backdrop-blur-sm shadow-sm"
                            style={{
                                backgroundColor: `${overlay.color}e6`,
                            }}
                        >
                            <span className="font-medium">{overlay.label}</span>
                        </div>
                    ))}
                </div>
            )}

            {/* Bounding box overlays */}
            {overlays.length > 0 && (
                <div className="pointer-events-none absolute inset-0 z-20">
                    {overlays.map((overlay) => (
                        <div
                            key={overlay.id}
                            className="absolute rounded-md border-2"
                            style={{
                                left: `${overlay.box.leftPct * 100}%`,
                                top: `${overlay.box.topPct * 100}%`,
                                width: `${overlay.box.widthPct * 100}%`,
                                height: `${overlay.box.heightPct * 100}%`,
                                borderColor: overlay.color,
                                boxShadow: `0 0 0 1px ${overlay.color}40`,
                            }}
                        >
                            <div
                                className="absolute left-0 top-0 flex items-center gap-2 rounded-md px-2 py-1 text-[11px] font-semibold"
                                style={{
                                    backgroundColor: `${overlay.color}e6`,
                                    color: "#0b1221",
                                }}
                            >
                                <span>{overlay.label}</span>
                                {typeof overlay.confidence === "number" &&
                                    !Number.isNaN(overlay.confidence) && (
                                        <span className="text-[10px] opacity-80">
                                            {(
                                                overlay.confidence > 1
                                                    ? overlay.confidence
                                                    : overlay.confidence * 100
                                            ).toFixed(1)}
                                            {overlay.confidence <= 1 ? "%" : ""}
                                        </span>
                                    )}
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {/* 상/하단 살짝 비네팅 느낌 */}
            <div className="pointer-events-none absolute inset-0 z-10 bg-gradient-to-b from-white/10 via-transparent to-white/10" />
        </div>
    );
}
