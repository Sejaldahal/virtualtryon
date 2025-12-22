import { useEffect, useState } from "react";

interface IntroAnimationProps {
  onComplete: () => void;
}

const IntroAnimation = ({ onComplete }: IntroAnimationProps) => {
  const [phase, setPhase] = useState<"wave" | "zoom" | "done">("wave");

  useEffect(() => {
    const waveTimer = setTimeout(() => setPhase("zoom"), 2000);
    const zoomTimer = setTimeout(() => {
      setPhase("done");
      onComplete();
    }, 2500);

    return () => {
      clearTimeout(waveTimer);
      clearTimeout(zoomTimer);
    };
  }, [onComplete]);

  if (phase === "done") return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-background overflow-hidden">
      {/* Ambient background */}
      <div className="absolute inset-0 bg-gradient-to-br from-background via-muted/20 to-background" />
      
      {/* Fabric wave element */}
      <div
        className={`relative transition-all duration-500 ${
          phase === "zoom" ? "animate-intro-zoom" : ""
        }`}
      >
        {/* Outer glow ring */}
        <div className="absolute inset-0 -m-20 rounded-full bg-gradient-to-r from-transparent via-foreground/5 to-transparent animate-spin-slow" />
        
        {/* Main fabric shape */}
        <div className="relative w-64 h-64 flex items-center justify-center">
          {/* Animated rings */}
          {[...Array(3)].map((_, i) => (
            <div
              key={i}
              className="absolute inset-0 rounded-full border border-foreground/20 animate-fabric-wave"
              style={{
                animationDelay: `${i * 0.3}s`,
                transform: `scale(${1 + i * 0.2})`,
              }}
            />
          ))}
          
          {/* Center fabric icon */}
          <div className="relative z-10 w-32 h-32 animate-float">
            <svg
              viewBox="0 0 100 100"
              className="w-full h-full"
              fill="none"
              stroke="currentColor"
              strokeWidth="0.5"
            >
              {/* Stylized fabric/cloth shape */}
              <path
                d="M20 30 Q50 10 80 30 Q90 50 80 70 Q50 90 20 70 Q10 50 20 30"
                className="animate-pulse-glow"
                style={{ filter: "drop-shadow(0 0 10px currentColor)" }}
              />
              <path
                d="M30 40 Q50 25 70 40 Q75 50 70 60 Q50 75 30 60 Q25 50 30 40"
                opacity="0.6"
              />
              <path
                d="M40 48 Q50 42 60 48 Q62 50 60 52 Q50 58 40 52 Q38 50 40 48"
                opacity="0.3"
              />
            </svg>
          </div>
          
          {/* Shimmer effect */}
          <div className="absolute inset-0 rounded-full overflow-hidden">
            <div className="w-full h-full shimmer opacity-30" />
          </div>
        </div>
        
        {/* Brand text */}
        <div className="absolute -bottom-16 left-1/2 -translate-x-1/2 whitespace-nowrap">
          <span className="text-2xl font-display tracking-[0.3em] text-foreground/80">
            VIRTUAL TRY-ON
          </span>
        </div>
      </div>
      
      {/* Corner decorations */}
      <div className="absolute top-8 left-8 w-16 h-16 border-l border-t border-foreground/10" />
      <div className="absolute top-8 right-8 w-16 h-16 border-r border-t border-foreground/10" />
      <div className="absolute bottom-8 left-8 w-16 h-16 border-l border-b border-foreground/10" />
      <div className="absolute bottom-8 right-8 w-16 h-16 border-r border-b border-foreground/10" />
    </div>
  );
};

export default IntroAnimation;
