import { useNavigate } from "react-router-dom";
import { useState, useEffect } from "react";
 import { Settings, Sparkles, Zap, Palette, Wand2 } from "lucide-react";
import { Button } from "@/components/ui/button";

const phrases = [
  "Try Clothes",
  "Change Designs",
  "Mix Styles",
  "Find Your Look",
  "Transform Instantly",
];

const LandingPage = () => {
  const navigate = useNavigate();
  const [currentPhraseIndex, setCurrentPhraseIndex] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);

  useEffect(() => {
    const phraseInterval = setInterval(() => {
      setIsAnimating(true);
      setTimeout(() => {
        setCurrentPhraseIndex((prev) => (prev + 1) % phrases.length);
        setIsAnimating(false);
      }, 400);
    }, 2500);

    return () => clearInterval(phraseInterval);
  }, []);

  return (
    <div className="relative min-h-screen overflow-hidden bg-background">
      {/* Gradient background */}
      <div className="absolute inset-0 bg-gradient-to-br from-background via-background to-accent/20" />

    
      {/* Main Content - Split Layout */}
      <div className="relative z-10 flex min-h-screen">
        {/* Left Side - Text Content */}
        <div className="flex-1 flex flex-col justify-center px-8 md:px-12 lg:px-20">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-5 py-2 rounded-full glass border border-foreground/10 w-fit mb-8">
            <Sparkles className="w-4 h-4 text-primary animate-pulse" />
            <span className="text-xs tracking-[0.3em] uppercase text-muted-foreground font-medium">
              Virtual Try-On
            </span>
          </div>

          {/* Animated Heading */}
          <h1 className="text-4xl md:text-5xl lg:text-6xl xl:text-7xl font-display font-extralight tracking-tight leading-[1.1] mb-8">
            <div className="h-[1.2em] overflow-hidden mb-2">
              <span
                className={`block bg-gradient-to-r from-foreground via-foreground to-foreground/70 bg-clip-text text-transparent transition-all duration-400 ${
                  isAnimating
                    ? "-translate-y-full opacity-0"
                    : "translate-y-0 opacity-100"
                }`}
              >
                {phrases[currentPhraseIndex]}
              </span>
            </div>
            <span className="block text-gradient font-light text-2xl md:text-4xl lg:text-5xl">
              Instantly.
            </span>
          </h1>

          {/* Subheading */}
          <p className="text-base md:text-lg text-muted-foreground max-w-md font-light leading-relaxed mb-8">
            Experience AI-powered virtual try-on technology.
            <span className="block mt-2 text-foreground/60">Transform your style in seconds.</span>
          </p>

          {/* Feature pills */}
          <div className="flex flex-wrap gap-3 mb-10">
            <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-foreground/5 border border-foreground/10">
              <Zap className="w-4 h-4 text-primary" />
              <span className="text-sm text-muted-foreground">Instant Results</span>
            </div>
            <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-foreground/5 border border-foreground/10">
              <Palette className="w-4 h-4 text-accent-foreground" />
              <span className="text-sm text-muted-foreground">Custom Styles</span>
            </div>
            
          </div>

          {/* CTA Button */}
          <div className="flex flex-col items-start gap-3">
            <Button
              onClick={() => navigate("/try-on")}
              size="lg"
              className="group relative px-10 py-6 text-base font-light tracking-wider bg-foreground text-background hover:bg-foreground/90 rounded-full glow-button shadow-2xl shadow-foreground/20"
            >
              <span className="relative z-10 flex items-center gap-3">
                Start Styling
                <svg
                  className="w-5 h-5 group-hover:translate-x-1 transition-transform"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M17 8l4 4m0 0l-4 4m4-4H3"
                  />
                </svg>
              </span>
            </Button>
            
          </div>
        </div>

        {/* Right Side - Video */}
        <div className="hidden lg:flex flex-1 items-center justify-center relative">
          <div className="relative w-full h-full max-w-2xl">
            {/* Video container with overlay */}
            <div className="absolute inset-8 rounded-3xl overflow-hidden shadow-2xl shadow-foreground/10">
              <video
                autoPlay
                muted
                loop
                playsInline
                className="w-full h-full object-cover"
              >
                <source src="/videos/hero-background.mp4" type="video/mp4" />
              </video>
              {/* Gradient overlay on video */}
              <div className="absolute inset-0 bg-gradient-to-t from-background/40 via-transparent to-transparent" />
              <div className="absolute inset-0 bg-gradient-to-r from-background/60 via-transparent to-transparent" />
            </div>

            {/* Decorative frame */}
            <div className="absolute top-4 left-4 w-24 h-24 border-l-2 border-t-2 border-foreground/20 rounded-tl-3xl" />
            <div className="absolute bottom-4 right-4 w-24 h-24 border-r-2 border-b-2 border-foreground/20 rounded-br-3xl" />

            {/* Floating badge */}
            <div className="absolute bottom-12 left-12 bg-background/90 backdrop-blur-sm px-4 py-3 rounded-2xl border border-foreground/10 shadow-lg">
              <p className="text-sm font-medium">Virtual Try-On</p>
              <p className="text-xs text-muted-foreground">AI-powered styling</p>
            </div>
          </div>
        </div>

        {/* Mobile Video - Below content */}
        <div className="lg:hidden absolute bottom-0 left-0 right-0 h-48 overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-t from-background via-background/80 to-transparent z-10" />
          <video
            autoPlay
            muted
            loop
            playsInline
            className="w-full h-full object-cover opacity-40"
          >
            <source src="/videos/hero-background.mp4" type="video/mp4" />
          </video>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;
