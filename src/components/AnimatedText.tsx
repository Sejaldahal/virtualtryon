import { useState, useEffect } from "react";

const phrases = [
  "Try Clothes",
  "Change Designs",
  "Mix Styles",
  "Find Your Look",
  "Transform Instantly",
];

const AnimatedText = () => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      setIsAnimating(true);
      setTimeout(() => {
        setCurrentIndex((prev) => (prev + 1) % phrases.length);
        setIsAnimating(false);
      }, 500);
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="h-[1.2em] overflow-hidden relative">
      <span
        className={`block transition-all duration-500 ${
          isAnimating
            ? "translate-y-full opacity-0"
            : "translate-y-0 opacity-100"
        }`}
      >
        {phrases[currentIndex]}
      </span>
    </div>
  );
};

export default AnimatedText;
