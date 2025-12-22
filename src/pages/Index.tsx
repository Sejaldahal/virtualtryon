import { useState, useEffect } from "react";
import IntroAnimation from "@/components/IntroAnimation";
import LandingPage from "@/components/LandingPage";

const Index = () => {
  const [showIntro, setShowIntro] = useState(true);

  useEffect(() => {
    document.documentElement.classList.add("dark");
  }, []);

  return (
    <>
      {showIntro && <IntroAnimation onComplete={() => setShowIntro(false)} />}
      <div className={showIntro ? "opacity-0" : "animate-fade-in"}>
        <LandingPage />
      </div>
    </>
  );
};

export default Index;
