import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { ArrowLeft, Moon, Sun, Save } from "lucide-react";

const STYLE_OPTIONS = ["Casual", "Formal", "Traditional", "Streetwear", "Party", "Minimalist", "Bohemian", "Athleisure"];
const COLOR_OPTIONS = ["Neutral", "Monochrome", "Earth Tones", "Pastels", "Bold", "Dark"];

const Settings = () => {
  const navigate = useNavigate();
  const { toast } = useToast();

  const [isDark, setIsDark] = useState(true);
  const [preferredStyles, setPreferredStyles] = useState<string[]>(["casual"]);
  const [preferredColors, setPreferredColors] = useState<string[]>(["neutral"]);

  useEffect(() => {
    // Check for stored theme preference
    const storedTheme = localStorage.getItem("theme");
    if (storedTheme === "light") {
      setIsDark(false);
      document.documentElement.classList.remove("dark");
    } else {
      setIsDark(true);
      document.documentElement.classList.add("dark");
    }

    // Load saved preferences from localStorage
    const savedStyles = localStorage.getItem("preferredStyles");
    const savedColors = localStorage.getItem("preferredColors");
    if (savedStyles) setPreferredStyles(JSON.parse(savedStyles));
    if (savedColors) setPreferredColors(JSON.parse(savedColors));
  }, []);

  const handleThemeToggle = (checked: boolean) => {
    setIsDark(checked);
    if (checked) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  };

  const toggleStyle = (style: string) => {
    const styleLower = style.toLowerCase();
    setPreferredStyles((prev) =>
      prev.includes(styleLower)
        ? prev.filter((s) => s !== styleLower)
        : [...prev, styleLower]
    );
  };

  const toggleColor = (color: string) => {
    const colorLower = color.toLowerCase();
    setPreferredColors((prev) =>
      prev.includes(colorLower)
        ? prev.filter((c) => c !== colorLower)
        : [...prev, colorLower]
    );
  };

  const handleSave = () => {
    localStorage.setItem("preferredStyles", JSON.stringify(preferredStyles));
    localStorage.setItem("preferredColors", JSON.stringify(preferredColors));
    toast({
      title: "Saved",
      description: "Your preferences have been updated.",
    });
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-40 bg-background/80 backdrop-blur-xl border-b border-border/20">
        <div className="flex items-center gap-4 px-6 py-4">
          <button
            onClick={() => navigate("/")}
            className="p-2 rounded-full hover:bg-accent transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <h1 className="text-xl font-semibold">Settings</h1>
        </div>
      </header>

      {/* Main Content */}
      <main className="pt-24 pb-8 px-4 md:px-8 max-w-2xl mx-auto">
        <div className="space-y-6">
          {/* Theme Toggle */}
          <div className="bg-card/10 backdrop-blur-xl border border-border/20 rounded-2xl p-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                {isDark ? (
                  <Moon className="w-5 h-5 text-muted-foreground" />
                ) : (
                  <Sun className="w-5 h-5 text-muted-foreground" />
                )}
                <div>
                  <Label className="text-base font-medium">Dark Mode</Label>
                  <p className="text-sm text-muted-foreground">
                    Toggle between light and dark themes
                  </p>
                </div>
              </div>
              <Switch
                checked={isDark}
                onCheckedChange={handleThemeToggle}
              />
            </div>
          </div>

          {/* Style Preferences */}
          <div className="bg-card/10 backdrop-blur-xl border border-border/20 rounded-2xl p-6">
            <h2 className="text-base font-medium mb-4">Preferred Styles</h2>
            <p className="text-sm text-muted-foreground mb-4">
              Select your favorite fashion styles for personalized suggestions
            </p>
            <div className="flex flex-wrap gap-2">
              {STYLE_OPTIONS.map((style) => (
                <button
                  key={style}
                  onClick={() => toggleStyle(style)}
                  className={`px-4 py-2 text-sm rounded-full border transition-colors ${
                    preferredStyles.includes(style.toLowerCase())
                      ? "bg-foreground text-background border-foreground"
                      : "border-border hover:border-foreground/50"
                  }`}
                >
                  {style}
                </button>
              ))}
            </div>
          </div>

          <div className="bg-card/10 backdrop-blur-xl border border-border/20 rounded-2xl p-6">
            <h2 className="text-base font-medium mb-4">Color Preferences</h2>
            <p className="text-sm text-muted-foreground mb-4">
              Choose color palettes you prefer
            </p>
            <div className="flex flex-wrap gap-2">
              {COLOR_OPTIONS.map((color) => (
                <button
                  key={color}
                  onClick={() => toggleColor(color)}
                  className={`px-4 py-2 text-sm rounded-full border transition-colors ${
                    preferredColors.includes(color.toLowerCase())
                      ? "bg-foreground text-background border-foreground"
                      : "border-border hover:border-foreground/50"
                  }`}
                >
                  {color}
                </button>
              ))}
            </div>
          </div>

          <Button
            onClick={handleSave}
            className="w-full py-6"
          >
            <Save className="w-4 h-4 mr-2" />
            Save Preferences
          </Button>
        </div>
      </main>
    </div>
  );
};

export default Settings;
