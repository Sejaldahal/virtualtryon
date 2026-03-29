// // import { useState, useRef } from "react";
// // import { useNavigate } from "react-router-dom";
// // import { supabase } from "@/integrations/supabase/client";
// // import { Button } from "@/components/ui/button";
// // import { Textarea } from "@/components/ui/textarea";
// // import { useToast } from "@/hooks/use-toast";
// // import {
// //   Settings,
// //   Upload,
// //   Sparkles,
// //   Download,
// //   Loader2,
// //   User,
// //   Shirt,
// //   X,
// // } from "lucide-react";

// // const PRESET_STYLES = ["Casual", "Formal", "Traditional", "Streetwear", "Party"];

// // // Sample model images (silhouettes/placeholders)
// // const SAMPLE_MODELS = [
// //   { id: 1, name: "Model 1", image: "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=200&h=300&fit=crop&crop=face" },
// //   { id: 2, name: "Model 2", image: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=200&h=300&fit=crop&crop=face" },
// //   { id: 3, name: "Model 3", image: "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=200&h=300&fit=crop&crop=face" },
// //   { id: 4, name: "Model 4", image: "https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?w=200&h=300&fit=crop&crop=face" },
// // ];

// // // Sample clothing items
// // const SAMPLE_CLOTHES = [
// //   { id: 1, name: "Classic Blazer", category: "Formal", image: "https://images.unsplash.com/photo-1594938298603-c8148c4dae35?w=200&h=250&fit=crop" },
// //   { id: 2, name: "Summer Dress", category: "Casual", image: "https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?w=200&h=250&fit=crop" },
// //   { id: 3, name: "Denim Jacket", category: "Streetwear", image: "https://images.unsplash.com/photo-1551028719-00167b16eac5?w=200&h=250&fit=crop" },
// //   { id: 4, name: "Silk Saree", category: "Traditional", image: "https://images.unsplash.com/photo-1610030469983-98e550d6193c?w=200&h=250&fit=crop" },
// //   { id: 5, name: "Party Gown", category: "Party", image: "https://images.unsplash.com/photo-1518611012118-696072aa579a?w=200&h=250&fit=crop" },
// //   { id: 6, name: "Casual Tee", category: "Casual", image: "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=200&h=250&fit=crop" },
// // ];

// // const TryOn = () => {
// //   const navigate = useNavigate();
// //   const { toast } = useToast();

// //   const [personImage, setPersonImage] = useState<string | null>(null);
// //   const [clothingImage, setClothingImage] = useState<string | null>(null);
// //   const [prompt, setPrompt] = useState("");
// //   const [selectedStyle, setSelectedStyle] = useState<string | null>(null);
// //   const [generating, setGenerating] = useState(false);
// //   const [result, setResult] = useState<string | null>(null);
// //   const [aiSuggestion, setAiSuggestion] = useState<string | null>(null);

// //   const personInputRef = useRef<HTMLInputElement>(null);
// //   const clothingInputRef = useRef<HTMLInputElement>(null);

// //   const handleImageUpload = (
// //     e: React.ChangeEvent<HTMLInputElement>,
// //     type: "person" | "clothing"
// //   ) => {
// //     const file = e.target.files?.[0];
// //     if (file) {
// //       const reader = new FileReader();
// //       reader.onload = (event) => {
// //         const dataUrl = event.target?.result as string;
// //         if (type === "person") {
// //           setPersonImage(dataUrl);
// //         } else {
// //           setClothingImage(dataUrl);
// //         }
// //       };
// //       reader.readAsDataURL(file);
// //     }
// //   };

// //   const handleSelectSampleModel = (imageUrl: string) => {
// //     setPersonImage(imageUrl);
// //     toast({
// //       title: "Model selected",
// //       description: "Sample model has been applied.",
// //     });
// //   };

// //   const handleSelectSampleClothing = (item: typeof SAMPLE_CLOTHES[0]) => {
// //     setClothingImage(item.image);
// //     setSelectedStyle(item.category);
// //     toast({
// //       title: "Clothing selected",
// //       description: `${item.name} has been applied.`,
// //     });
// //   };

// //   const handleGenerate = async () => {
// //     if (!personImage && !clothingImage && !prompt && !selectedStyle) {
// //       toast({
// //         title: "Missing input",
// //         description: "Please upload images or describe your desired style.",
// //         variant: "destructive",
// //       });
// //       return;
// //     }

// //     setGenerating(true);
// //     setAiSuggestion(null);

// //     try {
// //       const fullPrompt = [
// //         selectedStyle ? `Style: ${selectedStyle}` : "",
// //         prompt,
// //       ]
// //         .filter(Boolean)
// //         .join(". ");

// //       const { data, error } = await supabase.functions.invoke("style-advisor", {
// //         body: {
// //           prompt: fullPrompt || "Suggest a fashionable outfit",
// //           hasPersonImage: !!personImage,
// //           hasClothingImage: !!clothingImage,
// //         },
// //       });

// //       if (error) {
// //         throw error;
// //       }

// //       setAiSuggestion(data.suggestion);
      
// //       if (personImage || clothingImage) {
// //         setResult(personImage || clothingImage);
// //       }

// //       toast({
// //         title: "Styling complete!",
// //         description: "Your AI-powered fashion suggestions are ready.",
// //       });
// //     } catch (err) {
// //       console.error("Generation error:", err);
// //       toast({
// //         title: "Generation failed",
// //         description: "Please try again later.",
// //         variant: "destructive",
// //       });
// //     } finally {
// //       setGenerating(false);
// //     }
// //   };

// //   const handleDownload = () => {
// //     if (result) {
// //       const link = document.createElement("a");
// //       link.href = result;
// //       link.download = "virtual-tryon-result.png";
// //       document.body.appendChild(link);
// //       link.click();
// //       document.body.removeChild(link);
// //       toast({
// //         title: "Downloaded",
// //         description: "Your styled look has been saved.",
// //       });
// //     }
// //   };

// //   // Filter clothes by selected style
// //   const filteredClothes = selectedStyle
// //     ? SAMPLE_CLOTHES.filter((c) => c.category === selectedStyle)
// //     : SAMPLE_CLOTHES;

// //   return (
// //     <div className="min-h-screen bg-background">
// //       {/* Header */}
// //       <header className="fixed top-0 left-0 right-0 z-40 glass-strong">
// //         <div className="flex items-center justify-between px-4 md:px-6 py-3 md:py-4">
// //           {/* Back Button */}
// //           <button
// //             onClick={() => navigate("/")}
// //             className="flex items-center gap-2 px-4 py-2 rounded-full hover:bg-accent transition-colors group"
// //           >
// //             <svg
// //               className="w-5 h-5 text-muted-foreground group-hover:-translate-x-1 transition-transform"
// //               fill="none"
// //               viewBox="0 0 24 24"
// //               stroke="currentColor"
// //             >
// //               <path
// //                 strokeLinecap="round"
// //                 strokeLinejoin="round"
// //                 strokeWidth={1.5}
// //                 d="M7 16l-4-4m0 0l4-4m-4 4h18"
// //               />
// //             </svg>
// //             <span className="text-sm text-muted-foreground hidden sm:inline">Back</span>
// //           </button>

// //           {/* Title - Centered */}
// //           <h1 className="text-lg md:text-xl font-display font-light tracking-wide absolute left-1/2 -translate-x-1/2">
// //             Virtual Try-On
// //           </h1>

// //           {/* Settings */}
// //           <button
// //             onClick={() => navigate("/settings")}
// //             className="p-2 rounded-full hover:bg-accent transition-colors"
// //             aria-label="Settings"
// //           >
// //             <Settings className="w-5 h-5 text-muted-foreground" />
// //           </button>
// //         </div>
// //       </header>

// //       {/* Main Content */}
// //       <main className="pt-20 pb-8 px-4 md:px-6 lg:px-8 max-w-7xl mx-auto">
// //         {/* Sample Models Bar - Top */}
// //         <div className="mb-6">
// //           <div className="flex items-center justify-between mb-3">
// //             <h2 className="text-sm font-medium text-muted-foreground">Choose a Model</h2>
// //             <span className="text-xs text-muted-foreground/60">or upload your own</span>
// //           </div>
// //           <div className="flex gap-3 overflow-x-auto pb-2 scrollbar-hide">
// //             {SAMPLE_MODELS.map((model) => (
// //               <button
// //                 key={model.id}
// //                 onClick={() => handleSelectSampleModel(model.image)}
// //                 className={`flex-shrink-0 w-14 h-18 md:w-16 md:h-20 rounded-xl overflow-hidden border-2 transition-all hover:scale-105 ${
// //                   personImage === model.image
// //                     ? "border-primary ring-2 ring-primary/30"
// //                     : "border-border/50 hover:border-foreground/30"
// //                 }`}
// //               >
// //                 <img
// //                   src={model.image}
// //                   alt={model.name}
// //                   className="w-full h-full object-cover"
// //                 />
// //               </button>
// //             ))}
// //             <button
// //               onClick={() => personInputRef.current?.click()}
// //               className="flex-shrink-0 w-14 h-18 md:w-16 md:h-20 rounded-xl border-2 border-dashed border-border/50 hover:border-foreground/30 flex items-center justify-center transition-colors"
// //             >
// //               <Upload className="w-5 h-5 text-muted-foreground" />
// //             </button>
// //           </div>
// //         </div>

// //         {/* Selected Items Row */}
// //         <div className="flex items-center justify-center gap-6 mb-6 p-4 glass rounded-2xl">
// //           {/* Person Thumbnail */}
// //           <div className="flex flex-col items-center gap-2">
// //             <div
// //               onClick={() => personInputRef.current?.click()}
// //               className="w-16 h-16 md:w-20 md:h-20 rounded-xl bg-accent/50 cursor-pointer overflow-hidden flex items-center justify-center hover:bg-accent transition-colors border border-border/50"
// //             >
// //               {personImage ? (
// //                 <img
// //                   src={personImage}
// //                   alt="Person"
// //                   className="w-full h-full object-cover"
// //                 />
// //               ) : (
// //                 <User className="w-6 h-6 text-muted-foreground" />
// //               )}
// //             </div>
// //             <span className="text-xs text-muted-foreground">Model</span>
// //           </div>

// //           {/* Plus icon */}
// //           <div className="text-2xl text-muted-foreground/50">+</div>

// //           {/* Clothing Thumbnail */}
// //           <div className="flex flex-col items-center gap-2">
// //             <div
// //               onClick={() => clothingInputRef.current?.click()}
// //               className="w-16 h-16 md:w-20 md:h-20 rounded-xl bg-accent/50 cursor-pointer overflow-hidden flex items-center justify-center hover:bg-accent transition-colors border border-border/50"
// //             >
// //               {clothingImage ? (
// //                 <img
// //                   src={clothingImage}
// //                   alt="Clothing"
// //                   className="w-full h-full object-cover"
// //                 />
// //               ) : (
// //                 <Shirt className="w-6 h-6 text-muted-foreground" />
// //               )}
// //             </div>
// //             <span className="text-xs text-muted-foreground">Outfit</span>
// //           </div>

// //           {/* Equals icon */}
// //           <div className="text-2xl text-muted-foreground/50">=</div>

// //           {/* Result preview mini */}
// //           <div className="flex flex-col items-center gap-2">
// //             <div className="w-16 h-16 md:w-20 md:h-20 rounded-xl bg-primary/10 border border-primary/30 flex items-center justify-center">
// //               {result ? (
// //                 <img
// //                   src={result}
// //                   alt="Result"
// //                   className="w-full h-full object-cover rounded-xl"
// //                 />
// //               ) : (
// //                 <Sparkles className="w-6 h-6 text-primary/50" />
// //               )}
// //             </div>
// //             <span className="text-xs text-muted-foreground">Result</span>
// //           </div>
// //         </div>

// //         {/* Hidden file inputs */}
// //         <input
// //           ref={personInputRef}
// //           type="file"
// //           accept="image/*"
// //           onChange={(e) => handleImageUpload(e, "person")}
// //           className="hidden"
// //         />
// //         <input
// //           ref={clothingInputRef}
// //           type="file"
// //           accept="image/*"
// //           onChange={(e) => handleImageUpload(e, "clothing")}
// //           className="hidden"
// //         />

// //         {/* Main Grid */}
// //         <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
// //           {/* Left Panel - Uploads */}
// //           <div className="space-y-4">
// //             <div className="glass rounded-2xl p-5">
// //               <h2 className="text-sm font-medium mb-4 flex items-center gap-2">
// //                 <Upload className="w-4 h-4" />
// //                 Upload Images
// //               </h2>

// //               <div className="space-y-3">
// //                 <button
// //                   onClick={() => personInputRef.current?.click()}
// //                   className="w-full p-4 rounded-xl border border-dashed border-border hover:border-foreground/30 transition-colors flex items-center gap-3"
// //                 >
// //                   <div className="w-10 h-10 rounded-lg bg-accent flex items-center justify-center">
// //                     <User className="w-5 h-5 text-muted-foreground" />
// //                   </div>
// //                   <div className="text-left">
// //                     <p className="text-sm font-medium">Person Image</p>
// //                     <p className="text-xs text-muted-foreground">
// //                       {personImage ? "Image uploaded" : "Upload your photo"}
// //                     </p>
// //                   </div>
// //                   {personImage && (
// //                     <button
// //                       onClick={(e) => {
// //                         e.stopPropagation();
// //                         setPersonImage(null);
// //                       }}
// //                       className="ml-auto p-1 hover:bg-destructive/20 rounded"
// //                     >
// //                       <X className="w-4 h-4 text-destructive" />
// //                     </button>
// //                   )}
// //                 </button>

// //                 <button
// //                   onClick={() => clothingInputRef.current?.click()}
// //                   className="w-full p-4 rounded-xl border border-dashed border-border hover:border-foreground/30 transition-colors flex items-center gap-3"
// //                 >
// //                   <div className="w-10 h-10 rounded-lg bg-accent flex items-center justify-center">
// //                     <Shirt className="w-5 h-5 text-muted-foreground" />
// //                   </div>
// //                   <div className="text-left">
// //                     <p className="text-sm font-medium">Clothing Image</p>
// //                     <p className="text-xs text-muted-foreground">
// //                       {clothingImage ? "Image uploaded" : "Upload outfit"}
// //                     </p>
// //                   </div>
// //                   {clothingImage && (
// //                     <button
// //                       onClick={(e) => {
// //                         e.stopPropagation();
// //                         setClothingImage(null);
// //                       }}
// //                       className="ml-auto p-1 hover:bg-destructive/20 rounded"
// //                     >
// //                       <X className="w-4 h-4 text-destructive" />
// //                     </button>
// //                   )}
// //                 </button>
// //               </div>
// //             </div>
// //           </div>

// //           {/* Center - Preview */}
// //           <div className="lg:col-span-1">
// //             <div className="glass rounded-2xl p-5 h-full min-h-[400px] flex flex-col">
// //               <h2 className="text-sm font-medium mb-4 text-center">Preview</h2>

// //               <div className="flex-1 flex items-center justify-center rounded-xl border border-dashed border-border/50 overflow-hidden bg-muted/20">
// //                 {generating ? (
// //                   <div className="text-center">
// //                     <Loader2 className="w-10 h-10 animate-spin text-muted-foreground mx-auto mb-3" />
// //                     <p className="text-sm text-muted-foreground">
// //                       Generating your look...
// //                     </p>
// //                   </div>
// //                 ) : result ? (
// //                   <img
// //                     src={result}
// //                     alt="Generated result"
// //                     className="w-full h-full object-contain"
// //                   />
// //                 ) : (
// //                   <div className="text-center p-8">
// //                     <Sparkles className="w-10 h-10 text-muted-foreground/50 mx-auto mb-3" />
// //                     <p className="text-sm text-muted-foreground">
// //                       Your styled look will appear here
// //                     </p>
// //                   </div>
// //                 )}
// //               </div>

// //               {result && (
// //                 <Button
// //                   onClick={handleDownload}
// //                   variant="outline"
// //                   className="mt-4 w-full"
// //                 >
// //                   <Download className="w-4 h-4 mr-2" />
// //                   Download
// //                 </Button>
// //               )}
// //             </div>
// //           </div>

// //           {/* Right Panel - AI Styling + Clothes */}
// //           <div className="space-y-4">
// //             {/* AI Styling */}
// //             <div className="glass rounded-2xl p-5">
// //               <h2 className="text-sm font-medium mb-4 flex items-center gap-2">
// //                 <Sparkles className="w-4 h-4" />
// //                 AI Styling
// //               </h2>

// //               <Textarea
// //                 value={prompt}
// //                 onChange={(e) => setPrompt(e.target.value)}
// //                 placeholder="Describe the outfit style, fit, color, or design..."
// //                 className="min-h-[80px] bg-background/50 border-border/50 resize-none mb-4"
// //               />

// //               <div className="mb-4">
// //                 <p className="text-xs text-muted-foreground mb-2">
// //                   Quick Styles
// //                 </p>
// //                 <div className="flex flex-wrap gap-2">
// //                   {PRESET_STYLES.map((style) => (
// //                     <button
// //                       key={style}
// //                       onClick={() =>
// //                         setSelectedStyle(
// //                           selectedStyle === style ? null : style
// //                         )
// //                       }
// //                       className={`px-3 py-1.5 text-xs rounded-full border transition-colors ${
// //                         selectedStyle === style
// //                           ? "bg-foreground text-background border-foreground"
// //                           : "border-border hover:border-foreground/50"
// //                       }`}
// //                     >
// //                       {style}
// //                     </button>
// //                   ))}
// //                 </div>
// //               </div>

// //               <Button
// //                 onClick={handleGenerate}
// //                 disabled={generating}
// //                 className="w-full glow-button"
// //               >
// //                 {generating ? (
// //                   <>
// //                     <Loader2 className="w-4 h-4 mr-2 animate-spin" />
// //                     Generating...
// //                   </>
// //                 ) : (
// //                   <>
// //                     <Sparkles className="w-4 h-4 mr-2" />
// //                     Generate Look
// //                   </>
// //                 )}
// //               </Button>
// //             </div>

// //             {/* Sample Clothes */}
// //             <div className="glass rounded-2xl p-5">
// //               <div className="flex items-center justify-between mb-4">
// //                 <h2 className="text-sm font-medium flex items-center gap-2">
// //                   <Shirt className="w-4 h-4" />
// //                   Sample Clothes
// //                 </h2>
// //                 {selectedStyle && (
// //                   <span className="text-xs px-2 py-1 rounded-full bg-foreground/10 text-muted-foreground">
// //                     {selectedStyle}
// //                   </span>
// //                 )}
// //               </div>

// //               <div className="grid grid-cols-3 gap-2">
// //                 {filteredClothes.map((item) => (
// //                   <button
// //                     key={item.id}
// //                     onClick={() => handleSelectSampleClothing(item)}
// //                     className={`aspect-[3/4] rounded-lg overflow-hidden border-2 transition-all hover:scale-105 ${
// //                       clothingImage === item.image
// //                         ? "border-primary ring-2 ring-primary/30"
// //                         : "border-border/30 hover:border-foreground/30"
// //                     }`}
// //                   >
// //                     <img
// //                       src={item.image}
// //                       alt={item.name}
// //                       className="w-full h-full object-cover"
// //                     />
// //                   </button>
// //                 ))}
// //               </div>
// //               <p className="text-xs text-muted-foreground/60 mt-3 text-center">
// //                 Click to select or upload your own
// //               </p>
// //             </div>

// //             {/* AI Suggestion */}
// //             {aiSuggestion && (
// //               <div className="glass rounded-2xl p-5 animate-fade-in">
// //                 <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
// //                   <Sparkles className="w-4 h-4 text-primary" />
// //                   AI Stylist Says
// //                 </h3>
// //                 <p className="text-sm text-muted-foreground leading-relaxed">
// //                   {aiSuggestion}
// //                 </p>
// //               </div>
// //             )}
// //           </div>
// //         </div>
// //       </main>
// //     </div>
// //   );
// // };

// // export default TryOn;

// import { useState, useRef } from "react";
// import { Button } from "@/components/ui/button";
// import { Textarea } from "@/components/ui/textarea";
// import {
//   Upload,
//   Sparkles,
//   User,
//   Shirt,
//   X,
// } from "lucide-react";

// const SAMPLE_MODELS = [
//   { id: 1, image: "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=200" },
//   { id: 2, image: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=200" },
//   { id: 3, image: "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=200" },
// ];

// const SAMPLE_CLOTHES = [
//   { id: 1, image: "https://images.unsplash.com/photo-1594938298603-c8148c4dae35?w=200" },
//   { id: 2, image: "https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?w=200" },
//   { id: 3, image: "https://images.unsplash.com/photo-1551028719-00167b16eac5?w=200" },
// ];

// export default function TryOn() {
//   const [personImage, setPersonImage] = useState<string | null>(null);
//   const [clothingImage, setClothingImage] = useState<string | null>(null);
//   const [prompt, setPrompt] = useState("");
//   const [result, setResult] = useState<string | null>(null);

//   const personRef = useRef<HTMLInputElement>(null);
//   const clothRef = useRef<HTMLInputElement>(null);

//   const handleUpload = (
//     e: React.ChangeEvent<HTMLInputElement>,
//     type: "person" | "cloth"
//   ) => {
//     const file = e.target.files?.[0];
//     if (!file) return;

//     const reader = new FileReader();
//     reader.onload = () => {
//       if (type === "person") setPersonImage(reader.result as string);
//       else setClothingImage(reader.result as string);
//     };
//     reader.readAsDataURL(file);
//   };

//   const generatePreview = () => {
//     // TEMP LOGIC (frontend only)
//     setResult(personImage || clothingImage || null);
//   };

//   return (
//     <div className="min-h-screen bg-background px-4 py-6 max-w-6xl mx-auto">

//       {/* HEADER */}
//       <h1 className="text-2xl font-semibold text-center mb-6">
//         Virtual Try-On
//       </h1>

//       {/* STEP 1: MODEL SELECTION */}
//       <div className="mb-6">
//         <h2 className="text-sm text-muted-foreground mb-2">Choose Model</h2>

//         <div className="flex gap-3 overflow-x-auto">
//           {SAMPLE_MODELS.map((m) => (
//             <img
//               key={m.id}
//               src={m.image}
//               onClick={() => setPersonImage(m.image)}
//               className={`w-16 h-20 rounded-lg object-cover cursor-pointer border-2 ${
//                 personImage === m.image ? "border-primary" : "border-transparent"
//               }`}
//             />
//           ))}

//           <button
//             onClick={() => personRef.current?.click()}
//             className="w-16 h-20 border-dashed border flex items-center justify-center rounded-lg"
//           >
//             <Upload size={18} />
//           </button>
//         </div>
//       </div>

//       {/* STEP 2: CLOTHING */}
//       <div className="mb-6">
//         <h2 className="text-sm text-muted-foreground mb-2">Choose Outfit</h2>

//         <div className="flex gap-3 overflow-x-auto">
//           {SAMPLE_CLOTHES.map((c) => (
//             <img
//               key={c.id}
//               src={c.image}
//               onClick={() => setClothingImage(c.image)}
//               className={`w-16 h-20 rounded-lg object-cover cursor-pointer border-2 ${
//                 clothingImage === c.image ? "border-primary" : "border-transparent"
//               }`}
//             />
//           ))}

//           <button
//             onClick={() => clothRef.current?.click()}
//             className="w-16 h-20 border-dashed border flex items-center justify-center rounded-lg"
//           >
//             <Upload size={18} />
//           </button>
//         </div>
//       </div>

//       {/* STEP 3: PROMPT */}
//       <div className="mb-6">
//         <Textarea
//           placeholder="Describe style (optional)..."
//           value={prompt}
//           onChange={(e) => setPrompt(e.target.value)}
//         />
//       </div>

//       {/* PREVIEW AREA */}
//       <div className="mb-6 border rounded-xl h-64 flex items-center justify-center bg-muted/20">
//         {result ? (
//           <img src={result} className="h-full object-contain" />
//         ) : (
//           <div className="text-center text-muted-foreground">
//             <Sparkles className="mx-auto mb-2" />
//             Preview will appear here
//           </div>
//         )}
//       </div>

//       {/* ACTION */}
//       <Button onClick={generatePreview} className="w-full">
//         Generate Look
//       </Button>

//       {/* HIDDEN INPUTS */}
//       <input
//         ref={personRef}
//         type="file"
//         hidden
//         onChange={(e) => handleUpload(e, "person")}
//       />
//       <input
//         ref={clothRef}
//         type="file"
//         hidden
//         onChange={(e) => handleUpload(e, "cloth")}
//       />
//     </div>
//   );
// }

// import { useState, useRef } from "react";
// import { Button } from "@/components/ui/button";
// import { Upload, Sparkles, User, Shirt } from "lucide-react";

// const SAMPLE_MODELS = [
//   { id: 1, image: "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=200" },
//   { id: 2, image: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=200" },
//   { id: 3, image: "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=200" },
// ];

// const SAMPLE_CLOTHES = [
//   { id: 1, image: "https://images.unsplash.com/photo-1594938298603-c8148c4dae35?w=200" },
//   { id: 2, image: "https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?w=200" },
//   { id: 3, image: "https://images.unsplash.com/photo-1551028719-00167b16eac5?w=200" },
// ];

// export default function TryOn() {
//   const [personImage, setPersonImage] = useState<string | null>(null);
//   const [clothingImage, setClothingImage] = useState<string | null>(null);
//   const [result, setResult] = useState<string | null>(null);

//   const personRef = useRef<HTMLInputElement>(null);
//   const clothRef = useRef<HTMLInputElement>(null);

//   const handleUpload = (e: React.ChangeEvent<HTMLInputElement>, type: "person" | "cloth") => {
//     const file = e.target.files?.[0];
//     if (!file) return;
//     const reader = new FileReader();
//     reader.onload = () => {
//       if (type === "person") setPersonImage(reader.result as string);
//       else setClothingImage(reader.result as string);
//     };
//     reader.readAsDataURL(file);
//   };

//   const generatePreview = () => {
//     if (!personImage && !clothingImage) return;
//     setResult(personImage || clothingImage || null);
//   };

//   return (
//     <div className="min-h-screen bg-background px-4 py-6 max-w-6xl mx-auto">
//       <h1 className="text-2xl font-semibold text-center mb-6">Virtual Try-On</h1>

//       {/* TOP SELECTION BARS */}
//       <div className="mb-6">
//         <h2 className="text-sm text-muted-foreground mb-2">Choose Model</h2>
//         <div className="flex gap-3 overflow-x-auto mb-4">
//           {SAMPLE_MODELS.map((m) => (
//             <img
//               key={m.id}
//               src={m.image}
//               onClick={() => setPersonImage(m.image)}
//               className={`w-20 h-28 rounded-lg object-cover cursor-pointer border-2 ${
//                 personImage === m.image ? "border-primary" : "border-transparent"
//               }`}
//             />
//           ))}
//           <button
//             onClick={() => personRef.current?.click()}
//             className="w-20 h-28 border-dashed border flex items-center justify-center rounded-lg"
//           >
//             <Upload size={24} />
//           </button>
//         </div>

//         <h2 className="text-sm text-muted-foreground mb-2">Choose Outfit</h2>
//         <div className="flex gap-3 overflow-x-auto">
//           {SAMPLE_CLOTHES.map((c) => (
//             <img
//               key={c.id}
//               src={c.image}
//               onClick={() => setClothingImage(c.image)}
//               className={`w-20 h-28 rounded-lg object-cover cursor-pointer border-2 ${
//                 clothingImage === c.image ? "border-primary" : "border-transparent"
//               }`}
//             />
//           ))}
//           <button
//             onClick={() => clothRef.current?.click()}
//             className="w-20 h-28 border-dashed border flex items-center justify-center rounded-lg"
//           >
//             <Upload size={24} />
//           </button>
//         </div>
//       </div>

//       {/* MINI PREVIEW BAR (Model + Clothing) */}
//       {(personImage || clothingImage) && (
//         <div className="flex items-center justify-center gap-4 mb-6">
//           {/* Model */}
//           <div className="w-24 h-32 rounded-lg border overflow-hidden flex items-center justify-center">
//             {personImage ? <img src={personImage} className="w-full h-full object-cover" /> : <User size={24} />}
//           </div>
//           <span className="text-2xl">+</span>
//           {/* Clothing */}
//           <div className="w-24 h-32 rounded-lg border overflow-hidden flex items-center justify-center">
//             {clothingImage ? <img src={clothingImage} className="w-full h-full object-cover" /> : <Shirt size={24} />}
//           </div>
//         </div>
//       )}

//       {/* GENERATE BUTTON */}
//       <Button onClick={generatePreview} className="w-full mb-6">
//         Generate Look
//       </Button>

//       {/* BIG PREVIEW AREA */}
//       <div className="border rounded-xl h-96 flex items-center justify-center bg-muted/20">
//         {result ? (
//           <div className="flex items-center justify-center gap-4">
//             {personImage && <img src={personImage} className="h-full object-contain rounded-lg" />}
//             {clothingImage && <img src={clothingImage} className="h-full object-contain rounded-lg" />}
//           </div>
//         ) : (
//           <div className="text-center text-muted-foreground">
//             <Sparkles className="mx-auto mb-2" size={36} />
//             Your styled look will appear here
//           </div>
//         )}
//       </div>

//       {/* HIDDEN INPUTS */}
//       <input ref={personRef} type="file" hidden onChange={(e) => handleUpload(e, "person")} />
//       <input ref={clothRef} type="file" hidden onChange={(e) => handleUpload(e, "cloth")} />
//     </div>
//   );
// }

import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Upload, Sparkles, User, Shirt } from "lucide-react";

const SAMPLE_MODELS = [
  { id: 1, image: "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=200" },
  { id: 2, image: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=200" },
  { id: 3, image: "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=200" },
];

const SAMPLE_CLOTHES = [
  { id: 1, image: "https://images.unsplash.com/photo-1594938298603-c8148c4dae35?w=200" },
  { id: 2, image: "https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?w=200" },
  { id: 3, image: "https://images.unsplash.com/photo-1551028719-00167b16eac5?w=200" },
];

export default function TryOn() {
  const [personImage, setPersonImage] = useState<string | null>(null);
  const [clothingImage, setClothingImage] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);

  const personRef = useRef<HTMLInputElement>(null);
  const clothRef = useRef<HTMLInputElement>(null);

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>, type: "person" | "cloth") => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      if (type === "person") setPersonImage(reader.result as string);
      else setClothingImage(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const generatePreview = () => {
    if (!personImage && !clothingImage) return;
    setResult(personImage || clothingImage || null);
  };

  return (
    <div className="min-h-screen bg-background px-4 py-6 max-w-6xl mx-auto">
      <h1 className="text-2xl font-semibold text-center mb-6">Virtual Try-On</h1>

      {/* TOP SELECTION BARS */}
      <div className="mb-6">
        <h2 className="text-sm text-muted-foreground mb-2">Choose Model</h2>
        <div className="flex gap-3 overflow-x-auto mb-4">
          {SAMPLE_MODELS.map((m) => (
            <img
              key={m.id}
              src={m.image}
              onClick={() => setPersonImage(m.image)}
              className={`w-20 h-28 rounded-lg object-cover cursor-pointer border-2 ${
                personImage === m.image ? "border-primary" : "border-transparent"
              }`}
            />
          ))}
          <button
            onClick={() => personRef.current?.click()}
            className="w-20 h-28 border-dashed border flex items-center justify-center rounded-lg"
          >
            <Upload size={24} />
          </button>
        </div>

        <h2 className="text-sm text-muted-foreground mb-2">Choose Outfit</h2>
        <div className="flex gap-3 overflow-x-auto">
          {SAMPLE_CLOTHES.map((c) => (
            <img
              key={c.id}
              src={c.image}
              onClick={() => setClothingImage(c.image)}
              className={`w-20 h-28 rounded-lg object-cover cursor-pointer border-2 ${
                clothingImage === c.image ? "border-primary" : "border-transparent"
              }`}
            />
          ))}
          <button
            onClick={() => clothRef.current?.click()}
            className="w-20 h-28 border-dashed border flex items-center justify-center rounded-lg"
          >
            <Upload size={24} />
          </button>
        </div>
      </div>

      {/* MINI PREVIEW BAR (Always Visible) */}
      <div className="flex items-center justify-center gap-4 mb-6">
        {/* Model */}
        <div className="w-24 h-32 rounded-lg border overflow-hidden flex items-center justify-center bg-background/50">
          {personImage ? (
            <img src={personImage} className="w-full h-full object-cover" />
          ) : (
            <User size={24} className="text-muted-foreground" />
          )}
        </div>
        <span className="text-2xl">+</span>
        {/* Clothing */}
        <div className="w-24 h-32 rounded-lg border overflow-hidden flex items-center justify-center bg-background/50">
          {clothingImage ? (
            <img src={clothingImage} className="w-full h-full object-cover" />
          ) : (
            <Shirt size={24} className="text-muted-foreground" />
          )}
        </div>
      </div>

      {/* GENERATE BUTTON */}
      <Button onClick={generatePreview} className="w-full mb-6">
        Generate Look
      </Button>

      {/* BIG PREVIEW AREA */}
      <div className="border rounded-xl h-96 flex items-center justify-center bg-muted/20">
        {result ? (
          <div className="flex items-center justify-center gap-4">
            {personImage && <img src={personImage} className="h-full object-contain rounded-lg" />}
            {clothingImage && <img src={clothingImage} className="h-full object-contain rounded-lg" />}
          </div>
        ) : (
          <div className="text-center text-muted-foreground">
            <Sparkles className="mx-auto mb-2" size={36} />
            Your styled look will appear here
          </div>
        )}
      </div>

      {/* HIDDEN INPUTS */}
      <input ref={personRef} type="file" hidden onChange={(e) => handleUpload(e, "person")} />
      <input ref={clothRef} type="file" hidden onChange={(e) => handleUpload(e, "cloth")} />
    </div>
  );
}