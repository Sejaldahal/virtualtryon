import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { prompt, hasPersonImage, hasClothingImage } = await req.json();
    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    
    if (!LOVABLE_API_KEY) {
      console.error("LOVABLE_API_KEY is not configured");
      throw new Error("AI service is not configured");
    }

    console.log("Generating style advice for prompt:", prompt);
    console.log("Has person image:", hasPersonImage);
    console.log("Has clothing image:", hasClothingImage);

    const systemPrompt = `You are an expert fashion stylist AI assistant. Your role is to provide personalized, trendy, and practical fashion advice.

When giving suggestions:
- Be specific about colors, patterns, and materials
- Consider body types and occasions
- Suggest accessories that complement the outfit
- Provide tips on how to style the look
- Keep responses concise but helpful (2-3 paragraphs max)

If the user has uploaded images, acknowledge them and provide relevant styling tips.`;

    const userPrompt = `${hasPersonImage ? "The user has uploaded their photo. " : ""}${hasClothingImage ? "The user has uploaded a clothing item they want to style. " : ""}

User's request: ${prompt || "Please suggest a fashionable outfit."}

Provide styling advice and outfit suggestions based on their preferences.`;

    const response = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-2.5-flash",
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt },
        ],
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("AI Gateway error:", response.status, errorText);
      
      if (response.status === 429) {
        return new Response(
          JSON.stringify({ error: "Rate limit exceeded. Please try again in a moment." }),
          { status: 429, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }
      
      if (response.status === 402) {
        return new Response(
          JSON.stringify({ error: "AI credits depleted. Please add funds to continue." }),
          { status: 402, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }
      
      throw new Error("Failed to generate style advice");
    }

    const data = await response.json();
    const suggestion = data.choices?.[0]?.message?.content;

    console.log("Generated suggestion successfully");

    return new Response(
      JSON.stringify({ suggestion }),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  } catch (error) {
    console.error("Style advisor error:", error);
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : "An error occurred" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
