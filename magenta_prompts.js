// magenta_prompts.js
// Minimal browser port of MagentaPrompts (Swift) with single-word vibes,
// descriptive instruments, and micro-genre leaning. Outputs 1–3 words.

export const MagentaPrompts = (() => {
  // ---- Base pools ----
  const instruments = [
    "electric guitar","acoustic guitar","flamenco guitar","bass guitar",
    "electric piano","grand piano","synth lead","synth arpeggio",
    "violin","cello","trumpet","saxophone","clarinet",
    "drums","808 drums","live drums",
    "strings","brass section","hammond organ","wurlitzer","moog bass","analog synth"
  ];

  // single-word vibes only (kept surprising like 'warmup')
  const vibes = [
    "warmup","afterglow","sunrise","midnight","dusk","twilight","daybreak","nocturne","aurora","ember",
    "neon","chrome","velvet","glass","granite","desert","oceanic","skyline","underground","warehouse",
    "dreamy","nostalgic","moody","uplifting","mysterious","energetic","chill","dark","bright","atmospheric",
    "spacey","groovy","ethereal","glitchy","dusty","tape","vintage","hazy","crystalline","shimmer",
    "magnetic","luminous","starlit","shadow","smolder","static","drift","bloom","horizon"
  ];

  const genres = [
    "synthwave","death metal","lofi hiphop","acid house","techno","ambient",
    "jazz","blues","rock","pop","electronic","hip hop","reggae","folk",
    "classical","funk","soul","disco","dubstep","drum and bass","trance","garage"
  ];

  const microGenres = [
    "breakbeat","boom bap","uk garage","two step","dub techno","deep house",
    "lofi house","minimal techno","progressive house","psytrance","goa",
    "liquid dnb","neurofunk","glitch hop","idm","electro","footwork",
    "phonk","dark trap","hyperpop","darksynth","chillwave","vaporwave","future garage"
  ];

  const genreQualifiers = ["deep","dub","dark","melodic","minimal","uplifting","lofi","industrial","retro","neo"];

  const genericTechniques = ["arpeggio","ostinato","staccato","legato","tremolo","harmonics","plucks","pad","chops"];

  const instrumentDescriptors = {
    "electric guitar": ["palm-muted","tremolo","shoegaze","chorused","lead","octave"],
    "acoustic guitar": ["fingerstyle","nylon","arpeggio","strummed"],
    "flamenco guitar": ["rasgueado","picado"],
    "bass guitar": ["slap","picked","sub","syncopated"],
    "moog bass": ["sub","resonant","rubbery"],
    "analog synth": ["pad","plucks","supersaw","arpeggio"],
    "synth lead": ["portamento","supersaw","mono"],
    "electric piano": ["rhodes","chorused","tine"],
    "wurlitzer": ["dirty","tremolo"],
    "grand piano": ["felt","upright","arpeggio"],
    "hammond organ": ["leslie","drawbar"],
    "strings": ["pizzicato","ostinato","legato"],
    "violin": ["pizzicato","legato","tremolo"],
    "cello": ["sul tasto","legato","pizzicato"],
    "trumpet": ["muted","harmon"],
    "saxophone": ["breathy","subtone"],
    "clarinet": ["staccato","legato"],
    "drums": ["brushed","breakbeat","rimshot"],
    "808 drums": ["808","trap"],
    "live drums": ["brushed","tight","roomy"],
    "brass section": ["stabs","swell"]
  };

  // ---- cycling state ----
  const categories = ["instrument","vibe","genre"];
  let currentCategoryIndex = 0;

  // ---- public API ----
  function getNextCyclingStyle() {
    const cat = categories[currentCategoryIndex];
    currentCategoryIndex = (currentCategoryIndex + 1) % categories.length;
    if (cat === "instrument") return getRandomInstrument();
    if (cat === "vibe")       return getRandomVibe();
    return getRandomGenre();
  }
  function getRandomInstrument() {
    const inst = oneOf(instruments) ?? "electric guitar";
    if (chance(0.45)) {
      const specific = oneOf(instrumentDescriptors[inst] || []);
      const tech = specific || oneOf(genericTechniques) || "arpeggio";
      return clip1to3([tech, inst]);
    }
    return inst;
  }
  function getRandomVibe() {
    return oneOf(vibes) ?? "warmup"; // single word only
  }
  function getRandomGenre() {
    if (chance(0.65)) {
      return oneOf(microGenres) ?? "breakbeat";
    } else {
      const pool = genres.filter(g => g.toLowerCase() !== "jazz"); // avoid pure "jazz" on dice
      const base = oneOf(pool) ?? "electronic";
      if (chance(0.30)) {
        const q = oneOf(genreQualifiers) ?? "deep";
        return clip1to3([q, base]);
      }
      return base;
    }
  }
  function getRandomStyle() {
    const r = Math.floor(Math.random()*3);
    return r===0 ? getRandomInstrument() : r===1 ? getRandomVibe() : getRandomGenre();
  }
  function resetCycle(){ currentCategoryIndex = 0; }

  // ---- helpers ----
  const chance = p => Math.random() < Math.max(0, Math.min(1, p));
  const oneOf = arr => (arr && arr.length) ? arr[Math.floor(Math.random()*arr.length)] : undefined;
  const clip1to3 = (words, max=3) =>
    words.flatMap(w => String(w).split(" ")).slice(0, max).join(" ");

  return { getNextCyclingStyle, getRandomStyle, resetCycle };
})();
