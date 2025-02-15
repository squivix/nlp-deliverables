// background.js

chrome.runtime.onMessage.addListener(async (message, sender, sendResponse) => {
  if (message.type === "CHECK_SPOILER") {
    try {
      const textToCheck = message.text.toLowerCase().trim();

      // 1) 如果包含 "spoiler" 且不包含 "not spoiler"/"no spoilers" 等等
      if (isDefinitelySpoiler(textToCheck)) {
        // 不调用OpenAI，直接返回true
        console.log("[Short-Circuit] 'spoiler' found => definitely spoiler");
        return sendResponse({ isSpoiler: true });
      }

      // 2) 从存储里读配置：spoilerLevel, customKeywords
      chrome.storage.sync.get(["spoilerLevel", "customKeywords"], async (data) => {
        const level = data.spoilerLevel || "medium"; 
        const userKeywords = data.customKeywords || [];

        // 如果文本里不含任意用户关键词，也跳过OpenAI
        // （说明既没包含"spoiler"也没包含用户关键词）
        if (!containsAnyUserKeyword(textToCheck, userKeywords)) {
          console.log("[Filter] No matching user keywords, skipping API");
          return sendResponse({ isSpoiler: false });
        }

        // 否则调用OpenAI
        const isSpoiler = await checkSpoilerWithOpenAI(textToCheck, level);
        console.log(`[OpenAI] Spoiler: ${isSpoiler}`);
        sendResponse({ isSpoiler });
      });
    } catch (error) {
      console.error("Error:", error);
      sendResponse({ isSpoiler: false });
    }
    return true; // 表示异步响应
  }
});

/** 
 * 如果文本里出现 'spoiler' 且没有出现 'not spoiler' / 'no spoilers' 等反例 
 * 就直接判定是剧透 
 */
function isDefinitelySpoiler(lowerText) {
  if (lowerText.includes("spoiler")) {
    // 如果用户也担心 "no spoiler"、"not spoiler"、"spoiler free" 这种情况，可以排除
    if (
      lowerText.includes("not spoiler") || 
      lowerText.includes("no spoiler") ||
      lowerText.includes("spoiler-free")
    ) {
      // 这里可根据需求做精细判断
      return false; 
    }
    return true;
  }
  return false;
}

/** 如果文本里匹配到任意一个用户自定义关键词 */
function containsAnyUserKeyword(lowerText, keywords) {
  // keywords 形如 ["dies", "killed", "ending" ...]
  return keywords.some(kw => lowerText.includes(kw.toLowerCase()));
}

/** 调用 OpenAI API 进行更智能的剧透判断 */
async function checkSpoilerWithOpenAI(text, level) {
  const OPENAI_API_KEY = "YOUR_OPENAI_API_KEY";
  const OPENAI_API_URL = "https://api.openai.com/v1/chat/completions";

  const prompt = buildPrompt(text, level);

  const response = await fetch(OPENAI_API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${OPENAI_API_KEY}`
    },
    body: JSON.stringify({
      model: "gpt-3.5-turbo",
      messages: [{ role: "user", content: prompt }],
      max_tokens: 10,
      temperature: 0
    })
  });

  if (!response.ok) {
    throw new Error(`OpenAI error: ${response.status}`);
  }

  const data = await response.json();
  const reply = data.choices?.[0]?.message?.content?.trim()?.toUpperCase() || "";
  return reply.includes("YES");
}

function buildPrompt(text, level) {
  // 这里和你原来的level逻辑一致
  let instructions = "";
  if (level === "low") {
    instructions = `
Consider it a spoiler only if it reveals a major twist, ending, or critical character death.
If so, respond ONLY "YES". Otherwise, "NO".
`;
  } else if (level === "high") {
    instructions = `
Consider it a spoiler even if it reveals minor plot details or any significant storyline elements.
If so, respond ONLY "YES". Otherwise, "NO".
`;
  } else {
    // medium
    instructions = `
Consider it a spoiler if it reveals important plot developments, major events, or endings.
If so, respond ONLY "YES". Otherwise, "NO".
`;
  }

  return `
You are a helpful assistant.
We want to detect spoilers in the following text from user input.
${instructions}
Text:
${text}
`;
}
