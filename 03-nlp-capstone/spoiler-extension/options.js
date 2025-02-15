// options.js

document.addEventListener("DOMContentLoaded", () => {
    const spoilerLevelSelect = document.getElementById("spoiler-level");
    const keywordsTextarea = document.getElementById("keywords");
    const saveBtn = document.getElementById("save-btn");
    const statusText = document.getElementById("status");
  
    // 1. 先从 storage 里加载设置
    chrome.storage.sync.get(["spoilerLevel", "customKeywords"], (data) => {
      if (data.spoilerLevel) {
        spoilerLevelSelect.value = data.spoilerLevel;
      }
      if (data.customKeywords) {
        keywordsTextarea.value = data.customKeywords.join(", ");
      }
    });
  
    // 2. 点击保存时，写回 storage
    saveBtn.addEventListener("click", () => {
      const level = spoilerLevelSelect.value;
      // 将逗号分隔的关键字转成数组
      const keywordsArr = keywordsTextarea.value
        .split(",")
        .map((kw) => kw.trim())
        .filter((kw) => kw.length > 0);
  
      chrome.storage.sync.set({
        spoilerLevel: level,
        customKeywords: keywordsArr
      }, () => {
        statusText.innerText = "Settings saved!";
        setTimeout(() => {
          statusText.innerText = "";
        }, 2000);
      });
    });
  });
  