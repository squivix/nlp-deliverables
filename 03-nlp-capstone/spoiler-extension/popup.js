// popup.js

document.addEventListener("DOMContentLoaded", () => {
    const enableDetectCheckbox = document.getElementById("enableDetect");
    const spoilerLevelSelect = document.getElementById("spoilerLevel");
    const keywordsInput = document.getElementById("keywords");
    const saveBtn = document.getElementById("saveBtn");
    const statusText = document.getElementById("status");
  
    // 1. 读取并填充UI
    chrome.storage.sync.get(["searchDetectionEnabled", "spoilerLevel", "customKeywords"], (data) => {
      enableDetectCheckbox.checked = data.searchDetectionEnabled || false;
      spoilerLevelSelect.value = data.spoilerLevel || "medium";
      if (data.customKeywords) {
        keywordsInput.value = data.customKeywords.join(", ");
      }
    });
  
    // 2. 用户点击启用检测
    enableDetectCheckbox.addEventListener("change", () => {
      chrome.storage.sync.set({ searchDetectionEnabled: enableDetectCheckbox.checked });
    });
  
    // 3. 点击保存按钮
    saveBtn.addEventListener("click", () => {
      const selectedLevel = spoilerLevelSelect.value;
      const keywordsArr = keywordsInput.value
        .split(",")
        .map((kw) => kw.trim())
        .filter(Boolean);
  
      chrome.storage.sync.set({
        spoilerLevel: selectedLevel,
        customKeywords: keywordsArr
      }, () => {
        statusText.textContent = "Settings saved!";
        setTimeout(() => { statusText.textContent = ""; }, 2000);
      });
    });
  });
  