// content.js

let detectionEnabled = false;
let searchInputListener = null;

// 用于记录已判定过的搜索字符串 -> boolean(是否剧透)
const processedSearches = new Map();

// 1. 读取用户设置
chrome.storage.sync.get(["searchDetectionEnabled"], (data) => {
  detectionEnabled = data.searchDetectionEnabled || false;
  if (detectionEnabled) {
    startListeningToSearchInput();
  }
});

// 2. 监听存储变化
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "sync" && changes.searchDetectionEnabled) {
    detectionEnabled = changes.searchDetectionEnabled.newValue;
    if (detectionEnabled) {
      startListeningToSearchInput();
    } else {
      stopListeningToSearchInput();
      hideSpoilerWarning();
    }
  }
});

/** 开始监听 */
function startListeningToSearchInput() {
  const searchInput = document.querySelector('input[name="q"], input[type="search"]');
  if (!searchInput) return;

  // 避免重复绑定
  if (searchInputListener) {
    searchInput.removeEventListener("input", searchInputListener);
  }

  searchInputListener = (e) => {
    const searchText = e.target.value.trim();
    if (!detectionEnabled || !searchText) {
      hideSpoilerWarning();
      return;
    }

    // 如果已经处理过此搜索文本，就直接用之前的结果
    if (processedSearches.has(searchText)) {
      const wasSpoiler = processedSearches.get(searchText);
      if (wasSpoiler) showSpoilerWarning();
      else hideSpoilerWarning();
      return;
    }

    // 否则，向 background 发消息
    chrome.runtime.sendMessage({ type: "CHECK_SPOILER", text: searchText }, (res) => {
      const isSpoiler = !!res?.isSpoiler;
      processedSearches.set(searchText, isSpoiler);

      if (isSpoiler) {
        showSpoilerWarning();
      } else {
        hideSpoilerWarning();
      }
    });
  };

  searchInput.addEventListener("input", searchInputListener);
}

/** 停止监听 */
function stopListeningToSearchInput() {
  const searchInput = document.querySelector('input[name="q"], input[type="search"]');
  if (searchInput && searchInputListener) {
    searchInput.removeEventListener("input", searchInputListener);
  }
  searchInputListener = null;
  hideSpoilerWarning();
}

/** 显示红色Banner */
function showSpoilerWarning() {
  if (document.getElementById("spoiler-warning-banner")) return;
  const banner = document.createElement("div");
  banner.id = "spoiler-warning-banner";
  banner.innerText = "🚨 Potential Spoiler Detected! 🚨";
  banner.style.position = "fixed";
  banner.style.top = "0";
  banner.style.left = "0";
  banner.style.width = "100%";
  banner.style.padding = "10px";
  banner.style.backgroundColor = "red";
  banner.style.color = "white";
  banner.style.fontWeight = "bold";
  banner.style.textAlign = "center";
  banner.style.zIndex = "999999";
  document.body.appendChild(banner);
}

/** 移除Banner */
function hideSpoilerWarning() {
  const banner = document.getElementById("spoiler-warning-banner");
  if (banner) banner.remove();
}
