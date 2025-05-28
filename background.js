chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status === "complete" && tab.url && tab.url.includes("youtube.com/watch")) {
        console.log("Back scr", tab.url);

        const urlParams = new URLSearchParams(new URL(tab.url).search);
        const videoId = urlParams.get("v");

        if (videoId) {
            console.log("Extracted videoid", videoId);
        }
    }
});
