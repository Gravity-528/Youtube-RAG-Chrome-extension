chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "videoId") {
        // console.log("Content script received videoId:", request.videoId);
        // console.log("Sender:", sender);
        // console.log("Request:", request);
        // console.log("Sender tab ID:", sender.tab ? sender.tab.id : "No tab info");
        
        sendResponse({ status: "success", videoId: request.videoId });
    }else if(request.action === "others"){
        sendResponse({ status: "success", url: request.url });
    } else {
        sendResponse({ status: "error", message: "Unknown action" });
    }
})